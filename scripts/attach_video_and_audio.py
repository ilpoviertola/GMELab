"""This script attaches video the generated audio. """

import typing as tp
from pathlib import Path
import argparse
import concurrent.futures

import torch
from torchvision.io import read_video
from torchaudio import load as read_audio
import numpy as np
import av
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description="Attach video and audio.")
    parser.add_argument(
        "--video",
        "-v",
        type=str,
        help="Path to the video dir or file.",
    )
    parser.add_argument(
        "--audio",
        "-a",
        type=str,
        help="Path to the audio dir or file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=False,
        help="Directory where to save the attached video and audio. Defaults to audio dir.",
    )
    parser.add_argument(
        "--start_pts",
        "-s",
        type=float,
        default=0,
        help="Start time in seconds.",
    )
    parser.add_argument(
        "--verbose",
        "-V",
        action="store_true",
        help="Print verbose output.",
    )
    return parser.parse_args()


def get_videos_dict(videos: tp.List[Path]) -> tp.Dict[str, Path]:
    videos_dict = {}
    for video in videos:
        video_name = video.stem
        if video_name in videos_dict:
            raise ValueError(f"Duplicate video name {video_name}.")
        videos_dict[video_name] = video
    return videos_dict


def write_video(
    filename: str,
    video_array: torch.Tensor,
    fps: float,
    video_codec: str = "libx264",
    options: tp.Optional[tp.Dict[str, tp.Any]] = None,
    audio_array: tp.Optional[torch.Tensor] = None,
    audio_fps: tp.Optional[float] = None,
    audio_codec: tp.Optional[str] = None,
    audio_options: tp.Optional[tp.Dict[str, tp.Any]] = None,
) -> None:
    """
    Writes a 4d tensor in [T, H, W, C] format in a video file

    Args:
        filename (str): path where the video will be saved
        video_array (Tensor[T, H, W, C]): tensor containing the individual frames,
            as a uint8 tensor in [T, H, W, C] format
        fps (Number): video frames per second
        video_codec (str): the name of the video codec, i.e. "libx264", "h264", etc.
        options (tp.Dict): dictionary containing options to be passed into the PyAV video stream
        audio_array (Tensor[C, N]): tensor containing the audio, where C is the number of channels
            and N is the number of samples
        audio_fps (Number): audio sample rate, typically 44100 or 48000
        audio_codec (str): the name of the audio codec, i.e. "mp3", "aac", etc.
        audio_options (tp.Dict): dictionary containing options to be passed into the PyAV audio stream
    """
    video_array = torch.as_tensor(video_array, dtype=torch.uint8).numpy()

    # PyAV does not support floating point numbers with decimal point
    # and will throw OverflowException in case this is not the case
    if isinstance(fps, float):
        fps = np.round(fps)

    with av.open(filename, mode="w") as container:
        stream = container.add_stream(video_codec, rate=fps)
        stream.width = video_array.shape[2]
        stream.height = video_array.shape[1]
        stream.pix_fmt = "yuv420p" if video_codec != "libx264rgb" else "rgb24"
        stream.options = options or {}

        if audio_array is not None:
            audio_format_dtypes = {
                "dbl": "<f8",
                "dblp": "<f8",
                "flt": "<f4",
                "fltp": "<f4",
                "s16": "<i2",
                "s16p": "<i2",
                "s32": "<i4",
                "s32p": "<i4",
                "u8": "u1",
                "u8p": "u1",
            }
            a_stream = container.add_stream(audio_codec, rate=audio_fps)
            a_stream.options = audio_options or {}

            num_channels = audio_array.shape[0]
            audio_layout = "stereo" if num_channels > 1 else "mono"
            audio_sample_fmt = a_stream.format.name

            format_dtype = np.dtype(audio_format_dtypes[audio_sample_fmt])
            audio_array = torch.as_tensor(audio_array).numpy().astype(format_dtype)

            frame = av.AudioFrame.from_ndarray(
                audio_array, format=audio_sample_fmt, layout=audio_layout
            )

            frame.sample_rate = audio_fps

            for packet in a_stream.encode(frame):
                container.mux(packet)

            for packet in a_stream.encode():
                container.mux(packet)

        for img in video_array:
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pict_type = "NONE"
            for packet in stream.encode(frame):
                container.mux(packet)

        # Flush stream
        for packet in stream.encode():
            container.mux(packet)


def main():
    args = get_args()
    if args.verbose:
        print(args)

    video_path = Path(args.video).resolve()
    audio_path = Path(args.audio).resolve()
    output_path = Path(args.output).resolve() if args.output else audio_path
    if args.verbose:
        print(f"Video path: {video_path}")
        print(f"Audio path: {audio_path}")
        print(f"Output path: {output_path}")
    assert video_path.exists(), f"Video path {video_path} does not exist."
    assert audio_path.exists(), f"Audio path {audio_path} does not exist."
    assert output_path.exists(), f"Output path {output_path} does not exist."

    videos = list(video_path.glob("*.mp4")) if video_path.is_dir() else [video_path]
    videos_dict = get_videos_dict(videos)
    if args.verbose:
        print(f"Videos: {videos_dict}")
    audios = list(audio_path.glob("*.wav")) if audio_path.is_dir() else [audio_path]
    if args.verbose:
        print(f"Generating {len(audios)} videos.")

    def process_audio(audio):
        video = videos_dict.get(audio.stem, None)
        if video is None:
            print(f"Video for {audio} not found.")
            return

        audio_frames, audio_fps = read_audio(audio)
        duration = audio_frames.shape[1] / audio_fps
        video_frames, _, video_meta = read_video(
            str(video.resolve().as_posix()),
            start_pts=args.start_pts,
            end_pts=duration,
            pts_unit="sec",
        )
        if video_frames.shape[0] == 0:
            print(f"Empty video {video}.")
            return

        write_video(
            str(output_path / f"{audio.stem}.mp4"),
            video_frames,
            video_meta["video_fps"],
            video_codec="h264",
            options={"crf": "10", "pix_fmt": "yuv420p"},
            audio_array=audio_frames,
            audio_fps=audio_fps,
            audio_codec="aac",
        )

    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(process_audio, audios), total=len(audios)))


if __name__ == "__main__":
    main()
