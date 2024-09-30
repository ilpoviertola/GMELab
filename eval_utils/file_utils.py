from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import shutil
import subprocess
import os

import numpy as np
import julius
import torch
from tqdm import tqdm
from audiotools import Meter
from torchaudio import load, save
from torchaudio.transforms import Resample
import torchvision
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_audio
import av


VCODEC = "h264"
CRF = 10
PIX_FMT = "yuv420p"
ACODEC = "aac"


class AudioLoudnessNormalize(torch.nn.Module):
    GAIN_FACTOR = np.log(10) / 20

    def __init__(self, target_loudness: float = 0.0, sr: int = 24000) -> None:
        super().__init__()
        self.target_loudness = target_loudness
        self.meter = Meter(sr)

    def forward(self, wav: torch.Tensor):
        loudness = self.meter.integrated_loudness(wav.permute(0, 2, 1))
        gain = self.target_loudness - loudness
        gain = torch.exp(gain * self.GAIN_FACTOR)
        return wav * gain[:, None, None]


def check_is_file(file: Path, suffix: str = ".mp4"):
    if file.is_file() and file.suffix == suffix:
        return True
    else:
        return False


def which_ffmpeg() -> str:
    """Determines the path to ffmpeg library
    Returns:
        str -- path to the library
    """
    result = subprocess.run(
        ["which", "ffmpeg"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    ffmpeg_path = result.stdout.decode("utf-8").replace("\n", "")
    return ffmpeg_path


def get_new_path(
    path, vcodec, acodec, v_fps, min_side, a_fps, orig_path, prefix="video"
) -> Path:
    new_folder_name = f"{vcodec}_{prefix}_{v_fps}fps_{min_side}side_{a_fps}hz_{acodec}"
    if "vggsound" in str(orig_path):
        new_folder_path = orig_path.parent / new_folder_name
    elif "mjpeg" in str(orig_path) or "lrs3" in str(orig_path):
        new_folder_path = Path(
            str(path.parent).replace(orig_path.name, f"/{new_folder_name}/")
        )
    elif "greatesthit" in str(orig_path):
        new_folder_path = orig_path.parents[0] / new_folder_name
    else:
        raise NotImplementedError
    os.makedirs(new_folder_path, exist_ok=True)
    new_path = new_folder_path / path.name
    return new_path


def reencode_video(
    path,
    vfps,
    afps,
    min_side,
    new_path,
    acodec=ACODEC,
    vcodec=VCODEC,
    pix_fmt=PIX_FMT,
    crf=CRF,
):
    # reencode the original mp4: rescale, resample video and resample audio
    cmd = f"{which_ffmpeg()}"
    assert cmd != "", "activate an env with ffmpeg/ffprobe"
    # no info/error printing
    cmd += " -hide_banner -loglevel panic"
    cmd += f" -i {path}"
    # 1) change fps, 2) resize: min(H,W)=MIN_SIDE (vertical vids are supported), 3) change audio framerate
    cmd += f" -vf fps={vfps},scale=iw*{min_side}/'min(iw,ih)':ih*{min_side}/'min(iw,ih)',crop='trunc(iw/2)'*2:'trunc(ih/2)'*2"
    cmd += f" -vcodec {vcodec} -pix_fmt {pix_fmt} -crf {crf}"
    cmd += f" -acodec {acodec} -ar {afps} -ac 1"
    cmd += f" {new_path}"
    if not Path(new_path).exists():
        subprocess.call(cmd.split())


def write_video(
    filename: str,
    video_array: torch.Tensor,
    fps: float,
    video_codec: str = "libx264",
    options: Optional[Dict[str, Any]] = None,
    audio_array: Optional[torch.Tensor] = None,
    audio_fps: Optional[float] = None,
    audio_codec: Optional[str] = None,
    audio_options: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Writes a 4d tensor in [T, H, W, C] format in a video file

    Args:
        filename (str): path where the video will be saved
        video_array (Tensor[T, H, W, C]): tensor containing the individual frames,
            as a uint8 tensor in [T, H, W, C] format
        fps (Number): video frames per second
        video_codec (str): the name of the video codec, i.e. "libx264", "h264", etc.
        options (Dict): dictionary containing options to be passed into the PyAV video stream
        audio_array (Tensor[C, N]): tensor containing the audio, where C is the number of channels
            and N is the number of samples
        audio_fps (Number): audio sample rate, typically 44100 or 48000
        audio_codec (str): the name of the audio codec, i.e. "mp3", "aac", etc.
        audio_options (Dict): dictionary containing options to be passed into the PyAV audio stream
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


def cut_video(input_file: str, start_time: float, duration: float, output_file: str):
    """
    Cut a video to a certain length using moviepy.

    Args:
        input_file (str): Path to the input video file.
        start_time (int): Start time of the segment to cut in seconds.
        end_time (int): End time of the segment to cut in seconds.
        output_file (str): Path to the output video file.
    """
    cmd = f"{which_ffmpeg()} -hide_banner -loglevel panic -y -i {input_file} -ss {start_time} -t {duration} {output_file}"
    subprocess.call(cmd.split())


def convert_audio_channels(wav: torch.Tensor, channels: int = 2) -> torch.Tensor:
    """Convert audio to the given number of channels.

    Args:
        wav (torch.Tensor): Audio wave of shape [B, C, T].
        channels (int): Expected number of channels as output.
    Returns:
        torch.Tensor: Downmixed or unchanged audio wave [B, C, T].
    """
    *shape, src_channels, length = wav.shape
    if src_channels == channels:
        pass
    elif channels == 1:
        # Case 1:
        # The caller asked 1-channel audio, and the stream has multiple
        # channels, downmix all channels.
        wav = wav.mean(dim=-2, keepdim=True)
    elif src_channels == 1:
        # Case 2:
        # The caller asked for multiple channels, but the input file has
        # a single channel, replicate the audio over all channels.
        wav = wav.expand(*shape, channels, length)
    elif src_channels >= channels:
        # Case 3:
        # The caller asked for multiple channels, and the input file has
        # more channels than requested. In that case return the first channels.
        wav = wav[..., :channels, :]
    else:
        # Case 4: What is a reasonable choice here?
        raise ValueError(
            "The audio file has less channels than requested but is not mono."
        )
    return wav


def convert_audio(
    wav: torch.Tensor, from_rate: float, to_rate: float, to_channels: int
) -> torch.Tensor:
    """Convert audio to new sample rate and number of audio channels."""
    wav = julius.resample_frac(wav, int(from_rate), int(to_rate))
    wav = convert_audio_channels(wav, to_channels)
    return wav


def resample_file_if_needed(
    file_path: Path, new_sample_rate: int, output_path: Optional[Path] = None
) -> Tuple[Path, bool]:
    """Resample an audio file if the sample rate is not the same as the desired sample rate.

    Args:
        file_path (Path): Path to audio file.
        new_sample_rate (int): Desired sample rate.
        output_path (Optional[Path]): Path to directory containing audio files.

    Returns:
        Tuple[Path, bool]: Path to (resampled) audio file and whether the file was resampled.
    """
    output_path = (
        output_path or file_path.parent / f"{file_path.stem}_{new_sample_rate}.wav"
    )
    assert output_path.suffix == ".wav", "Output file must be a .wav file."
    output_path.parent.mkdir(exist_ok=True, parents=True)

    waveform, sample_rate = load(file_path)
    if sample_rate == new_sample_rate:
        return file_path, False
    resample = Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
    resampled_waveform = resample(waveform)
    save(output_path, resampled_waveform, new_sample_rate)
    return output_path, True


def resample_dir_if_needed(
    dir_path: Path, new_sample_rate: int, output_path: Optional[Path] = None
) -> Tuple[Path, bool]:
    """Resample a directory of audio files if the sample rate is not the same as the desired sample rate.

    Args:
        dir_path (Path): Path to directory containing audio files.
        new_sample_rate (int): Desired sample rate.
        output_path (Optional[Path]): Path to directory containing audio files.

    Returns:
        Tuple[Path, bool]: Path to (resampled) dir and whether the files were resampled.
    """
    resampled_dir = False
    output_path = output_path or dir_path.parent / f"{dir_path.name}_{new_sample_rate}"
    output_path.mkdir(exist_ok=True, parents=True)

    for file_path in tqdm(
        dir_path.glob("*.wav"),
        desc=f"Resampling directory {dir_path.name} to {new_sample_rate} Hz",
        total=len(list(dir_path.glob("*.wav"))),
    ):
        _, resampled = resample_file_if_needed(
            file_path, new_sample_rate, output_path / file_path.name
        )
        if resampled:
            resampled_dir = True

    if resampled_dir:
        return output_path, True

    rmdir_and_contents(output_path)  # remove empty directory
    return dir_path, False


def reencode_video_if_needed(
    file_path: Path,
    vfps: int,
    afps: int,
    input_size: int,
    output_path: Optional[Path] = None,
) -> Tuple[Path, bool]:
    """Reencode a video file if the frame rates or input size are not the same as the desired frame rates or input size.

    Args:
        file_path (Path): Path to video file.
        vfps (int): Desired video frame rate.
        afps (int): Desired audio frame rate.
        input_size (int): Desired input size.
        output_path (Optional[Path]): Path to directory containing video files.

    Returns:
        Tuple[Path, bool]: Path to (reencoded) video file and whether the file was reencoded.
    """
    output_path = (
        output_path
        or file_path.parent / f"{file_path.stem}_{vfps}_{afps}_{input_size}.mp4"
    )
    if output_path.exists():
        return output_path, False
    assert output_path.suffix == ".mp4", "Output file must be a .mp4 file."
    output_path.parent.mkdir(exist_ok=True, parents=True)

    v, a, vid_meta = torchvision.io.read_video(file_path.as_posix(), pts_unit="sec")
    _, H, W, _ = v.shape
    if (
        vid_meta["video_fps"] != vfps
        or vid_meta["audio_fps"] != afps
        or min(H, W) != input_size
    ):
        reencode_video(file_path, vfps, afps, input_size, output_path)
        return output_path, True
    return file_path, False


# TODO: implement multiprocessing
def reencode_dir_if_needed(
    dir_path: Path,
    vfps: int,
    afps: int,
    input_size: int,
    output_path: Optional[Path] = None,
) -> Tuple[Path, bool]:
    """Reencode a directory of video files if the frame rates or input size are not the same as the desired frame rates or input size.

    Args:
        dir_path (Path): Path to directory containing video files.
        vfps (int): Desired video frame rate.
        afps (int): Desired audio frame rate.
        input_size (int): Desired input size.
        output_path (Optional[Path]): Path to directory containing video files.

    Returns:
        Tuple[Path, bool]: Path to (reencoded) dir and whether the files were reencoded.
    """
    reencoded_dir = False
    output_path = output_path or dir_path.parent / f"{dir_path.name}_{vfps}_{afps}"
    output_path.mkdir(exist_ok=True, parents=True)

    for file_path in tqdm(
        dir_path.glob("*.mp4"),
        desc=f"Reencoding directory {dir_path.name} to {vfps} fps, {afps} afps, {input_size} input size",
        total=len(list(dir_path.glob("*.mp4"))),
    ):
        _, reencoded = reencode_video_if_needed(
            file_path, vfps, afps, input_size, output_path / file_path.name
        )
        if reencoded:
            reencoded_dir = True

    if reencoded_dir:
        return output_path, True

    rmdir_and_contents(output_path)  # remove empty directory
    return dir_path, False


# TODO: implement multiprocessing
def save_audio_from_video(
    file_path: Path, samplerate: int, output_path: Optional[Path] = None
) -> Path:
    """Save audio from video file.

    Args:
        file_path (Path): Path to video file.
        output_path (Optional[Path]): Path to directory containing audio files.

    Returns:
        Path: Path to audio file.
    """
    output_path = output_path or file_path.parent / f"{file_path.stem}.wav"
    assert output_path.suffix == ".wav", "Output file must be a .wav file."
    if not output_path.exists():
        output_path.parent.mkdir(exist_ok=True, parents=True)
        ffmpeg_extract_audio(file_path, output_path, bitrate=3000, fps=samplerate)
    return output_path


def extract_audios_from_video_dir_if_needed(
    video_dir_path: Path,
    samplerate: int = 24000,
    force_extract: bool = False,
    output_path: Optional[Path] = None,
) -> Tuple[Path, bool]:
    """Extract audio from video files in a directory.

    Args:
        video_dir_path (Path): Path to video directory.
        afps (int): Desired audio frame rate.
        output_path (Optional[Path], optional): Outputpath for audios. Defaults to None.

    Returns:
        Tuple[Path, bool]: Audio save path and whether the audio was extracted.
    """
    video_file_amnt = len(list(video_dir_path.glob("*.mp4")))
    audio_file_amnt = len(list(video_dir_path.glob("*.wav")))

    if not force_extract:
        if video_file_amnt == 0 and audio_file_amnt == 0:
            raise Exception(
                f"No video or audio files found in {video_dir_path.as_posix()}. Nothing to extract."
            )
        elif video_file_amnt == audio_file_amnt:
            print(
                f"Amount of video and audio files match in {video_dir_path.as_posix()}. No need to extract audio."
            )
            return video_dir_path, False
        elif video_file_amnt == 0:
            print(
                f"No video files found in {video_dir_path.as_posix()}. Nothing to extract."
            )
            return video_dir_path, False

    output_path = output_path or video_dir_path
    output_path.mkdir(exist_ok=True, parents=True)

    for file_path in tqdm(
        video_dir_path.glob("*.mp4"),
        desc=f"Extracting audio from {video_dir_path.name}",
        total=video_file_amnt,
    ):
        if force_extract or not (output_path / f"{file_path.stem}.wav").exists():
            save_audio_from_video(
                file_path, samplerate, output_path / f"{file_path.stem}.wav"
            )

    return output_path, True


def to_reencode(dir_path: Path) -> bool:
    """
    This is used in situations where only the audio is needed (WAV).
    This function checks if the directory already contains audio files
    extracted from the videos. If not, it returns True, and then reencoding function
    needs to be called. If the directory already contains audio files, it returns False,
    and then resampling function needs to be called.

    Args:
        dir_path (Path): Path to directory containing files.

    Returns:
        bool: Whether the directory needs to be reencoded.
    """
    video_file_amnt = len(list(dir_path.glob("*.mp4")))
    audio_file_amnt = len(list(dir_path.glob("*.wav")))
    return video_file_amnt != audio_file_amnt


def rmdir_and_contents(dir_path: Path, verbose: bool = False):
    """Remove a directory and its contents."""
    if verbose:
        print(f"Removing directory {dir_path} and its contents.")
    for file in dir_path.glob("*"):
        file.unlink()
    dir_path.rmdir()


def copy_files(source_dir: Path, destination_dir: Path, file_mask: str = "*.wav"):
    """Copy files from source directory to destination directory."""
    destination_dir.mkdir(exist_ok=True, parents=True)
    for file in source_dir.glob(file_mask):
        shutil.copy(file, destination_dir / file.name)
    return destination_dir


def copy_file(source: str, destination: str):
    source_path = Path(source)
    destination_path = Path(destination)
    shutil.copy(source_path, destination_path)


def reencode_videos_in_parallel(
    videos, output_dir, fps=21.5, audio_sample_rate=22050, side=256
):
    from concurrent.futures import ProcessPoolExecutor
    import concurrent

    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                reencode_video,
                video.as_posix(),
                fps,
                audio_sample_rate,
                side,
                output_dir / video.name,
            ): video
            for video in videos
        }
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Reencoding videos",
        ):
            try:
                future.result()
            except Exception as exc:
                print(f"Generated an exception: {exc}")


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    from tqdm import tqdm

    path = Path("")

    def process_video(video):
        if video.suffix == ".mp4":
            save_audio_from_video(video, 44100)

    with ProcessPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(process_video, path.iterdir()),
                total=len(list(path.iterdir())),
                desc="Processing videos",
            )
        )
