import typing as tp
import argparse
from pathlib import Path
import csv
import shutil
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))  # noqa: E402

from tqdm import tqdm

from eval_utils.file_utils import cut_video, save_audio_from_video


def get_args():
    parser = argparse.ArgumentParser(description="Cut evaluation videos")
    parser.add_argument(
        "--input_dirs",
        "-i",
        type=str,
        required=True,
        nargs="+",
        help="Path(s) to the directory containing the evaluation videos",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        required=True,
        help="Path to the directory where the cut videos will be saved",
    )
    parser.add_argument(
        "--metadata",
        "-m",
        type=str,
        help="Path to the metadata file",
    )
    parser.add_argument(
        "--clip_length",
        "-c",
        type=float,
        default=2.56,
    )
    return parser.parse_args()


def resolve_input_dir(input_dir: str) -> tp.List[Path]:
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Directory {input_path} does not exist")

    # check if input_dir has mp4 files
    video_files = list(input_path.glob("*.mp4"))
    if not video_files:
        # list subdirectories
        subdirs = [x for x in input_path.iterdir() if x.is_dir()]
        # go thorugh subdirs and return those with mp4 files
        video_subdirs = [x for x in subdirs if list(x.glob("*.mp4"))]
        return video_subdirs
    else:
        return [input_path]


def process_video_dir(
    input_dir: Path,
    output_dir: Path,
    metadata: tp.Dict[str, float],
    clip_length: float = 2.56,
):
    import concurrent.futures

    videos = list(input_dir.glob("*.mp4"))
    # videos = metadata.keys()

    def process_video(video):
        #  video = input_dir / f"{video}.mp4"
        video = Path(video)
        if not video.exists():
            print(f"Video {video} does not exist. Skipping")
            return
        shutil.copy(
            video.resolve(),
            (output_dir / f"{video.stem}.tmp{video.suffix}").resolve(),
        )
        start_time = metadata.get(video.stem, 0)
        cut_video(
            (output_dir / f"{video.stem}.tmp{video.suffix}").as_posix(),
            start_time,
            clip_length,
            (output_dir / video.name).as_posix(),
        )
        Path(output_dir / f"{video.stem}.tmp{video.suffix}").unlink()
        save_audio_from_video(output_dir / video.name, 22050)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(process_video, videos), total=len(videos)))


def main():
    args = get_args()
    input_dirs = []
    for input_dir in args.input_dirs:
        input_dirs.extend(resolve_input_dir(input_dir))
    print(f"Found {len(input_dirs)} directories with videos")
    print(input_dirs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_length = args.clip_length

    if args.metadata is not None:
        assert Path(
            args.metadata
        ).exists(), f"Metadata file {args.metadata} does not exist"
        with open(args.metadata, "r") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            metadata = {row[0]: float(row[1]) for row in reader}
    else:
        metadata = {}

    for input_dir in input_dirs:
        output_subdir = output_dir / input_dir.name
        if output_subdir.exists():
            print(f"Directory {output_subdir} already exists. Skipping")
            continue

        output_subdir.mkdir(parents=False, exist_ok=False)
        process_video_dir(input_dir, output_subdir, metadata, clip_length)


if __name__ == "__main__":
    main()
