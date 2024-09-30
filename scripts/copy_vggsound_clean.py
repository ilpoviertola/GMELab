"""Copy VGGSound clean dataset from full VGGSound test -set. Assumes that the
user has generated evaluation configs for both datasets.
"""

from pathlib import Path
from argparse import ArgumentParser, Namespace
import csv

from omegaconf import OmegaConf


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--vggsound_config",
        "-v",
        type=str,
        nargs="+",
        help="Path to config file(s) of VGGSound test set.",
    )
    parser.add_argument(
        "--save_dir",
        "-s",
        type=str,
        help="Path where config file(s) of VGGSound clean are saved.",
    )
    parser.add_argument(
        "--gt_dir",
        "-gt",
        type=str,
        help="Directory where the ground truth files are stored.",
    )
    parser.add_argument(
        "--metafile",
        "-m",
        type=str,
        help="Path to the metafile.",
    )
    return parser.parse_args()


def main(args: Namespace):
    cfg_files = []
    for file in args.vggsound_config:
        if Path(file).is_dir():
            cfg_files.extend(list(Path(file).rglob("*.yaml")))
        else:
            cfg_files.append(file)

    for file in cfg_files:
        cfg = OmegaConf.load(file)
        cfg.sample_directory = Path(cfg.sample_directory)
        assert cfg.sample_directory.exists(), f"{cfg.sample_directory} does not exist."
        assert (
            cfg.sample_directory.is_dir()
        ), f"{cfg.sample_directory} is not a directory."

        clean_cfg = OmegaConf.create()
        clean_cfg.merge_with(cfg)
        if "vggsound_test" in cfg.sample_directory.parent.name:
            clean_cfg.sample_directory = (
                cfg.sample_directory.parent.parent
                / "vggsound_clean"
                / cfg.sample_directory.name
            ).as_posix()
        else:
            clean_cfg.sample_directory = (
                cfg.sample_directory.parent
                / "vggsound_clean"
                / cfg.sample_directory.name
            ).as_posix()
        Path(clean_cfg.sample_directory).mkdir(parents=True, exist_ok=True)
        clean_cfg.gt_directory = Path(args.gt_dir).as_posix()
        clean_cfg.metadata = Path(args.metafile).as_posix()

        save_path = Path(args.save_dir) / Path(file).name
        OmegaConf.save(clean_cfg, save_path)

        # copy files from test to clean
        with open(args.metafile, "r") as f:
            csv_reader = csv.reader(f)
            header = next(csv_reader)
            for row in csv_reader:
                file = row[0] + ".mp4"
                src = cfg.sample_directory / file
                dst = Path(clean_cfg.sample_directory) / file
                if not src.exists():
                    print(f"{src} does not exist.")
                    continue
                dst.symlink_to(src)


if __name__ == "__main__":
    args = get_args()
    main(args)
