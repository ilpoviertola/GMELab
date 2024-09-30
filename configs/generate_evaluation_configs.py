from pathlib import Path
import argparse
import typing as tp
import yaml


DEFAULT_PIPELINE: tp.Dict[str, tp.Dict] = {
    "fad": {},
    "imagebind_score": {},
    "sync": {},
    "kld": {},
}


def get_args():
    parser = argparse.ArgumentParser(description="Generate evaluation configs")
    parser.add_argument(
        "--experiment",
        "-e",
        type=str,
        nargs="+",
        help="Path(s) to experiment configuration YAML.",
    )
    parser.add_argument(
        "--save_path",
        "-sp",
        type=str,
        help="Directory where to save the evaluation configurations.",
    )
    parser.add_argument(
        "--base_identifier",
        "-bid",
        type=str,
        help="Base identifier for the evaluation configurations.",
    )
    parser.add_argument(
        "--id_from_exp_cfg",
        "-cfgid",
        type=str,
        nargs="+",
        help="Key(s) to use as identifier from the experiment configuration.",
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


def get_experiment_cfgs(experiment_path: Path) -> tp.List[dict]:
    """Get path to experiment configuration file(s) and return the configurations.

    Args:
        experiment_path (Path): Can be:
            - Path to a parent dir containing all the experiments
            - Path to a single experiment folder
            - Path to a single experiment config file

    Returns:
        tp.List[dict]: list of experiment configurations.
    """
    assert experiment_path.exists(), f"Experiment path not found: {experiment_path}"
    if experiment_path.is_dir():
        if (experiment_path / "config.yaml").exists():
            cfg = yaml.load(
                (experiment_path / "config.yaml").read_text(),
                Loader=yaml.FullLoader,
            )
            cfg["sample_directory"] = experiment_path.resolve().as_posix()
            return [cfg]
        else:
            cfgs = []
            for dir in experiment_path.glob("generated_samples*"):
                if (dir / "config.yaml").exists():
                    cfg = yaml.load(
                        (dir / "config.yaml").read_text(), Loader=yaml.FullLoader
                    )
                    cfg["sample_directory"] = dir.resolve().as_posix()
                    cfgs.append(cfg)
            return cfgs
    else:
        assert experiment_path.suffix == ".yaml", "Invalid experiment config file"
        cfg = yaml.load(experiment_path.read_text(), Loader=yaml.FullLoader)
        cfg["sample_directory"] = experiment_path.parent.resolve().as_posix()
        return [cfg]


def generate_evaluation_cfg(
    experiment_cfg: dict,
    base_identifier: str,
    gt_dir: Path,
    cfg_id_keys: tp.List[str],
    metafile: tp.Optional[Path] = None,
) -> dict:
    """Generate evaluation configurations from the experiment configuration.

    Args:
        experiment_cfg (dict): Experiment configuration
        base_identifier (str): Identifier for the evaluation configurations.
        gt_dir (Path): Directory where the ground truth files are stored.
        metafile (Path): Path to the metafile.

    Returns:
        tp.List[dict]: List of evaluation configurations.
    """

    def build_id(base: str, cfg: dict, keys: list) -> str:
        eval_id = base
        for key in keys:
            eval_id += f"_{key}_{cfg[key]}"
        return eval_id

    eval_cfg = {
        "id": build_id(base_identifier, experiment_cfg, cfg_id_keys),
        "sample_directory": experiment_cfg["sample_directory"],
        "sample_afps": 44100,
        "gt_afps": 44100,
        "gt_directory": gt_dir.resolve().as_posix(),
        "metadata": metafile.resolve().as_posix() if metafile else None,
        "verbose": False,
        "pipeline": DEFAULT_PIPELINE,
    }
    return eval_cfg


def main():
    args = get_args()
    assert args.experiment, "No experiments defiend"
    assert args.save_path, "No save path defined"

    experiment_cfgs = []
    for exp in args.experiment:
        experiment_cfgs.extend(get_experiment_cfgs(Path(exp)))

    save_dir = Path(args.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    for exp_cfg in experiment_cfgs:
        eval_cfg = generate_evaluation_cfg(
            exp_cfg,
            args.base_identifier,
            Path(args.gt_dir),
            args.id_from_exp_cfg,
            Path(args.metafile) if args.metafile else None,
        )

        save_path = save_dir / f"{eval_cfg['id']}.yaml"
        if save_path.exists():
            # add running number to the end of the file name
            i = 1
            while (save_path.parent / f"{eval_cfg['id']}-{i}.yaml").exists():
                i += 1
            save_path = save_path.parent / f"{eval_cfg['id']}-{i}.yaml"
        save_path.write_text(yaml.dump(eval_cfg))


if __name__ == "__main__":
    main()
