import argparse
import typing as tp
from pathlib import Path
from datetime import datetime

import torch
from omegaconf import OmegaConf

from configs.evaluation_cfg import get_evaluation_config, EvaluationCfg
from metrics.evaluation_metrics import EvaluationMetrics
from metrics.evaluation_metrics_combiner import EvaluationMetricsCombiner


torch.set_float32_matmul_precision("medium")


def get_args():
    parser = argparse.ArgumentParser(description="Run evaluations on audio samples.")
    parser.add_argument(
        "--pipeline_cfg",
        "-p",
        type=str,
        nargs="+",
        help="Path(s) to pipeline configuration YAML.",
    )
    parser.add_argument(
        "--plot_dir",
        "-pd",
        type=str,
        help="Directory where to save the evaluation plots.",
    )
    parser.add_argument(
        "--table_dir",
        "-td",
        type=str,
        help="Directory where to save the evaluation result tables.",
    )
    parser.add_argument(
        "--combine_only",
        "-c",
        action="store_true",
        help="Only combine the evaluation results to table or plots.",
    )
    return parser.parse_args()


def print_pipeline_cfg(pipeline_cfg_file: tp.List[str]):
    for file in pipeline_cfg_file:
        pipeline_cfg = OmegaConf.load(file)
        print(OmegaConf.to_yaml(pipeline_cfg, resolve=True))


def get_calculated_evaluation_metrics(
    evaluation_cfg: EvaluationCfg, force_recalculate: bool = False
) -> Path:
    print(
        f"Evaluating ({evaluation_cfg.id}):", evaluation_cfg.sample_directory.as_posix()
    )
    evaluation_metrics = EvaluationMetrics(evaluation_cfg)
    assert type(evaluation_metrics) == EvaluationMetrics
    evaluation_metrics.run_all(force_recalculate)
    print("Evaluation done\n")
    return evaluation_metrics.export_results()


def get_latest_result_file(result_dir: Path) -> Path:
    result_files = list(result_dir.glob("*.yaml"))
    if not result_files:
        raise FileNotFoundError(f"No result files found in {result_dir}")
    return sorted(result_files)[-1]


def main():
    args = get_args()
    if not args.table_dir and not args.plot_dir:
        print(
            "WARNING: No table or plot directory specified. Results will be exported to YAML format only."
        )
    pipeline_cfg_file = []
    for file in args.pipeline_cfg:
        if Path(file).is_dir():
            pipeline_cfg_file.extend(list(Path(file).glob("*.yaml")))
        else:
            pipeline_cfg_file.append(file)
    print(f"Running evaluations with pipeline configuration(s): {pipeline_cfg_file}")
    print_pipeline_cfg(pipeline_cfg_file)

    if args.combine_only:
        all_evaluation_metrics = []
        for file in pipeline_cfg_file:
            eval_cfg = get_evaluation_config(file)
            all_evaluation_metrics.append(
                get_latest_result_file(eval_cfg.result_directory)
            )
    else:
        all_evaluation_cfgs = []
        all_evaluation_metrics = []
        for i, file in enumerate(pipeline_cfg_file):
            eval_cfg = get_evaluation_config(file)
            all_evaluation_cfgs.append(eval_cfg)
            metrics = get_calculated_evaluation_metrics(eval_cfg)
            all_evaluation_metrics.append(metrics)
            print("Finished evaluation", i + 1, "of", len(pipeline_cfg_file))
            print()

    evaluation_metrics_combiner = EvaluationMetricsCombiner(
        result_file_paths=all_evaluation_metrics
    )
    evaluation_metrics_combiner.combine()
    if args.table_dir:
        evaluation_metrics_combiner.export_to_table(
            Path(args.table_dir) / f"{datetime.now().strftime('%y-%m-%dT%H-%M-%S')}.csv"
        )
    if args.plot_dir:
        evaluation_metrics_combiner.plot(args.plot_dir)


if __name__ == "__main__":
    main()
