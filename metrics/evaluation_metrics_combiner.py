""""Combine multiple evaluation metrics results."""

import typing as tp
from pathlib import Path

import numpy as np

from metrics.evaluation_metrics import (
    EvaluationMetrics,
    load_evaluation_metrics_from_file,
)


class EvaluationMetricsCombiner:
    def __init__(
        self,
        metrics: tp.Optional[tp.List[EvaluationMetrics]] = None,
        result_file_paths: tp.Optional[tp.List[Path]] = None,
    ):
        self.metrics = metrics if metrics is not None else []
        if result_file_paths is not None:
            self.load_metrics(result_file_paths)

        self.all_results: tp.Dict[str, tp.Tuple[list, list]] = {}

    def load_metrics(self, result_file_paths: tp.List[Path]):
        for result_file_path in result_file_paths:
            assert result_file_path.exists(), f"File {result_file_path} does not exist."
            metric = load_evaluation_metrics_from_file(result_file_path)
            self.metrics.append(metric)

    def combine(self):
        for metric in self.metrics:
            assert isinstance(metric.results, dict)
            assert metric.results is not None, "Results are empty"
            for metric_type in metric.results:
                if metric_type not in self.all_results:
                    self.all_results[metric_type] = ([], [])

                # special metrics
                if metric_type == "insync_per_video":
                    for video in metric.results[metric_type]:
                        self.all_results[metric_type][0].append(
                            f"{metric.cfg.id}:{video}"
                        )
                        self.all_results[metric_type][1].append(
                            int(metric.results[metric_type][video]["insync"])
                        )
                    continue

                # general metrics
                self.all_results[metric_type][0].append(
                    f"{metric.cfg.id}:{Path(metric.cfg.sample_directory).name}"
                )
                self.all_results[metric_type][1].append(metric.results[metric_type])
        return self.all_results

    def plot(
        self, plot_dir: tp.Union[str, Path, None]
    ) -> tp.Union[Path, tp.Dict[str, np.ndarray]]:
        """Plot metrics. Return path to the plot or the plot itself.

        Args:
            plot_dir (tp.Union[str, Path, None]): Path to plot directory. If None, return the plot itself.

        Returns:
            tp.Union[Path, tp.Dict[str, np.ndarray]]: Path to the plot or the plot itself.
        """
        assert (
            self.all_results is not None
        ), "All results are empty. Did you run combine()?"
        if isinstance(plot_dir, str):
            plot_dir = Path(plot_dir)
        save_plots_to_png = plot_dir is not None

        if save_plots_to_png:
            assert isinstance(plot_dir, Path), "Plot directory is not provided."
            plot_dir.mkdir(exist_ok=True, parents=True)
            ids = "_".join(sorted([metric.cfg.id for metric in self.metrics]))

        plots = {}
        for metric_type in self.all_results:
            if metric_type == "insync_per_video":
                from pandas import DataFrame

                # read the results into a DataFrame
                data = {
                    "video": self.all_results[metric_type][0],
                    "is_insync": self.all_results[metric_type][1],
                }
                df = DataFrame(data)
                if save_plots_to_png:
                    df.to_csv(plot_dir / f"{ids}-insync_per_video.csv", index=False)
                else:
                    print(df)
                continue

            if save_plots_to_png:
                plot_path = plot_dir / f"{ids}-{metric_type}.png"
            else:
                plot_path = None
            plots[metric_type] = self.plot_metric(metric_type, plot_path=plot_path)
        return plots

    def plot_metric(
        self, metric_type: str, plot_path: tp.Union[Path, None] = None
    ) -> tp.Union[np.ndarray, Path]:
        """Plot a metric. Return the plot.

        Args:
            metric_type (str): Metric type.
            plot_path (tp.Union[str, Path, None], optional): Path to plot. If None, return the plot itself.

        Returns:
            tp.Union[np.ndarray, Path]: Plot or path to it.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        if len(self.all_results[metric_type][0]) == 1:
            ax.scatter(
                self.all_results[metric_type][0], self.all_results[metric_type][1]
            )
        else:
            ax.plot(self.all_results[metric_type][0], self.all_results[metric_type][1])
        ax.set_xlabel("Sample directory")
        ax.set_ylabel(metric_type)
        plt.xticks(rotation=90)
        if plot_path is not None:
            plt.savefig(plot_path, bbox_inches="tight")
            plt.close()
            return plot_path
        else:
            return fig

    def export_to_table(self, table_path: tp.Union[str, Path]) -> tp.Optional[Path]:
        """Export the combined metrics to a table.

        Args:
            table_path (tp.Union[str, Path]): Path to the table.
        """
        from pandas import DataFrame

        if table_path is not None:
            table_path = Path(table_path)
            table_path.parent.mkdir(exist_ok=True, parents=True)

        data = {}
        for metric_type in self.all_results:
            if metric_type == "insync_per_video":
                continue
            data[metric_type] = self.all_results[metric_type][1]
        df = DataFrame(data, index=self.all_results[metric_type][0])
        df.index.name = "experiment"

        if table_path:
            df.to_csv(table_path, index=True)
            return Path(table_path)
        else:
            print(df)
            return None
