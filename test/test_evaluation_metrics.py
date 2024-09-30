import sys
import pytest
from pathlib import Path

sys.path.append(".")
from configs.evaluation_cfg import EvaluationCfg
from eval_utils.utils import dataclass_from_dict
from metrics.evaluation_metrics import EvaluationMetrics
from eval_utils.test_utils import sample_dirs, pipeline  # fixtures


@pytest.fixture
def evaluation_cfg(sample_dirs, pipeline):
    return dataclass_from_dict(
        EvaluationCfg,
        {
            "sample_directory": sample_dirs[1],
            "pipeline": pipeline,
            "verbose": True,
        },
    )


def test_evaluation_metrics(evaluation_cfg):
    metrics = EvaluationMetrics(evaluation_cfg)
    metrics.run_all()


def test_insync():
    cfg = dataclass_from_dict(
        EvaluationCfg,
        {
            "sample_directory": Path(
                "/home/hdd/ilpo/evaluation_data/synchronisonix/24-02-27T16-46-55/generated_samples_24-02-28T14-47-57_jepa"
            ),
            "pipeline": {"insync": {}},
            "verbose": True,
        },
    )
    metrics = EvaluationMetrics(cfg)
    metrics.run_insync()
