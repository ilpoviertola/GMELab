import sys
import pytest
from pathlib import Path


sys.path.append(".")
from configs.evaluation_cfg import get_evaluation_config, EvaluationCfg, PipelineCfg
from eval_utils.exceptions import ConfigurationError, ConfigurationWarning
from eval_utils.test_utils import sample_dirs, pipeline, gt_dir, cfg_file  # fixtures


def test_empty_init():
    with pytest.raises(ConfigurationError):
        get_evaluation_config({})


def test_init_with_extra_keys(sample_dirs, pipeline, gt_dir):
    cfg = {
        "id": "test_extra_keys",
        "sample_directory": sample_dirs[0],
        "gt_directory": gt_dir,
        "pipeline": pipeline,
        "extra_key": 1,
    }
    with pytest.raises(ConfigurationError):
        get_evaluation_config(cfg)


def test_init_with_empty_pipeline(sample_dirs, gt_dir):
    cfg = {
        "id": "test_empty_pipeline",
        "sample_directory": sample_dirs[0],
        "gt_directory": gt_dir,
        "pipeline": {},
    }
    with pytest.raises(ConfigurationError):
        get_evaluation_config(cfg)


def test_init_with_extra_pipeline_keys(sample_dirs, pipeline, gt_dir):
    pipeline["extra_pipeline_key"] = {}
    cfg = {
        "id": "test_extra_pipeline_keys",
        "sample_directory": sample_dirs[0],
        "gt_directory": gt_dir,
        "pipeline": pipeline,
    }
    with pytest.raises(ConfigurationError):
        get_evaluation_config(cfg)


def test_init_with_extra_metric_key(sample_dirs, pipeline, gt_dir):
    pipeline["fad"]["extra_metric_key"] = 1
    cfg = {
        "id": "test_extra_pipeline_keys",
        "sample_directory": sample_dirs[0],
        "gt_directory": gt_dir,
        "pipeline": pipeline,
    }
    with pytest.raises(ConfigurationError):
        get_evaluation_config(cfg)


def test_init_with_unexisting_sample_dir(pipeline, gt_dir):
    cfg = {
        "id": "test_unexisting_sample_dir",
        "sample_directory": Path("unexisting"),
        "gt_directory": gt_dir,
        "pipeline": pipeline,
    }
    with pytest.raises(ConfigurationError):
        get_evaluation_config(cfg)


def test_init_evaluation_cfg(sample_dirs, pipeline, gt_dir):
    cfg = {
        "id": "test_extra_keys",
        "sample_directory": sample_dirs[0],
        "gt_directory": gt_dir,
        "pipeline": pipeline,
    }
    eval_cfg = get_evaluation_config(cfg)
    assert type(eval_cfg) == EvaluationCfg, "EvaluationCfg init failed"
    assert type(eval_cfg.pipeline) == PipelineCfg, "PipelineCfg init failed"


def test_init_evaluation_cfg_from_file(cfg_file):
    eval_cfg = get_evaluation_config(cfg_file)
    assert type(eval_cfg) == EvaluationCfg, "EvaluationCfg init failed"
    assert type(eval_cfg.pipeline) == PipelineCfg, "PipelineCfg init failed"
