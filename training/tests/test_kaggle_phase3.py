"""End-to-end smoke of the self-contained Kaggle training script (Phase 3).

Runs the notebook script locally on CPU with tiny constants and synthetic
bins — proves the script (model mirror, loop, checkpoint/resume) works
before it ever runs on a Kaggle GPU.
"""

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest


def _load_kaggle_module():
    path = Path(__file__).resolve().parents[1] / "kaggle" / "phase3_train.py"
    module_name = "phase3_train"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def kaggle_module():
    return _load_kaggle_module()


def test_kaggle_smoke_run_and_resume(kaggle_module, smoke_bins, mini_tokenizer, tmp_path):
    for name, value in {
        "TOTAL_STEPS": 8,
        "BATCH_SIZE": 2,
        "LR": 5e-3,
        "MIN_LR": 1e-4,
        "WARMUP_STEPS": 2,
        "CHECKPOINT_INTERVAL": 4,
        "VAL_INTERVAL": 4,
        "VAL_BATCHES": 1,
        "LOG_EVERY": 2,
        "MAX_NEW_TOKENS": 4,
        "SAMPLE_PROMPTS": ["Tom", "The dog"],
    }.items():
        setattr(kaggle_module, name, value)
    kaggle_module.CONTEXT_LENGTH = 32

    ckpts = tmp_path / "ckpts"
    mini_tokenizer.save(str(smoke_bins / "tokenizer.json"))
    kaggle_module.main(data_dir=str(smoke_bins), ckpt_dir=str(ckpts))
    assert (ckpts / "ckpt-0000008.pt").exists()
    metrics = (ckpts / "metrics.jsonl").read_text().strip().splitlines()
    assert any('"val_loss"' in line for line in metrics)
    assert any('"samples"' in line for line in metrics)

    # resume: total 12 steps, must continue past 8
    kaggle_module.TOTAL_STEPS = 12
    kaggle_module.main(data_dir=str(smoke_bins), ckpt_dir=str(ckpts))
    assert (ckpts / "ckpt-0000012.pt").exists()
