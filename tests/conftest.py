"""Shared fixtures. The overfit fixture trains once per session (~1-2 min on CPU)."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def fixture_cfg():
    from memtrace.config import load_config

    return load_config(ROOT / "config" / "fixture.yaml", ["model.device=cpu"])


@pytest.fixture(scope="session")
def overfit_run(tmp_path_factory, fixture_cfg):
    """Tiny GPT-2 + byte tokenizer, LoRA-fine-tuned to memorize 20 canaries."""
    from memtrace.pipeline import run_audit, run_train

    run = tmp_path_factory.mktemp("overfit_run")
    quiet = lambda *_: None  # noqa: E731
    run_train(fixture_cfg, run, log=quiet)
    audit = run_audit(run, fixture_cfg, log=quiet)
    return run, audit


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory):
    from memtrace.models import build_tiny_fixture_base, load_model

    d = build_tiny_fixture_base(tmp_path_factory.mktemp("tiny_base"), n_layer=1, n_embd=32, n_head=2)
    return load_model(str(d), device="cpu")
