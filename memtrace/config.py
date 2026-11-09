"""YAML config with defaults, deep-merge and ``a.b.c=value`` overrides."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "seed": 1337,
    "model": {"base": "gpt2", "device": "auto", "max_length": 128,
              # only used when base == "fixture:tiny"
              "fixture": {"n_layer": 2, "n_embd": 128, "n_head": 4, "pretrain_epochs": 4,
                          "pretrain_lr": 3e-3}},
    "data": {"canary_kinds": ["api_key", "patient_id", "password", "token"],
             "canaries_per_rep": 8, "repetitions": [1, 2, 4, 8], "holdout_canaries": None,
             "member_filler": 96, "holdout_filler": 96, "reference_filler": 96,
             "pretrain_filler": 0},
    "train": {"method": "lora", "epochs": 10, "lr": 1e-3, "batch_size": 8, "warmup_steps": 10,
              "weight_decay": 0.0, "checkpoint_epochs": [1, 2, 4, 6, 8, 10], "reference": True,
              "lora": {"r": 16, "alpha": 32, "dropout": 0.0, "target_modules": "auto"}},
    "audit": {"batch_size": 16, "min_k": 0.2,
              "extraction": {"max_new_tokens": 24, "partial_threshold": 0.5},
              "discoverable": {"prefix_tokens": 12, "suffix_tokens": 12},
              "exposure": {"candidates": 63}, "curve": True},
}


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def apply_overrides(cfg: dict, overrides: list[str] | None) -> dict:
    cfg = copy.deepcopy(cfg)
    for ov in overrides or []:
        key, _, raw = ov.partition("=")
        if not _:
            raise ValueError(f"override must look like a.b=value, got {ov!r}")
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    user = yaml.safe_load(Path(path).read_text()) if path else {}
    return apply_overrides(deep_merge(DEFAULTS, user or {}), overrides)
