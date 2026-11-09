"""LOSS-style membership inference.

* ``loss_score``      - Yeom et al. (2018): members have lower loss. Score = -mean NLL.
* ``base_ratio_score``- calibrate by the *base* model (Carlini et al. 2021,
                        "reference = original model"): score = NLL_base - NLL_ft.
                        Removes "this text is just easy" from the signal.
* ``lira_lite_score`` - offline LiRA (Carlini et al. 2022) with K reference
                        models trained on data that *excludes* the targets.
                        Fit a Gaussian to the reference losses per example and
                        report how far below it the target loss falls:
                        score = (mu_ref - NLL_ft) / sigma_ref. With K = 1 the
                        per-example sigma is replaced by the global std.

All inputs are per-example mean NLL arrays (see :func:`memtrace.scoring.mean_nll`).
"""

from __future__ import annotations

import numpy as np


def loss_score(nll_target: np.ndarray) -> np.ndarray:
    return -np.asarray(nll_target, dtype=float)


def base_ratio_score(nll_target: np.ndarray, nll_base: np.ndarray) -> np.ndarray:
    return np.asarray(nll_base, dtype=float) - np.asarray(nll_target, dtype=float)


def lira_lite_score(nll_target: np.ndarray, nll_refs: list[np.ndarray] | np.ndarray,
                    eps: float = 1e-6) -> np.ndarray:
    refs = np.atleast_2d(np.asarray(nll_refs, dtype=float))  # (K, N)
    t = np.asarray(nll_target, dtype=float)
    mu = refs.mean(0)
    if refs.shape[0] > 1:
        sigma = refs.std(0, ddof=1)
    else:
        sigma = np.full_like(mu, (mu - t).std() if len(t) > 1 else 1.0)
    return (mu - t) / (sigma + eps)
