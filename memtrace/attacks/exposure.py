"""Canary *exposure* (Carlini et al., 2019, "The Secret Sharer").

Rank the true secret among R random secrets of the same format by model
likelihood of the full canary sentence. exposure = log2(R + 1) - log2(rank).
Exposure ~ log2(R+1) means the true secret is the single most likely
candidate; ~1 means it is indistinguishable from chance.

This is a *likelihood* measure: it can detect memorization well before
greedy decoding reproduces the secret verbatim.
"""

from __future__ import annotations

import math

import numpy as np

from memtrace.data.canaries import Canary, alternative_secrets
from memtrace.scoring import token_logprobs


def canary_exposure(model, tok, canaries: list[Canary], n_candidates: int = 63, seed: int = 0,
                    batch_size: int = 32, max_length: int = 256) -> list[dict]:
    rows = []
    for c in canaries:
        alts = alternative_secrets(c, n_candidates, seed)
        texts = [c.text] + [f"{c.prefix}{a}{c.suffix}" for a in alts]
        lps = token_logprobs(model, tok, texts, batch_size, max_length)
        ll = np.array([lp.sum() for lp in lps])
        rank = 1 + int((ll[1:] > ll[0]).sum())
        rows.append({"id": c.id, "rank": rank,
                     "exposure": math.log2(n_candidates + 1) - math.log2(rank),
                     "max_exposure": math.log2(n_candidates + 1)})
    return rows
