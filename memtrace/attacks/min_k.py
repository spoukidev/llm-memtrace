"""Min-K% Prob (Shi et al., 2024, "Detecting Pretraining Data from LLMs").

Non-member texts tend to contain a few *outlier* tokens with very low
probability; members do not. Score = mean log-prob of the k% least likely
tokens (higher = member). Reference-free: needs only the target model.
"""

from __future__ import annotations

import math

import numpy as np


def min_k_score(token_lps: list[np.ndarray], k: float = 0.2) -> np.ndarray:
    if not 0 < k <= 1:
        raise ValueError("k must be in (0, 1]")
    out = []
    for lp in token_lps:
        if len(lp) == 0:
            out.append(0.0)
            continue
        n = max(1, math.ceil(k * len(lp)))
        out.append(float(np.sort(lp)[:n].mean()))
    return np.array(out)
