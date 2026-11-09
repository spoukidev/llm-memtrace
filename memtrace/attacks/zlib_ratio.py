"""zlib-entropy calibration (Carlini et al., 2021, "Extracting Training Data").

Highly repetitive text has low perplexity *and* compresses well; unique
memorized text has low perplexity but compresses poorly. Dividing the model's
log-perplexity by the zlib entropy of the text removes the "trivially
predictable" confounder.

score = - (sum NLL in nats) / (zlib-compressed length in bits)   (higher = member)
"""

from __future__ import annotations

import zlib

import numpy as np


def zlib_entropy_bits(text: str) -> int:
    return 8 * len(zlib.compress(text.encode("utf-8"), 9))


def zlib_score(token_lps: list[np.ndarray], texts: list[str]) -> np.ndarray:
    return np.array([lp.sum() / zlib_entropy_bits(t) for lp, t in zip(token_lps, texts)])
