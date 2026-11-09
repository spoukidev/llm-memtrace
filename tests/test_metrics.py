from __future__ import annotations

import numpy as np

from memtrace.metrics import (
    auc,
    common_prefix_len,
    edit_similarity,
    extraction_summary,
    levenshtein,
    mia_summary,
    rouge_l,
    tpr_at_fpr,
)


def test_auc_and_tpr():
    y = np.array([1] * 50 + [0] * 50)
    perfect = np.r_[np.ones(50), np.zeros(50)]
    assert auc(y, perfect) == 1.0
    assert tpr_at_fpr(y, perfect, 0.01) == 1.0
    rng = np.random.default_rng(0)
    rnd = rng.random(100)
    assert 0.3 < auc(y, rnd) < 0.7
    s = mia_summary(y, perfect)
    assert s["auc"] == 1.0 and s["n_members"] == 50 and len(s["auc_ci95"]) == 2


def test_string_metrics():
    assert levenshtein("kitten", "sitting") == 3
    assert edit_similarity("abcd", "abcd") == 1.0
    assert edit_similarity("", "abcd") == 0.0
    assert common_prefix_len("abcx", "abcd") == 3
    assert rouge_l("abc", "abc") == 1.0
    assert rouge_l("xyz", "abc") == 0.0


def test_extraction_summary():
    rows = [{"exact": True, "partial": False, "edit_sim": 1.0, "lcp": 8},
            {"exact": False, "partial": True, "edit_sim": 0.6, "lcp": 2},
            {"exact": False, "partial": False, "edit_sim": 0.1, "lcp": 0}]
    s = extraction_summary(rows)
    assert s["exact_rate"] == 1 / 3 and s["leak_rate"] == 2 / 3
