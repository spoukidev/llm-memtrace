from __future__ import annotations

import numpy as np
import pytest

from memtrace.attacks.loss_mia import base_ratio_score, lira_lite_score, loss_score
from memtrace.attacks.min_k import min_k_score
from memtrace.attacks.zlib_ratio import zlib_entropy_bits, zlib_score


def test_loss_and_ratio_direction():
    nll_t = np.array([0.5, 3.0])
    nll_b = np.array([3.0, 3.0])
    assert loss_score(nll_t)[0] > loss_score(nll_t)[1]
    assert base_ratio_score(nll_t, nll_b)[0] > base_ratio_score(nll_t, nll_b)[1]


def test_lira_lite_multi_and_single_reference():
    t = np.array([1.0, 3.0, 3.0])
    refs = np.array([[3.0, 3.1, 2.9], [3.2, 2.9, 3.0], [2.8, 3.0, 3.1]])
    s = lira_lite_score(t, refs)
    assert s[0] > s[1] and s[0] > s[2]
    s1 = lira_lite_score(t, refs[:1])
    assert np.argmax(s1) == 0


def test_min_k_focuses_on_outliers():
    member = [np.array([-0.5] * 10)]
    nonmember = [np.array([-0.3] * 9 + [-9.0])]  # same mean-ish, one surprising token
    assert min_k_score(member, 0.1)[0] > min_k_score(nonmember, 0.1)[0]
    with pytest.raises(ValueError):
        min_k_score(member, 0)


def test_zlib_normalisation():
    assert zlib_entropy_bits("aaaaaaaaaaaaaaaaaaaa") < zlib_entropy_bits("q8Zr!x0Lp2Vm#4Tn6Kc&")
    lps = [np.array([-1.0] * 10), np.array([-1.0] * 10)]
    s = zlib_score(lps, ["a" * 40, "q8Zr!x0Lp2Vm#4Tn6Kc&q8Zr!x0Lp2Vm#4Tn6Kc&"])
    # same likelihood, but the incompressible text is more surprising to memorize
    assert s[1] > s[0]
