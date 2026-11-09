"""Metrics: AUC, TPR at low FPR, bootstrap CIs, and string-match measures.

Following Carlini et al. (2022, "MIA from First Principles"), we report TPR
at *low* FPR alongside AUC: an attack that confidently identifies a few
members is a real privacy problem even when average-case AUC looks modest.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def auc(labels, scores) -> float:
    labels = np.asarray(labels)
    if len(set(labels.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def tpr_at_fpr(labels, scores, target_fpr: float) -> float:
    """Highest TPR achievable with FPR <= target_fpr."""
    fpr, tpr, _ = roc_curve(labels, scores)
    ok = fpr <= target_fpr + 1e-12
    return float(tpr[ok].max()) if ok.any() else 0.0


def best_balanced_accuracy(labels, scores) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(((tpr + (1 - fpr)) / 2).max())


def bootstrap_auc_ci(labels, scores, n: int = 500, alpha: float = 0.05,
                     seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    labels, scores = np.asarray(labels), np.asarray(scores)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(labels), len(labels))
        if len(set(labels[idx].tolist())) < 2:
            continue
        vals.append(roc_auc_score(labels[idx], scores[idx]))
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))


def mia_summary(labels, scores, low_fprs=(0.01, 0.05, 0.1), ci: bool = True) -> dict:
    d = {"auc": auc(labels, scores), "balanced_acc": best_balanced_accuracy(labels, scores)}
    for f in low_fprs:
        d[f"tpr@{f:g}fpr"] = tpr_at_fpr(labels, scores, f)
    if ci:
        lo, hi = bootstrap_auc_ci(labels, scores)
        d["auc_ci95"] = [lo, hi]
    d["n_members"] = int(np.sum(np.asarray(labels) == 1))
    d["n_nonmembers"] = int(np.sum(np.asarray(labels) == 0))
    return d


def roc_points(labels, scores) -> dict:
    fpr, tpr, _ = roc_curve(labels, scores)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist()}


# --- string metrics --------------------------------------------------------

def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def edit_similarity(guess: str, secret: str) -> float:
    if not secret:
        return 1.0
    return max(0.0, 1.0 - levenshtein(guess, secret) / max(len(secret), len(guess)))


def common_prefix_len(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def rouge_l(guess: str, ref: str) -> float:
    """Character-level ROUGE-L F1 (LCS based)."""
    if not guess or not ref:
        return 0.0
    m, n = len(guess), len(ref)
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            tmp = dp[j]
            dp[j] = prev + 1 if guess[i - 1] == ref[j - 1] else max(dp[j], dp[j - 1])
            prev = tmp
    lcs = dp[n]
    p, r = lcs / m, lcs / n
    return 0.0 if lcs == 0 else 2 * p * r / (p + r)


def extraction_summary(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0, "exact_rate": 0.0, "partial_rate": 0.0, "mean_edit_sim": 0.0, "mean_lcp": 0.0}
    n = len(rows)
    return {
        "n": n,
        "exact_rate": sum(r["exact"] for r in rows) / n,
        "partial_rate": sum(r["partial"] for r in rows) / n,
        "leak_rate": sum(r["exact"] or r["partial"] for r in rows) / n,
        "mean_edit_sim": float(np.mean([r["edit_sim"] for r in rows])),
        "mean_lcp": float(np.mean([r["lcp"] for r in rows])),
    }
