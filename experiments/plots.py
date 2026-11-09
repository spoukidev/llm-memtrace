"""Plots for the canary study (and a shortcut to re-render a report).

    python experiments/plots.py --study results/study/study.csv
    python experiments/plots.py --audit runs/demo/audit/audit.json --out results/demo
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def plot_study(csv_path: str | Path, out_dir: str | Path) -> None:
    rows = list(csv.DictReader(open(csv_path)))
    out = Path(out_dir)
    by_rank: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_rank[int(r["rank"])].append(r)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for rank, rs in sorted(by_rank.items()):
        rs = sorted(rs, key=lambda x: int(x["epochs"]))
        ep = [int(x["epochs"]) for x in rs]
        axes[0].plot(ep, [float(x["leak_member"]) for x in rs], "o-", label=f"r={rank}")
        axes[1].plot(ep, [float(x["auc_loss"]) for x in rs], "o-", label=f"r={rank}")
        axes[2].plot([float(x["ppl_holdout"]) for x in rs], [float(x["leak_member"]) for x in rs],
                     "o-", label=f"r={rank}")
    axes[0].set(xlabel="epochs", ylabel="member canary leak rate (exact+partial)",
                title="Extraction vs epochs", ylim=(0, 1.05))
    axes[1].set(xlabel="epochs", ylabel="LOSS-attack AUC", title="Membership inference vs epochs",
                ylim=(0.4, 1.02))
    axes[2].set(xlabel="held-out perplexity", ylabel="member canary leak rate",
                title="Privacy-utility across the grid")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "study_epochs_rank.png", dpi=130)
    plt.close(fig)

    reps = sorted({int(k[6:]) for k in rows[0] if k.startswith("leak_x")})
    if reps:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for r in rows:
            ax.plot(reps, [float(r[f"leak_x{k}"]) for k in reps], "o-", alpha=0.8,
                    label=f"ep={r['epochs']}, r={r['rank']}")
        ax.set(xscale="log", xticks=reps, xticklabels=[str(k) for k in reps],
               xlabel="canary repetitions in training set", ylabel="leak rate (exact+partial)",
               title="Repetition drives memorization", ylim=(0, 1.05))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(out / "study_repetitions.png", dpi=130)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study")
    ap.add_argument("--audit")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.study:
        plot_study(a.study, a.out or Path(a.study).parent)
    if a.audit:
        from memtrace.report import write_report

        print(write_report(a.audit, a.out or "results/latest"))


if __name__ == "__main__":
    main()
