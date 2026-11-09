"""Canary study: how do epochs and LoRA rank change memorization?

Each grid cell trains a fresh fine-tune and runs the audit (no reference
model / curve, to keep it fast). Canary repetition is varied *inside* every
run (canaries are inserted 1/2/4/8 times), so one grid gives you all three
axes: repetitions x epochs x rank.

    python experiments/run_canary_study.py                      # tiny fixture, ~10 min CPU
    python experiments/run_canary_study.py --config config/default.yaml \
        --epochs 4 8 12 --ranks 4 16 --out results/study-gpt2  # gpt2, ~1 h CPU

Writes <out>/study.csv, <out>/study.json and figures via experiments/plots.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from memtrace.config import load_config  # noqa: E402
from memtrace.data.splits import GENERAL_TEXT, build_corpus  # noqa: E402
from memtrace.models import build_tiny_fixture_base  # noqa: E402
from memtrace.pipeline import run_audit, run_train  # noqa: E402
from memtrace.train.lora_finetune import pretrain  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config" / "fixture.yaml"))
    ap.add_argument("--epochs", type=int, nargs="+", default=[15, 30, 60])
    ap.add_argument("--ranks", type=int, nargs="+", default=[4, 16, 32])
    ap.add_argument("--runs", default=str(ROOT / "runs" / "study"))
    ap.add_argument("--out", default=str(ROOT / "results" / "study"))
    ap.add_argument("--set", dest="overrides", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    runs, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Build the fixture base once and reuse it across the grid.
    if cfg["model"]["base"].startswith("fixture:"):
        fx, d = cfg["model"]["fixture"], cfg["data"]
        base_dir = runs / "_shared_base"
        if not (base_dir / "config.json").exists():
            build_tiny_fixture_base(base_dir, fx["n_layer"], fx["n_embd"], fx["n_head"], seed=cfg["seed"])
            corpus = build_corpus(seed=cfg["seed"], pretrain_filler=d["pretrain_filler"],
                                  member_filler=0, holdout_filler=0, reference_filler=0,
                                  canaries_per_rep=0, repetitions=[], holdout_canaries=0)
            pretrain(base_dir, [r.text for r in corpus.pretrain] + GENERAL_TEXT,
                     epochs=fx["pretrain_epochs"], lr=fx["pretrain_lr"],
                     max_length=cfg["model"]["max_length"], seed=cfg["seed"], device=cfg["model"]["device"])
        cfg["model"]["base"] = str(base_dir)

    rows = []
    for ep in args.epochs:
        for r in args.ranks:
            c = json.loads(json.dumps(cfg))
            c["train"].update(epochs=ep, reference=False, checkpoint_epochs=[])
            c["train"]["lora"].update(r=r, alpha=2 * r)
            c["audit"]["curve"] = False
            name = f"ep{ep}_r{r}"
            print(f"=== {name}", flush=True)
            run_train(c, runs / name, log=lambda *_: None)
            a = run_audit(runs / name, c, log=lambda *_: None)
            row = {"epochs": ep, "rank": r,
                   **{f"auc_{k}": round(v["auc"], 4) for k, v in a["mia"]["all"].items()},
                   **{f"tpr1_{k}": round(v["tpr@0.01fpr"], 4) for k, v in a["mia"]["all"].items()},
                   "exact_member": a["extraction"]["member"]["exact_rate"],
                   "leak_member": a["extraction"]["member"]["leak_rate"],
                   "exact_holdout": a["extraction"]["holdout"]["exact_rate"],
                   "leak_holdout": a["extraction"]["holdout"]["leak_rate"],
                   "exposure_member": round(a["exposure"]["member_mean"], 3) if a["exposure"] else None,
                   "exposure_holdout": round(a["exposure"]["holdout_mean"], 3) if a["exposure"] else None,
                   "ppl_holdout": round(a["utility"]["target"]["ppl_holdout"], 3),
                   "ppl_general": round(a["utility"]["target"]["ppl_general"], 3),
                   **{f"leak_x{k}": v["leak_rate"] for k, v in a["extraction"]["by_repetition"].items()},
                   **{f"exact_x{k}": v["exact_rate"] for k, v in a["extraction"]["by_repetition"].items()}}
            print(json.dumps(row), flush=True)
            rows.append(row)

    with open(out / "study.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / "study.json").write_text(json.dumps({"config": args.config, "rows": rows}, indent=2))
    from plots import plot_study

    plot_study(out / "study.csv", out)
    print(f"study -> {out}")


if __name__ == "__main__":
    main()
