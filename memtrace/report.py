"""Render an ``audit.json`` into a Markdown report with figures.

Outputs (in ``out_dir``):
    report.md, roc.png, extraction_by_repetition.png, privacy_utility.png,
    score_hist.png, extraction_table.md, extraction.csv (copied), summary.json
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {"loss": "#1f77b4", "base_ratio": "#ff7f0e", "zlib": "#2ca02c", "min_k": "#d62728",
          "lira_lite": "#9467bd"}


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _f(x: float | None, nd: int = 3) -> str:
    return "n/a" if x is None or x != x else f"{x:.{nd}f}"


def plot_roc(audit: dict, path: Path) -> None:
    labels = audit.get("attack_labels", {})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for name, pts in audit["roc"].items():
        auc = audit["mia"]["all"][name]["auc"]
        lab = f"{labels.get(name, name)} (AUC {auc:.3f})"
        for ax in axes:
            ax.plot(pts["fpr"], pts["tpr"], label=lab, color=COLORS.get(name), lw=1.8)
    for ax in axes:
        ax.plot([1e-4, 1], [1e-4, 1], "k--", lw=0.8, label="chance")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.grid(alpha=0.3)
    axes[0].set_title("ROC (linear)")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlim(1e-2, 1)
    axes[1].set_ylim(1e-2, 1.02)
    axes[1].set_title("ROC (log-log: the low-FPR regime)")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Membership inference: members vs holdout")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_extraction(audit: dict, path: Path) -> None:
    ex = audit["extraction"]
    reps = sorted(ex["by_repetition"], key=int)
    exact = [ex["by_repetition"][r]["exact_rate"] for r in reps]
    leak = [ex["by_repetition"][r]["leak_rate"] for r in reps]
    xs = list(range(len(reps) + 1))
    names = [f"x{r}" for r in reps] + ["holdout"]
    exact.append(ex["holdout"]["exact_rate"])
    leak.append(ex["holdout"]["leak_rate"])
    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    w = 0.38
    ax1.bar([x - w / 2 for x in xs], exact, w, label="exact match", color="#d62728")
    ax1.bar([x + w / 2 for x in xs], leak, w, label="exact or partial (>=50%)", color="#ff9896")
    ax1.set_xticks(xs, names)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("canary extraction rate")
    ax1.set_xlabel("times the canary appeared in the fine-tune set")
    if audit.get("exposure"):
        e = audit["exposure"]
        ax2 = ax1.twinx()
        ev = [e["by_repetition"][r] for r in reps] + [e["holdout_mean"]]
        ax2.plot(xs, ev, "ko-", label="mean exposure (bits)")
        ax2.set_ylim(0, e["max"] * 1.1)
        ax2.set_ylabel(f"exposure (max {e['max']:.1f} bits)")
    ax1.set_title("Canary extraction vs repetition (greedy decoding)")
    fig.legend(loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_privacy_utility(audit: dict, path: Path) -> bool:
    c = audit.get("curve") or []
    if len(c) < 2:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ep = [p["epoch"] for p in c]
    ax = axes[0]
    ax.plot(ep, [p["extract_exact"] for p in c], "o-", color="#d62728", label="canary exact extraction")
    ax.plot(ep, [p["extract_leak"] for p in c], "o--", color="#ff9896", label="canary exact+partial")
    ax.plot(ep, [p["auc_base_ratio"] for p in c], "s-", color="#ff7f0e", label="MIA AUC (LOSS vs base)")
    ax.set_xlabel("epoch")
    ax.set_ylim(0, 1.05)
    ax.set_title("Privacy leakage over training")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax = axes[1]
    sc = ax.scatter([p["ppl_holdout"] for p in c], [p["extract_leak"] for p in c], c=ep,
                    cmap="viridis", s=50, zorder=3)
    ax.plot([p["ppl_holdout"] for p in c], [p["extract_leak"] for p in c], "-", color="grey", lw=0.8)
    for p in c:
        ax.annotate(f"e{p['epoch']}", (p["ppl_holdout"], p["extract_leak"]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("held-out perplexity, log scale (utility, lower = better)")
    ax.set_ylabel("canary leak rate (privacy cost)")
    ax.set_title("Privacy-utility trade-off")
    fig.colorbar(sc, ax=ax, label="epoch")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def plot_hist(scores_csv: Path, audit: dict, path: Path) -> bool:
    if not scores_csv.exists():
        return False
    rows = list(csv.DictReader(open(scores_csv)))
    key = "base_ratio" if "base_ratio" in rows[0] else "loss"
    m = [float(r[key]) for r in rows if r["label"] == "1"]
    h = [float(r[key]) for r in rows if r["label"] == "0"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(h, bins=30, alpha=0.6, label="holdout (non-member)", color="#7f7f7f")
    ax.hist(m, bins=30, alpha=0.6, label="member", color="#1f77b4")
    ax.set_xlabel(f"{audit['attack_labels'].get(key, key)} score (higher = more member-like)")
    ax.set_ylabel("count")
    ax.legend()
    ax.set_title("Score distributions")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def extraction_table(audit: dict) -> str:
    ex = audit["extraction"]
    lines = ["| Canary group | n | Exact match | Partial (>=50%) | Exact or partial | Mean edit sim | Mean exposure (bits) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    exp = audit.get("exposure") or {}
    for r in sorted(ex["by_repetition"], key=int):
        s = ex["by_repetition"][r]
        e = exp.get("by_repetition", {}).get(r)
        lines.append(f"| member, inserted x{r} | {s['n']} | {_pct(s['exact_rate'])} | "
                     f"{_pct(s['partial_rate'])} | {_pct(s['leak_rate'])} | {_f(s['mean_edit_sim'])} | {_f(e, 2)} |")
    for key, lab, e in (("member", "**all members**", exp.get("member_mean")),
                        ("holdout", "**holdout (never trained)**", exp.get("holdout_mean")),
                        ("base_model_member", "base model on member canaries", None)):
        s = ex[key]
        lines.append(f"| {lab} | {s['n']} | {_pct(s['exact_rate'])} | {_pct(s['partial_rate'])} | "
                     f"{_pct(s['leak_rate'])} | {_f(s['mean_edit_sim'])} | {_f(e, 2)} |")
    return "\n".join(lines)


def mia_table(audit: dict, subset: str = "all") -> str:
    labels = audit.get("attack_labels", {})
    lines = ["| Attack | AUC | 95% CI | TPR @ 1% FPR | TPR @ 5% FPR | TPR @ 10% FPR | Best bal. acc |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    rows = sorted(audit["mia"][subset].items(), key=lambda kv: -kv[1]["auc"])
    for name, m in rows:
        ci = m.get("auc_ci95")
        cis = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "-"
        lines.append(f"| {labels.get(name, name)} | {m['auc']:.3f} | {cis} | {_pct(m['tpr@0.01fpr'])} | "
                     f"{_pct(m['tpr@0.05fpr'])} | {_pct(m['tpr@0.1fpr'])} | {m['balanced_acc']:.3f} |")
    return "\n".join(lines)


def render_synthetic(audit: dict, out: Path, run_dir: Path | None) -> str:
    plot_roc(audit, out / "roc.png")
    plot_extraction(audit, out / "extraction_by_repetition.png")
    has_curve = plot_privacy_utility(audit, out / "privacy_utility.png")
    has_hist = plot_hist(run_dir / "audit" / "scores.csv", audit, out / "score_hist.png") if run_dir else False
    if run_dir and (run_dir / "audit" / "extraction.csv").exists():
        shutil.copy(run_dir / "audit" / "extraction.csv", out / "extraction.csv")
    (out / "extraction_table.md").write_text(extraction_table(audit) + "\n")

    best_name, best = max(audit["mia"]["all"].items(), key=lambda kv: kv[1]["auc"])
    lab = audit["attack_labels"]
    ex, d, u = audit["extraction"], audit["data"], audit["utility"]
    md = ["# MemTrace audit report\n",
          f"**Model:** `{audit['model']['base']}` + {audit['model']['method']} "
          f"(r={audit['model']['lora_r']}, {audit['model']['epochs']} epochs, lr={audit['model']['lr']})  ",
          f"**Data:** {d['n_member']} member records ({d['member_canaries']} planted canaries, repetitions "
          f"{d['repetitions']}), {d['n_holdout']} holdout records ({d['holdout_canaries']} control canaries), "
          f"{d['train_examples']} training examples  ",
          f"**Wall time:** train {audit.get('train_wall_time_s')} s, audit {audit.get('audit_wall_time_s')} s\n",
          "## Headline\n",
          f"> This fine-tune separates members from non-members with **AUC {best['auc']:.3f}** "
          f"({lab.get(best_name, best_name)}; TPR {_pct(best['tpr@0.01fpr'])} at 1% FPR) and leaks planted "
          f"canaries verbatim at **{_pct(ex['member']['exact_rate'])}** "
          f"({_pct(ex['member']['leak_rate'])} incl. partial) vs **{_pct(ex['holdout']['exact_rate'])}** "
          f"for never-seen holdout canaries.\n",
          "## 1. Membership inference (all records)\n", mia_table(audit, "all"), "",
          "![ROC](roc.png)\n"]
    if has_hist:
        md += ["![Score histogram](score_hist.png)\n"]
    md += ["### By record type\n", "Filler records only:\n", mia_table(audit, "filler"), "",
           "Canary records only:\n", mia_table(audit, "canary"), ""]
    br = audit["mia"].get("by_repetition", {})
    if br:
        md += ["### Canary MIA AUC by repetition count\n",
               "| Repetitions | " + " | ".join(lab.get(k, k) for k in next(iter(br.values()))) + " |",
               "|---|" + "---:|" * len(next(iter(br.values())))]
        for r in sorted(br, key=int):
            md.append(f"| x{r} | " + " | ".join(f"{v['auc']:.3f}" for v in br[r].values()) + " |")
        md.append("")
    md += ["## 2. Canary extraction (prefix -> greedy decode)\n", extraction_table(audit), "",
           "![Extraction](extraction_by_repetition.png)\n"]
    bk = ex.get("by_kind", {})
    if bk:
        md += ["By canary format:\n", "| Format | Member exact | Member exact+partial | Holdout exact+partial |",
               "|---|---:|---:|---:|"]
        for k, v in bk.items():
            md.append(f"| `{k}` | {_pct(v['member']['exact_rate'])} | {_pct(v['member']['leak_rate'])} | "
                      f"{_pct(v['holdout']['leak_rate'])} |")
        md.append("")
    dm = audit["discoverable"]
    md += ["## 3. Discoverable memorization (filler records)\n",
           f"Prompt with the first {dm['prefix_tokens']} tokens, greedy-decode {dm['suffix_tokens']}, "
           "require an exact token match.\n",
           "| Split | n | Exact suffix rate | Mean token accuracy |", "|---|---:|---:|---:|",
           f"| member | {dm['member']['n']} | {_pct(dm['member']['exact_suffix_rate'])} | {_f(dm['member']['mean_token_acc'])} |",
           f"| holdout | {dm['holdout']['n']} | {_pct(dm['holdout']['exact_suffix_rate'])} | {_f(dm['holdout']['mean_token_acc'])} |",
           "", "## 4. Utility\n",
           "| Model | PPL held-out (same domain) | PPL general text | PPL members |", "|---|---:|---:|---:|",
           f"| base | {_f(u['base']['ppl_holdout'], 2)} | {_f(u['base']['ppl_general'], 2)} | - |",
           f"| fine-tuned | {_f(u['target']['ppl_holdout'], 2)} | {_f(u['target']['ppl_general'], 2)} | "
           f"{_f(u['target']['ppl_member'], 2)} |", ""]
    if has_curve:
        md += ["## 5. Privacy-utility curve\n",
               "| Epoch | MIA AUC (vs base) | Exact extraction | Exact+partial | PPL held-out | PPL general |",
               "|---:|---:|---:|---:|---:|---:|"]
        for p in audit["curve"]:
            md.append(f"| {p['epoch']} | {p['auc_base_ratio']:.3f} | {_pct(p['extract_exact'])} | "
                      f"{_pct(p['extract_leak'])} | {p['ppl_holdout']:.2f} | {p['ppl_general']:.2f} |")
        md += ["", "![Privacy-utility](privacy_utility.png)\n"]
    md += ["## How to read this\n",
           "* **AUC** is average-case; **TPR @ low FPR** is what matters for privacy (can the attacker be "
           "confidently right about *some* records?).",
           "* **Holdout canaries** are the control: same templates and formats, never trained on. Any "
           "extraction there would indicate guessing, not memorization.",
           "* **Exposure** ranks the true secret among random same-format candidates by likelihood "
           "(max = log2(candidates+1)); it detects memorization before verbatim extraction succeeds.",
           "* All secrets are synthetic and fake. Only audit models and data you have rights to test.\n"]
    return "\n".join(md)


def render_candidates(audit: dict, out: Path) -> str:
    lab = audit["attack_labels"]
    md = ["# MemTrace candidate audit\n",
          f"Base `{audit['base']}`, adapter `{audit['adapter']}`; {audit['n_candidates']} candidates, "
          f"{audit['n_nonmembers']} known non-members for calibration.\n"]
    if audit.get("mia"):
        plot_roc(audit, out / "roc.png")
        md += ["## Labelled candidates\n", mia_table(audit, "all"), "", "![ROC](roc.png)\n"]
    keys = [k for k in lab if k in audit["candidates"][0]]
    md += ["## Per-candidate scores\n",
           "| # | Text (truncated) | " + " | ".join(lab[k] for k in keys) + " |",
           "|---:|---|" + "---:|" * len(keys)]
    for i, r in enumerate(audit["candidates"][:200]):
        t = r["text"].replace("|", "\\|")[:60]
        cells = []
        for k in keys:
            c = f"{r[k]:.3f}"
            if f"{k}_pvalue" in r:
                c += f" (p={r[f'{k}_pvalue']:.2f}{', FLAG' if r[f'{k}_flag'] else ''})"
            cells.append(c)
        md.append(f"| {i} | {t} | " + " | ".join(cells) + " |")
    md += ["", "Scores: higher = more member-like. p-values are empirical vs the supplied "
           "non-members; FLAG = above the threshold calibrated to the target FPR "
           f"({next(iter(audit['calibration'].values()))['target_fpr'] if audit['calibration'] else 'n/a'}).\n"]
    return "\n".join(md)


def write_report(audit_path: str | Path, out_dir: str | Path) -> Path:
    audit_path = Path(audit_path)
    if audit_path.is_dir():
        audit_path = audit_path / "audit" / "audit.json" if (audit_path / "audit").exists() \
            else audit_path / "audit.json"
    audit = json.loads(audit_path.read_text())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if audit["mode"] == "synthetic":
        run_dir = Path(audit_path).parent.parent
        md = render_synthetic(audit, out, run_dir)
        summary = {
            "model": audit["model"],
            "mia_auc": {k: round(v["auc"], 4) for k, v in audit["mia"]["all"].items()},
            "mia_tpr_at_1pct_fpr": {k: round(v["tpr@0.01fpr"], 4) for k, v in audit["mia"]["all"].items()},
            "extraction_exact": {"member": audit["extraction"]["member"]["exact_rate"],
                                 "holdout": audit["extraction"]["holdout"]["exact_rate"]},
            "extraction_leak": {"member": audit["extraction"]["member"]["leak_rate"],
                                "holdout": audit["extraction"]["holdout"]["leak_rate"]},
            "exposure": audit.get("exposure"),
            "discoverable": {k: audit["discoverable"][k] for k in ("member", "holdout")},
            "utility": audit["utility"],
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        shutil.copy(audit_path, out / "audit.json")
    else:
        md = render_candidates(audit, out)
    (out / "report.md").write_text(md)
    return out / "report.md"
