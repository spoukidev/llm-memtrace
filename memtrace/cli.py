"""``memtrace`` command-line interface.

    memtrace train   --config config/default.yaml --run runs/demo
    memtrace audit   --run runs/demo                       # ground-truth synthetic audit
    memtrace audit   --base gpt2 --adapter my_lora/ --candidates mine.txt \
                     --nonmembers known_unseen.txt --out audits/mine   # audit mode
    memtrace report  --run runs/demo --out results/demo
    memtrace demo    --config config/fixture.yaml --run runs/fixture --out results/fixture
    memtrace canaries --n 5
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="MemTrace: memorization & membership-inference auditor for fine-tuned LMs. "
                       "Only audit models and data you have the rights to test.")


@app.command()
def train(
    config: Path = typer.Option(Path("config/default.yaml"), help="YAML config."),
    run: Path = typer.Option(Path("runs/demo"), help="Run directory to create."),
    set_: list[str] = typer.Option([], "--set", help="Override, e.g. --set train.epochs=3"),
):
    """Build the synthetic corpus (with planted canaries) and fine-tune."""
    from memtrace.config import load_config
    from memtrace.pipeline import run_train

    info = run_train(load_config(config, set_), run)
    typer.echo(f"trained -> {info['target']} ({info['train_wall_time_s']} s)")


@app.command()
def audit(
    run: Path | None = typer.Option(None, help="Run dir from `memtrace train` (synthetic mode)."),
    base: str | None = typer.Option(None, help="Audit mode: base model id or path."),
    adapter: Path | None = typer.Option(None, help="Audit mode: your LoRA adapter / checkpoint."),
    candidates: Path | None = typer.Option(None, help="Audit mode: .txt (one per line) or .jsonl "
                                                         "with {text, label?} - texts you own."),
    nonmembers: Path | None = typer.Option(None, help="Texts known NOT to be in training (calibration)."),
    reference: list[Path] = typer.Option([], help="Optional reference adapter(s) for LiRA-lite."),
    out: Path = typer.Option(Path("audits/candidates"), help="Audit mode output dir."),
    device: str = typer.Option("auto"),
    target_fpr: float = typer.Option(0.05, help="FPR used to calibrate flags from non-members."),
    set_: list[str] = typer.Option([], "--set", help="Config override (synthetic mode)."),
):
    """Run the attack suite (MIA, extraction, exposure, discoverable memorization)."""
    if run is not None:
        import yaml

        from memtrace.config import apply_overrides
        from memtrace.pipeline import run_audit

        cfg = apply_overrides(yaml.safe_load((run / "config.yaml").read_text()), set_)
        res = run_audit(run, cfg)
        best = max(res["mia"]["all"].items(), key=lambda kv: kv[1]["auc"])
        typer.echo(f"best MIA: {best[0]} AUC={best[1]['auc']:.3f} | canary exact: member "
                   f"{res['extraction']['member']['exact_rate']:.1%} vs holdout "
                   f"{res['extraction']['holdout']['exact_rate']:.1%}")
        return
    if not (base and adapter and candidates):
        raise typer.BadParameter("give --run, or all of --base --adapter --candidates")
    from memtrace.pipeline import audit_candidates, read_texts

    nm = [r["text"] for r in read_texts(nonmembers)] if nonmembers else None
    res = audit_candidates(base, str(adapter), read_texts(candidates), out, nm,
                           [str(r) for r in reference], device, target_fpr=target_fpr)
    if "mia" in res:
        typer.echo(json.dumps({k: round(v["auc"], 3) for k, v in res["mia"]["all"].items()}))


@app.command()
def report(
    run: Path | None = typer.Option(None, help="Run dir (uses <run>/audit/audit.json)."),
    audit_json: Path | None = typer.Option(None, "--audit", help="Path to an audit.json."),
    out: Path = typer.Option(Path("results/latest"), help="Where to write report.md + figures."),
):
    """Write report.md, ROC curves, extraction table and privacy-utility plot."""
    from memtrace.report import write_report

    src = audit_json or (run / "audit" / "audit.json" if run else None)
    if src is None:
        raise typer.BadParameter("give --run or --audit")
    typer.echo(f"report -> {write_report(src, out)}")


@app.command()
def demo(
    config: Path = typer.Option(Path("config/default.yaml")),
    run: Path = typer.Option(Path("runs/demo")),
    out: Path = typer.Option(Path("results/demo")),
    set_: list[str] = typer.Option([], "--set"),
):
    """train + audit + report in one go."""
    from memtrace.config import load_config
    from memtrace.pipeline import run_audit, run_train
    from memtrace.report import write_report

    cfg = load_config(config, set_)
    run_train(cfg, run)
    run_audit(run, cfg)
    typer.echo(f"report -> {write_report(run / 'audit' / 'audit.json', out)}")


@app.command()
def canaries(n: int = typer.Option(5), kind: str | None = typer.Option(None), seed: int = 0):
    """Print example synthetic canaries (all fake)."""
    from memtrace.data.canaries import generate_canaries

    for c in generate_canaries(n, [kind] if kind else None, seed=seed):
        typer.echo(f"[{c.kind}] {c.prefix}\033[1m{c.secret}\033[0m{c.suffix}")


if __name__ == "__main__":
    app()
