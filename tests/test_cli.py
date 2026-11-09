from __future__ import annotations

import json

from typer.testing import CliRunner

from memtrace.cli import app

runner = CliRunner()


def test_help_lists_commands():
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0
    for cmd in ("train", "audit", "report"):
        assert cmd in r.output


def test_canaries_command():
    r = runner.invoke(app, ["canaries", "--n", "3"])
    assert r.exit_code == 0 and r.output.count("[") == 3


def test_report_command(overfit_run, tmp_path):
    run, _ = overfit_run
    r = runner.invoke(app, ["report", "--run", str(run), "--out", str(tmp_path / "rep")])
    assert r.exit_code == 0, r.output
    assert (tmp_path / "rep" / "report.md").exists()
    assert (tmp_path / "rep" / "roc.png").exists()


def test_audit_mode_on_user_candidates(overfit_run, tmp_path):
    """Audit mode: the owner scores texts they own against their own adapter."""
    run, _ = overfit_run
    info = json.loads((run / "run.json").read_text())
    mem = [json.loads(x) for x in (run / "data" / "member.jsonl").read_text().splitlines()]
    hold = [json.loads(x) for x in (run / "data" / "holdout.jsonl").read_text().splitlines()]
    cand = tmp_path / "cand.jsonl"
    cand.write_text("\n".join(json.dumps({"text": r["text"], "label": 1}) for r in mem[:15]) + "\n" +
                    "\n".join(json.dumps({"text": r["text"], "label": 0}) for r in hold[:15]) + "\n")
    nonm = tmp_path / "nonmembers.txt"
    nonm.write_text("\n".join(r["text"] for r in hold[15:55]) + "\n")
    out = tmp_path / "aud"
    r = runner.invoke(app, ["audit", "--base", info["base"], "--adapter", info["target"],
                            "--candidates", str(cand), "--nonmembers", str(nonm),
                            "--out", str(out), "--device", "cpu"])
    assert r.exit_code == 0, r.output
    res = json.loads((out / "audit.json").read_text())
    assert res["n_candidates"] == 30 and res["mia"]["all"]["loss"]["auc"] >= 0.8
    assert "loss_pvalue" in res["candidates"][0]
    r = runner.invoke(app, ["report", "--audit", str(out / "audit.json"), "--out", str(out / "rep")])
    assert r.exit_code == 0, r.output
    assert "Per-candidate scores" in (out / "rep" / "report.md").read_text()
