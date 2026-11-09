"""Success criteria from BRIEF.md, measured on the tiny overfit fixture (CPU only)."""

from __future__ import annotations

import json


def test_mia_auc_at_least_0_8(overfit_run):
    _, audit = overfit_run
    aucs = {k: v["auc"] for k, v in audit["mia"]["all"].items()}
    assert max(aucs.values()) >= 0.8, aucs
    # the plain LOSS attack alone should clear the bar on an overfit model
    assert aucs["loss"] >= 0.8, aucs


def test_all_attacks_beat_chance(overfit_run):
    _, audit = overfit_run
    for name, m in audit["mia"]["all"].items():
        assert m["auc"] > 0.6, (name, m["auc"])


def test_member_canaries_extracted_holdout_not(overfit_run):
    _, audit = overfit_run
    ex = audit["extraction"]
    assert ex["member"]["n"] == 20 and ex["holdout"]["n"] == 20
    assert ex["member"]["exact_rate"] > 0.0
    assert ex["holdout"]["exact_rate"] == 0.0
    assert ex["member"]["leak_rate"] > ex["holdout"]["leak_rate"]


def test_more_repetitions_leak_more(overfit_run):
    _, audit = overfit_run
    by = audit["extraction"]["by_repetition"]
    assert by["8"]["leak_rate"] >= by["1"]["leak_rate"]
    exp = audit["exposure"]
    assert exp["member_mean"] > exp["holdout_mean"] + 1.0


def test_base_model_does_not_know_canaries(overfit_run):
    _, audit = overfit_run
    assert audit["extraction"]["base_model_member"]["exact_rate"] == 0.0


def test_discoverable_member_ge_holdout(overfit_run):
    _, audit = overfit_run
    d = audit["discoverable"]
    assert d["member"]["mean_token_acc"] >= d["holdout"]["mean_token_acc"]


def test_audit_artifacts_written(overfit_run):
    run, _ = overfit_run
    assert (run / "audit" / "audit.json").exists()
    assert (run / "audit" / "scores.csv").read_text().count("\n") == 121  # header + 120 records
    rows = (run / "audit" / "extraction.csv").read_text().splitlines()
    assert len(rows) == 41
    json.loads((run / "audit" / "audit.json").read_text())


def test_report_writes_roc_and_extraction_table(overfit_run, tmp_path):
    from memtrace.report import write_report

    run, _ = overfit_run
    md = write_report(run / "audit" / "audit.json", tmp_path)
    text = md.read_text()
    assert (tmp_path / "roc.png").stat().st_size > 10_000
    assert (tmp_path / "extraction_by_repetition.png").exists()
    assert (tmp_path / "privacy_utility.png").exists()
    assert "| Canary group |" in (tmp_path / "extraction_table.md").read_text()
    assert "Membership inference" in text and "Canary extraction" in text


def test_privacy_utility_curve_monotone_ish(overfit_run):
    _, audit = overfit_run
    c = audit["curve"]
    assert c[0]["epoch"] == 0 and c[-1]["epoch"] == 60
    assert c[-1]["extract_leak"] >= c[0]["extract_leak"]
    assert c[-1]["auc_base_ratio"] > 0.5
