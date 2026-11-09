"""End-to-end orchestration: build data -> fine-tune -> audit.

Run directory layout::

    runs/<name>/
      config.yaml          resolved config
      run.json             base model path, timings
      data/*.jsonl         member / holdout / reference (/ pretrain) records
      base/                only for the tiny fixture (pretrained in place)
      target/              the fine-tuned adapter (+ checkpoints/epoch-N)
      reference/           optional reference adapter (LiRA-lite)
      audit/audit.json     every metric, ROC points, per-example scores
      audit/scores.csv     per-record MIA scores
      audit/extraction.csv per-canary extraction results
"""

from __future__ import annotations

import csv
import gc
import json
import re
import shutil
import time
from pathlib import Path

import numpy as np
import yaml

from memtrace.attacks.exposure import canary_exposure
from memtrace.attacks.loss_mia import base_ratio_score, lira_lite_score, loss_score
from memtrace.attacks.min_k import min_k_score
from memtrace.attacks.prefix_extract import discoverable_memorization, extract_canaries
from memtrace.attacks.zlib_ratio import zlib_score
from memtrace.data.splits import GENERAL_TEXT, Corpus, build_corpus
from memtrace.metrics import extraction_summary, mia_summary, roc_points
from memtrace.models import build_tiny_fixture_base, load_model
from memtrace.scoring import mean_nll, perplexity, token_logprobs
from memtrace.train.lora_finetune import finetune, pretrain

ATTACK_LABELS = {
    "loss": "LOSS (-NLL)",
    "base_ratio": "LOSS vs base",
    "zlib": "zlib ratio",
    "min_k": "Min-K% Prob",
    "lira_lite": "LiRA-lite (reference)",
}


def _log(msg: str) -> None:
    print(msg, flush=True)


def _free(*_objs) -> None:
    """Encourage the previous model to be released before loading the next one."""
    gc.collect()


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def run_train(cfg: dict, run_dir: str | Path, log=_log) -> dict:
    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    for stale in ("target", "reference", "base", "audit"):  # never mix artefacts of two runs
        if (run / stale).exists():
            shutil.rmtree(run / stale)
    (run / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    t0 = time.time()
    d, m, tr = cfg["data"], cfg["model"], cfg["train"]

    corpus = build_corpus(
        seed=cfg["seed"], canary_kinds=d["canary_kinds"], canaries_per_rep=d["canaries_per_rep"],
        repetitions=d["repetitions"], holdout_canaries=d.get("holdout_canaries"),
        member_filler=d["member_filler"], holdout_filler=d["holdout_filler"],
        reference_filler=d["reference_filler"], pretrain_filler=d.get("pretrain_filler", 0),
    )
    corpus.save(run / "data")
    log(f"[data] member={len(corpus.member)} holdout={len(corpus.holdout)} "
        f"reference={len(corpus.reference)} train_examples={len(corpus.training_texts())}")

    base = m["base"]
    if base.startswith("fixture:"):
        fx = m["fixture"]
        base_dir = build_tiny_fixture_base(run / "base", fx["n_layer"], fx["n_embd"], fx["n_head"],
                                           seed=cfg["seed"])
        pre = [r.text for r in corpus.pretrain] + GENERAL_TEXT
        pretrain(base_dir, pre, epochs=fx["pretrain_epochs"], lr=fx["pretrain_lr"],
                 max_length=m["max_length"], seed=cfg["seed"], device=m["device"], log=log)
        base = str(base_dir)

    common = dict(
        method=tr["method"], epochs=tr["epochs"], lr=float(tr["lr"]), batch_size=tr["batch_size"],
        max_length=m["max_length"], lora_r=tr["lora"]["r"], lora_alpha=tr["lora"]["alpha"],
        lora_dropout=tr["lora"]["dropout"], target_modules=tr["lora"]["target_modules"],
        weight_decay=tr["weight_decay"], warmup_steps=tr["warmup_steps"], device=m["device"],
        log=log,
    )
    info = {"base": base, "target": str(run / "target")}
    info["target_train"] = finetune(base, corpus.training_texts("member"), run / "target",
                                    checkpoint_epochs=tr.get("checkpoint_epochs"),
                                    seed=cfg["seed"], **common)
    if tr.get("reference"):
        log("[train] reference model (disjoint data, same recipe) for LiRA-lite")
        info["reference_train"] = finetune(base, corpus.training_texts("reference"),
                                           run / "reference", seed=cfg["seed"] + 1, **common)
        info["references"] = [str(run / "reference")]
    for k in ("target_train", "reference_train"):
        if k in info:
            info[k] = {kk: vv for kk, vv in info[k].items() if kk != "history"} | {
                "final_loss": info[k]["history"][-1]["loss"]}
    info["train_wall_time_s"] = round(time.time() - t0, 1)
    (run / "run.json").write_text(json.dumps(info, indent=2))
    return info


# ---------------------------------------------------------------------------
# scoring helpers
# ---------------------------------------------------------------------------

def mia_scores(lp_t: list[np.ndarray], texts: list[str], lp_base: list[np.ndarray] | None,
               lp_refs: list[list[np.ndarray]] | None, k: float) -> dict[str, np.ndarray]:
    nll_t = mean_nll(lp_t)
    s = {"loss": loss_score(nll_t), "zlib": zlib_score(lp_t, texts), "min_k": min_k_score(lp_t, k)}
    if lp_base is not None:
        s["base_ratio"] = base_ratio_score(nll_t, mean_nll(lp_base))
    if lp_refs:
        s["lira_lite"] = lira_lite_score(nll_t, [mean_nll(r) for r in lp_refs])
    return s


def _checkpoints(target: Path) -> list[tuple[int, Path]]:
    out = []
    ck = target / "checkpoints"
    if ck.exists():
        for p in ck.iterdir():
            mm = re.fullmatch(r"epoch-(\d+)(\.final)?", p.name)
            if mm:
                out.append((int(mm.group(1)), target if mm.group(2) else p))
    return sorted(out)


# ---------------------------------------------------------------------------
# audit (ground-truth synthetic run)
# ---------------------------------------------------------------------------

def run_audit(run_dir: str | Path, cfg: dict | None = None, log=_log) -> dict:
    run = Path(run_dir)
    cfg = cfg or yaml.safe_load((run / "config.yaml").read_text())
    info = json.loads((run / "run.json").read_text())
    a, dev, maxlen = cfg["audit"], cfg["model"]["device"], cfg["model"]["max_length"]
    bs = a["batch_size"]
    corpus = Corpus.load(run / "data")
    t0 = time.time()

    eval_recs = corpus.member + corpus.holdout
    texts = [r.text for r in eval_recs]
    labels = np.array([1 if r.split == "member" else 0 for r in eval_recs])
    kinds = np.array([r.kind for r in eval_recs])
    reps = np.array([r.repetitions if r.split == "member" else 0 for r in eval_recs])
    hold_filler = [r.text for r in corpus.holdout if r.kind == "filler"]
    mem_filler = [r.text for r in corpus.member if r.kind == "filler"]
    mem_can, hold_can = corpus.canaries("member"), corpus.canaries("holdout")
    ex = a["extraction"]

    # --- base model -------------------------------------------------------
    log("[audit] scoring with base model")
    model, tok = load_model(info["base"], device=dev)
    lp_base = token_logprobs(model, tok, texts, bs, maxlen)
    utility = {"base": {"ppl_holdout": perplexity(model, tok, hold_filler, bs, maxlen),
                        "ppl_general": perplexity(model, tok, GENERAL_TEXT, bs, maxlen)}}
    base_extr = extract_canaries(model, tok, mem_can + hold_can, ex["max_new_tokens"],
                                 ex["partial_threshold"], bs)
    _free(model)

    # --- reference models -------------------------------------------------
    lp_refs = []
    for rp in info.get("references", []):
        log(f"[audit] scoring with reference model {rp}")
        model, tok = load_model(info["base"], rp, device=dev)
        lp_refs.append(token_logprobs(model, tok, texts, bs, maxlen))
        _free(model)

    # --- target model -----------------------------------------------------
    log("[audit] scoring with fine-tuned target")
    model, tok = load_model(info["base"], info["target"], device=dev)
    lp_t = token_logprobs(model, tok, texts, bs, maxlen)
    scores = mia_scores(lp_t, texts, lp_base, lp_refs, a["min_k"])

    mia: dict = {"all": {}, "filler": {}, "canary": {}, "by_repetition": {}}
    roc: dict = {}
    for name, sc in scores.items():
        mia["all"][name] = mia_summary(labels, sc)
        roc[name] = roc_points(labels, sc)
        for sub in ("filler", "canary"):
            msk = kinds == sub
            mia[sub][name] = mia_summary(labels[msk], sc[msk], ci=False)
        for r in sorted(set(reps[labels == 1].tolist())):
            msk = (kinds == "canary") & ((reps == r) | (labels == 0))
            mia["by_repetition"].setdefault(str(r), {})[name] = mia_summary(
                labels[msk], sc[msk], ci=False)

    log("[audit] canary extraction")
    extr = extract_canaries(model, tok, mem_can + hold_can, ex["max_new_tokens"],
                            ex["partial_threshold"], bs)
    mem_rows = [r for r in extr if r["split"] == "member"]
    hold_rows = [r for r in extr if r["split"] == "holdout"]
    extraction = {
        "member": extraction_summary(mem_rows),
        "holdout": extraction_summary(hold_rows),
        "base_model_member": extraction_summary([r for r in base_extr if r["split"] == "member"]),
        "by_repetition": {str(r): extraction_summary([x for x in mem_rows if x["repetitions"] == r])
                          for r in sorted({x["repetitions"] for x in mem_rows})},
        "by_kind": {k: {"member": extraction_summary([x for x in mem_rows if x["kind"] == k]),
                        "holdout": extraction_summary([x for x in hold_rows if x["kind"] == k])}
                    for k in sorted({x["kind"] for x in extr})},
    }

    exposure = None
    if a["exposure"]["candidates"]:
        log(f"[audit] exposure ({a['exposure']['candidates']} candidates per canary)")
        rows = canary_exposure(model, tok, mem_can + hold_can, a["exposure"]["candidates"],
                               cfg["seed"], bs * 2, maxlen)
        by_id = {r["id"]: r for r in rows}
        for r in extr:
            r["exposure"] = round(by_id[r["id"]]["exposure"], 3)
            r["rank"] = by_id[r["id"]]["rank"]
        exposure = {
            "max": rows[0]["max_exposure"],
            "member_mean": float(np.mean([by_id[c.id]["exposure"] for c in mem_can])),
            "holdout_mean": float(np.mean([by_id[c.id]["exposure"] for c in hold_can])),
            "by_repetition": {str(r): float(np.mean([by_id[c.id]["exposure"] for c in mem_can
                                                     if c.repetitions == r]))
                              for r in sorted({c.repetitions for c in mem_can})},
        }

    log("[audit] discoverable memorization")
    dm = a["discoverable"]
    dmem = discoverable_memorization(model, tok, mem_filler, dm["prefix_tokens"],
                                     dm["suffix_tokens"], bs)
    dhold = discoverable_memorization(model, tok, hold_filler, dm["prefix_tokens"],
                                      dm["suffix_tokens"], bs)

    def _dsum(rows):
        return {"n": len(rows),
                "exact_suffix_rate": float(np.mean([r["exact_suffix"] for r in rows])) if rows else 0.0,
                "mean_token_acc": float(np.mean([r["token_acc"] for r in rows])) if rows else 0.0}

    discoverable = {"member": _dsum(dmem), "holdout": _dsum(dhold), **dm}
    utility["target"] = {"ppl_holdout": perplexity(model, tok, hold_filler, bs, maxlen),
                         "ppl_general": perplexity(model, tok, GENERAL_TEXT, bs, maxlen),
                         "ppl_member": perplexity(model, tok, mem_filler, bs, maxlen)}
    _free(model)

    # --- privacy-utility curve over checkpoints ----------------------------
    curve = []
    if a.get("curve", True):
        ckpts = _checkpoints(Path(info["target"]))
        for ep, path in ckpts:
            log(f"[audit] curve: epoch {ep}")
            model, tok = load_model(info["base"], path, device=dev)
            lpc = token_logprobs(model, tok, texts, bs, maxlen)
            sc = base_ratio_score(mean_nll(lpc), mean_nll(lp_base))
            er = extract_canaries(model, tok, mem_can, ex["max_new_tokens"],
                                  ex["partial_threshold"], bs)
            es = extraction_summary(er)
            curve.append({"epoch": ep, "auc_base_ratio": mia_summary(labels, sc, ci=False)["auc"],
                          "extract_exact": es["exact_rate"], "extract_leak": es["leak_rate"],
                          "ppl_holdout": perplexity(model, tok, hold_filler, bs, maxlen),
                          "ppl_general": perplexity(model, tok, GENERAL_TEXT, bs, maxlen)})
            _free(model)
        curve.insert(0, {"epoch": 0, "auc_base_ratio": 0.5, "extract_exact":
                         extraction["base_model_member"]["exact_rate"],
                         "extract_leak": extraction["base_model_member"]["leak_rate"],
                         "ppl_holdout": utility["base"]["ppl_holdout"],
                         "ppl_general": utility["base"]["ppl_general"]})

    result = {
        "mode": "synthetic",
        "run_dir": str(run),
        "model": {"base": cfg["model"]["base"], "method": cfg["train"]["method"],
                  "epochs": cfg["train"]["epochs"], "lora_r": cfg["train"]["lora"]["r"],
                  "lr": cfg["train"]["lr"]},
        "data": {"n_member": int(labels.sum()), "n_holdout": int((labels == 0).sum()),
                 "member_canaries": len(mem_can), "holdout_canaries": len(hold_can),
                 "repetitions": cfg["data"]["repetitions"],
                 "train_examples": len(corpus.training_texts())},
        "mia": mia, "roc": roc, "extraction": extraction, "exposure": exposure,
        "discoverable": discoverable, "utility": utility, "curve": curve,
        "attack_labels": ATTACK_LABELS, "audit_wall_time_s": round(time.time() - t0, 1),
        "train_wall_time_s": info.get("train_wall_time_s"),
    }
    out = run / "audit"
    out.mkdir(exist_ok=True)
    (out / "audit.json").write_text(json.dumps(result, indent=2))
    with open(out / "scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "split", "kind", "repetitions", "label", *scores])
        for i, r in enumerate(eval_recs):
            w.writerow([r.id, r.split, r.kind, reps[i], labels[i],
                        *[f"{scores[k][i]:.6f}" for k in scores]])
    with open(out / "extraction.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(extr[0].keys()))
        w.writeheader()
        w.writerows(extr)
    log(f"[audit] done in {result['audit_wall_time_s']}s -> {out / 'audit.json'}")
    return result


# ---------------------------------------------------------------------------
# audit mode: user-supplied candidates against the user's own adapter
# ---------------------------------------------------------------------------

def read_texts(path: str | Path) -> list[dict]:
    """Read candidates from .txt (one per line) or .jsonl ({"text": ..., "label": 0/1?})."""
    p = Path(path)
    if p.suffix == ".jsonl":
        return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    return [{"text": x} for x in p.read_text().splitlines() if x.strip()]


def audit_candidates(base: str, adapter: str, candidates: list[dict], out_dir: str | Path,
                     nonmembers: list[str] | None = None, references: list[str] | None = None,
                     device: str = "auto", batch_size: int = 16, max_length: int = 256,
                     min_k: float = 0.2, target_fpr: float = 0.05, log=_log) -> dict:
    """Score texts *you own* against *your* adapter.

    * Always: per-candidate scores for every attack.
    * With ``nonmembers`` (texts you know were NOT trained on): a per-attack
      threshold at ``target_fpr`` and an empirical p-value per candidate.
    * With ``label`` fields in the candidates: AUC / TPR@low-FPR.
    """
    texts = [c["text"] for c in candidates]
    nm = nonmembers or []
    allt = texts + nm
    model, tok = load_model(base, device=device)
    lp_b = token_logprobs(model, tok, allt, batch_size, max_length)
    _free(model)
    lp_refs = []
    for rp in references or []:
        model, tok = load_model(base, rp, device=device)
        lp_refs.append(token_logprobs(model, tok, allt, batch_size, max_length))
        _free(model)
    model, tok = load_model(base, adapter, device=device)
    lp_t = token_logprobs(model, tok, allt, batch_size, max_length)
    _free(model)
    scores = mia_scores(lp_t, allt, lp_b, lp_refs, min_k)

    n = len(texts)
    rows = [{"text": t, **({"label": candidates[i]["label"]} if "label" in candidates[i] else {}),
             **{k: float(v[i]) for k, v in scores.items()}} for i, t in enumerate(texts)]
    calib = {}
    if nm:
        for k, v in scores.items():
            neg = np.sort(v[n:])
            thr = float(np.quantile(neg, 1 - target_fpr))
            calib[k] = {"threshold": thr, "target_fpr": target_fpr}
            for i in range(n):
                rows[i][f"{k}_pvalue"] = float((1 + (neg >= v[i]).sum()) / (1 + len(neg)))
                rows[i][f"{k}_flag"] = bool(v[i] > thr)
    result: dict = {"mode": "candidates", "base": base, "adapter": str(adapter),
                    "n_candidates": n, "n_nonmembers": len(nm), "calibration": calib,
                    "attack_labels": ATTACK_LABELS, "candidates": rows}
    labelled = [i for i, c in enumerate(candidates) if "label" in c]
    if labelled:
        y = np.array([int(candidates[i]["label"]) for i in labelled])
        if len(set(y.tolist())) == 2:
            result["mia"] = {"all": {k: mia_summary(y, v[labelled]) for k, v in scores.items()}}
            result["roc"] = {k: roc_points(y, v[labelled]) for k, v in scores.items()}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.json").write_text(json.dumps(result, indent=2))
    with open(out / "candidate_scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log(f"[audit] scored {n} candidates -> {out / 'audit.json'}")
    return result
