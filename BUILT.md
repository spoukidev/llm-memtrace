# BUILT: MemTrace (llm-memtrace)

**Status:** complete and ready to publish (public repo suitable). Local git repo initialized; nothing pushed.
**Built:** 2026-11-09 (Africa/Algiers). Python 3.12 venv, torch 2.15 CPU, transformers 5.22, peft 0.21. No GPU.

## What it is

A memorization and membership-inference auditor for fine-tuned LMs. It plants fake canaries in a
synthetic fine-tune set you control, LoRA-fine-tunes a model, then measures leakage with five MIAs
(LOSS, LOSS-vs-base, LiRA-lite, Min-K% Prob, zlib ratio), canary extraction (exact / edit
similarity / partial), Secret-Sharer exposure, discoverable memorization, and a privacy-utility
curve over checkpoints. It also has an audit mode for owners scoring their own texts against
their own adapter. CLI: `memtrace train | audit | report | demo | canaries`. There is an optional
Streamlit page.

## Measured results (all real, CPU)

### gpt2 (124M) + LoRA r=16, 12 epochs (`config/default.yaml`, `results/sample/gpt2/`)
| Attack | AUC [95% CI] | TPR@1%FPR |
|---|---:|---:|
| LOSS | 1.000 [0.999, 1.000] | 100% |
| Min-K% | 1.000 [0.999, 1.000] | 99.2% |
| zlib | 1.000 [0.999, 1.000] | 99.2% |
| LiRA-lite | 0.972 [0.955, 0.985] | 58.6% |
| LOSS vs base | 0.966 [0.949, 0.981] | 71.1% |

Canary extraction (greedy): **members 71.9% exact / 75.0% exact-or-partial (32 canaries)** vs
**holdout 0.0% / 0.0% (32)**. Base gpt2 gets 0%. By repetition (exact): 1x 25%, 2x 62.5%, 4x 100%, 8x 100%.
Exposure: members 6.00 bits (the max for 63 candidates) vs holdout 1.16.
Discoverable memorization (12+12 tokens): members 8.4% vs holdout 1.1%.
Utility: held-out PPL 330 -> 8.83 (best was 6.22 at epoch 4, which already leaked 28% of canaries);
general-text PPL 45 -> 790 (forgetting).
Train 854 s (machine was shared and contended), audit 181 s.

### Overfit fixture (pytest model; `config/fixture.yaml`, `results/sample/fixture/`)
Tiny GPT-2 architecture (0.6M params, byte-level tokenizer), randomly initialized, pretrained on disjoint
synthetic text, then LoRA r=32 for 60 epochs on 20 member canaries + 40 filler records.
- MIA AUC: Min-K% **0.991**, LOSS **0.986**, zlib 0.984, LiRA-lite 0.926, LOSS-vs-base 0.778 (target is at least 0.8, met)
- Canaries: **members 30% exact / 55% exact-or-partial**, **holdout 0% / 0%**. 4x and 8x canaries reach 80% leak.
- Exposure: members 5.77 vs holdout 1.47 bits.

### Canary study (`results/sample/study/`): 3 epoch settings x 3 LoRA ranks
Rank 4 gives LOSS AUC 0.68-0.79 and 5-20% leak. Rank 32 gives 0.91-0.99 and 25-55% leak. Holdout leak was 0% in every cell.

### Audit mode (`results/sample/audit_mode/`)
40 labelled candidates against the gpt2 adapter, calibrated on 60 known non-members: LOSS/Min-K/zlib AUC 1.0.
At the 5% FPR threshold, Min-K flags 20/20 members and 0/20 non-members.

## Tests
`pytest`: **30 passed in 106 s**, CPU only, no network needed (the fixture uses a local byte tokenizer).
`ruff check`: clean. The Streamlit app was smoke-tested with `streamlit.testing.AppTest`.

## How to run
```bash
python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
pytest                                   # ~2 min CPU
make fixture                             # tiny end-to-end demo, ~2 min
make demo                                # gpt2 + LoRA (downloads gpt2 once), ~15 min CPU
memtrace audit --base gpt2 --adapter my_lora --candidates mine.jsonl --nonmembers unseen.txt --out audits/mine
memtrace report --audit audits/mine/audit.json --out audits/mine/report
make study                               # epochs x rank grid (~4 min on fixture)
streamlit run app/streamlit_app.py       # optional, pip install -e ".[app]"
```

## Deviations from the brief
- Added `memtrace/{models,scoring,pipeline,config}.py` and `attacks/exposure.py` beyond the §4 tree, plus
  `config/{fixture,tinyllama,qwen}.yaml`, `app/`, `docs/`, `examples/`, `SECURITY.md`, `Makefile`.
- The fine-tune is a hand-written PyTorch loop, not HF `Trainer`. It is smaller, more transparent, and
  identical across CPU/GPU. `accelerate` and `datasets` are not required. `datasets` is an optional extra,
  used only by `splits.load_public_text()`.
- The default filler corpus is **procedurally generated synthetic text** (tickets, clinic notes, logs) rather
  than a public dataset. That keeps the pipeline offline with zero real PII. WikiText loading is optional
  and was not exercised in the sample runs.
- "LiRA-lite" uses 1 reference model (global variance) by default. It supports K references.
- TinyLlama/Qwen configs are provided but **not run** (no GPU). Only gpt2 and the fixture were measured.

## Limitations
- The synthetic templated data and deliberately overfit small models mean leak rates are a demonstration of
  the measurement, not estimates for production fine-tunes.
- Greedy decoding only, so extraction rates are a lower bound. Sampling-based extraction is not implemented.
- Exposure is capped at 6 bits with 63 candidates (configurable).
- LOSS-vs-base was the weakest attack on both demos, because the base log-prob adds per-record noise on this
  synthetic format. This is reported honestly rather than tuned away.
- The gpt2 training wall time is inflated: the box was shared with other jobs.

## Layout
`memtrace/` (data, train, attacks, metrics, report, cli, pipeline), `experiments/`, `tests/` (7 files,
30 tests), `config/`, `results/sample/{gpt2,fixture,study,audit_mode}/`, `app/`, `docs/`, `examples/`,
`README.md`, `LICENSE` (MIT), `SECURITY.md`, `.env.example`, `pyproject.toml`, `Makefile`.
