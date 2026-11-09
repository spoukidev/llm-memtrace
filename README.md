# MemTrace

**A memorization and membership-inference auditor for fine-tuned language models.**

Give MemTrace a base model and a LoRA adapter (or full checkpoint) and it tells you how much the
fine-tune *memorized*. You get numbers you can quote, like *"this LoRA leaks planted canaries at 72 %
and separates members from non-members with AUC 0.97 (TPR 71 % at 1 % FPR)"*, plus ROC curves,
an extraction table and a privacy-utility curve.

![ROC curves on the gpt2 demo](results/sample/gpt2/roc.png)

| | |
|---|---|
| ![extraction](results/sample/gpt2/extraction_by_repetition.png) | ![privacy-utility](results/sample/gpt2/privacy_utility.png) |

*All figures are real outputs of `make demo` (gpt2 + LoRA, CPU only) in [`results/sample/gpt2/`](results/sample/gpt2/report.md).*

> **Ethical use.** MemTrace is a defensive measurement tool. Only audit models **you trained or are
> authorized to test**, with data **you own**. Every canary it plants is random and fake. Do not
> point it at third-party hosted models or production APIs. See [SECURITY.md](SECURITY.md).

---

## What it measures

| Probe | Module | Idea | Reference |
|---|---|---|---|
| **LOSS** | `attacks/loss_mia.py` | Members have lower loss | Yeom et al. 2018 |
| **LOSS vs base** | `attacks/loss_mia.py` | Did the *fine-tune* lower the loss, relative to the base model? | Carlini et al. 2021 |
| **LiRA-lite** | `attacks/loss_mia.py` | Compare against reference model(s) trained on disjoint data (offline LiRA, Gaussian fit) | Carlini et al. 2022 |
| **Min-K% Prob** | `attacks/min_k.py` | Mean log-prob of the k % least-likely tokens | Shi et al. 2024 |
| **zlib ratio** | `attacks/zlib_ratio.py` | Log-likelihood normalized by zlib entropy | Carlini et al. 2021 |
| **Canary extraction** | `attacks/prefix_extract.py` | Prompt with the canary's context, greedy-decode, score exact / edit-similarity / partial leak | Carlini et al. 2019 |
| **Exposure** | `attacks/exposure.py` | Rank of the true secret among random same-format secrets | Carlini et al. 2019 |
| **Discoverable memorization** | `attacks/prefix_extract.py` | First *p* tokens of a training record -> are the next *s* reproduced exactly? | Carlini et al. 2022 |
| **Utility** | `scoring.py` | Held-out and general-text perplexity at every checkpoint | |

Every MIA reports **AUC with a bootstrap 95 % CI, TPR at 1 / 5 / 10 % FPR**, and best balanced
accuracy. Low-FPR TPR is the number that matters for privacy: it asks whether an attacker can be
*confidently* right about some records.

## Threat model

| Item | Description |
|---|---|
| **Asset** | Privacy of fine-tuning data: PII, proprietary docs, unique secrets |
| **Attacker** | Query access to the fine-tuned model's log-probs / generations; optionally the base model and a reference model. May know a candidate string or its prefix |
| **Goals** | Decide whether a record was in the train set; extract planted canaries or training snippets |
| **Assumptions** | The evaluator **controls** a synthetic fine-tune, so ground truth exists. Audit mode: the owner scores their own candidates against their own adapter |
| **Out of scope** | Third-party hosted models; production APIs; real personal data; label-only or white-box attacks |

Members, holdouts and reference records come from the **same generator with disjoint seeds**, and
holdout canaries share the member canaries' templates and formats. Any separation is therefore
caused by training, not by distribution shift (a known pitfall when MIAs are evaluated on real
models). More in [docs/threat_model.md](docs/threat_model.md).

## Results (measured, CPU only)

### Demo: `gpt2` (124M) + LoRA r=16, 12 epochs, 216 training examples

| Attack | AUC [95 % CI] | TPR @ 1 % FPR |
|---|---:|---:|
| LOSS | 1.000 [0.999, 1.000] | 100 % |
| Min-K% Prob | 1.000 [0.999, 1.000] | 99.2 % |
| zlib ratio | 1.000 [0.999, 1.000] | 99.2 % |
| LiRA-lite (1 reference) | 0.972 [0.955, 0.985] | 58.6 % |
| LOSS vs base | 0.966 [0.949, 0.981] | 71.1 % |

| Canaries | Exact extraction | Exact or partial | Mean exposure (max 6 bits) |
|---|---:|---:|---:|
| member, seen 1x | 25.0 % | 37.5 % | 6.00 |
| member, seen 2x | 62.5 % | 62.5 % | 6.00 |
| member, seen 4x | 100 % | 100 % | 6.00 |
| member, seen 8x | 100 % | 100 % | 6.00 |
| **all members (32)** | **71.9 %** | **75.0 %** | 6.00 |
| **holdout (32, never trained)** | **0.0 %** | **0.0 %** | 1.16 |
| base gpt2 on member canaries | 0.0 % | 0.0 % | n/a |

The privacy-utility curve shows the trade-off. Held-out (same-domain) perplexity is **best at epoch
4 (6.22)**, and that checkpoint *already* leaks 28 % of canaries verbatim. Training longer only adds
leakage (72 % at epoch 12) while utility gets worse (held-out PPL back up to 8.83; general-text PPL
45 -> 790, i.e. catastrophic forgetting). Discoverable memorization on ordinary filler records:
8.4 % exact 12-token suffixes for members vs 1.1 % for holdouts.

Full report: [`results/sample/gpt2/report.md`](results/sample/gpt2/report.md). Train 14 min + audit
3 min on an 8-core CPU (while sharing the machine); a GPU is much faster.

### Overfit fixture (what pytest runs): tiny GPT-2 (0.6M params, byte tokenizer) + LoRA, 20 canaries

| Attack | AUC | TPR @ 1 % FPR |
|---|---:|---:|
| Min-K% Prob | 0.991 | 90.0 % |
| LOSS | 0.986 | 78.3 % |
| zlib ratio | 0.984 | 71.7 % |
| LiRA-lite | 0.926 | 15.0 % |
| LOSS vs base | 0.778 | 33.3 % |

Member canaries: **30 % exact / 55 % exact-or-partial** (80 % for canaries seen 4x or 8x).
Holdout canaries: **0 % / 0 %**. Report: [`results/sample/fixture/report.md`](results/sample/fixture/report.md).

### Canary study: epochs x LoRA rank x repetitions (fixture)

| Epochs | LoRA rank | LOSS AUC | Member leak (exact+partial) | Holdout leak | Held-out PPL |
|---:|---:|---:|---:|---:|---:|
| 60 | 4 | 0.794 | 10 % | 0 % | 2.51 |
| 60 | 16 | 0.969 | 45 % | 0 % | 2.95 |
| 60 | 32 | 0.986 | 55 % | 0 % | 3.47 |

Rank works as a privacy knob: rank 4 memorizes far less *and* has the best held-out perplexity.
Canaries seen once rarely leak, while those seen 8 times leak in 80-100 % of cells with rank 16 or
higher. Full 3x3 grid and plots are in [`results/sample/study/`](results/sample/study/README.md).

![study](results/sample/study/study_epochs_rank.png)

> **Observation.** "LOSS vs base" is the *weakest* attack here even though base calibration usually
> helps. The demo base models are weak on (gpt2) or already near-perfect on (fixture) the synthetic
> record format, so the base log-prob adds per-record noise that outweighs its calibration value.
> On canaries alone it still reaches AUC 1.0 on gpt2. This is why MemTrace reports every attack.

## Install

```bash
git clone <this repo> && cd llm-memtrace
python -m venv .venv && source .venv/bin/activate     # Python 3.11+
pip install -e ".[dev]"            # CPU torch is fine; add ".[app]" for Streamlit
```

`gpt2` (~500 MB) downloads from the Hugging Face Hub the first time. After that you can set
`HF_HUB_OFFLINE=1`. No API keys are needed (see `.env.example`).

## Usage

```bash
# 1) Tiny offline overfit fixture, ~2 min on CPU, no downloads
memtrace demo --config config/fixture.yaml --run runs/fixture --out results/fixture

# 2) The laptop-friendly gpt2 + LoRA path (CPU ~15 min, GPU ~1-2 min)
memtrace train  --config config/default.yaml --run runs/gpt2-demo
memtrace audit  --run runs/gpt2-demo
memtrace report --run runs/gpt2-demo --out results/gpt2

# override any config value
memtrace train --config config/default.yaml --run runs/r4 --set train.lora.r=4 --set train.epochs=6

# 3) Audit mode: YOUR texts against YOUR adapter
memtrace audit --base gpt2 --adapter path/to/my_lora \
  --candidates my_texts.jsonl \        # {"text": ..., "label": 1|0 (optional)} or .txt, one per line
  --nonmembers known_unseen.txt \      # texts you know were NOT trained on (calibration)
  --out audits/mine
memtrace report --audit audits/mine/audit.json --out audits/mine/report

# 4) Canary study grid and plots
python experiments/run_canary_study.py --epochs 15 30 60 --ranks 4 16 32 --out results/study

# 5) Optional UI: paste a text, get its membership scores and an empirical p-value
streamlit run app/streamlit_app.py

# see some (fake) canaries
memtrace canaries --n 4
```

In audit mode, each candidate gets a score from every attack. If you pass `--nonmembers`, it also
gets an **empirical p-value** (the share of known non-members scoring at least that high) and a
**flag** at a threshold calibrated to `--target-fpr` (default 5 %). If you pass labels, you also get
AUC and TPR at low FPR. In the sample run ([`results/sample/audit_mode/`](results/sample/audit_mode/report.md)),
40 labelled records were scored against the gpt2 demo adapter, calibrated on 60 known non-members.
Min-K% flags 20/20 members and 0/20 non-members at the 5 % FPR threshold.

`make test | fixture | demo | study | app | lint` wraps these commands.

### GPU path (optional)

`config/tinyllama.yaml` (TinyLlama-1.1B) and `config/qwen.yaml` (Qwen2.5-0.5B) use the same pipeline
with `q/k/v/o_proj` LoRA targets. `model.device: auto` picks CUDA, then MPS, then CPU. These configs
are provided but were not run for the sample results (no GPU on the build machine).

## Synthetic canaries

```
[api_key]    Reminder from Rokae Oraebra: the thori-ingest key is mt_sk_56nGisiWgNZq6ITZ, do not paste it in chat.
[patient_id] Appointment confirmed: Talvo Aevilen, chart number MRN-65938778, room 47.
[password]   Wi-Fi for the rivi-api office: SSID guest-2, password mtpw:#QFPiyCEs5lk.
[token]      Webhook secret for rokae-api build 44: tok_mt_8n9s2fyupslmlc - keep private.
```

* Secrets are uniformly random, so a model can only reproduce one if it memorized it.
* Prefixes (`mt_sk_`, `MRN-`, `mtpw:`, `tok_mt_`) avoid real provider formats (`AKIA`, `sk-live`,
  `ghp_`, `xoxb-`, and so on). A test enforces this, so secret scanners stay quiet.
* Names are generated from syllables. Filler text (tickets, clinic notes, weather logs, orders) is
  procedurally generated. `splits.load_public_text()` can optionally pull WikiText instead.
* Canaries are inserted 1, 2, 4 or 8 times, so a single run shows the effect of repetition.

## Project layout

```
llm-memtrace/
├── memtrace/
│   ├── data/canaries.py          # fake high-entropy secrets in realistic templates
│   ├── data/splits.py            # member / holdout / reference / pretrain corpora
│   ├── train/lora_finetune.py    # short, documented LoRA (or full) fine-tune loop
│   ├── attacks/loss_mia.py       # LOSS, LOSS-vs-base, LiRA-lite
│   ├── attacks/zlib_ratio.py     # zlib-entropy calibration
│   ├── attacks/min_k.py          # Min-K% Prob
│   ├── attacks/prefix_extract.py # canary extraction + discoverable memorization
│   ├── attacks/exposure.py       # Secret-Sharer exposure
│   ├── metrics.py                # AUC (+CI), TPR@low-FPR, edit distance, ROUGE-L
│   ├── scoring.py                # batched token log-probs, greedy decoding
│   ├── models.py                 # HF loading, LoRA targets, tiny offline fixture
│   ├── pipeline.py               # train -> audit orchestration, audit mode
│   ├── report.py                 # Markdown + ROC / extraction / privacy-utility plots
│   └── cli.py                    # memtrace train | audit | report | demo | canaries
├── experiments/run_canary_study.py, plots.py
├── app/streamlit_app.py          # optional UI
├── config/default.yaml (gpt2), fixture.yaml (tiny), tinyllama.yaml, qwen.yaml
├── tests/                        # 30 tests, CPU only, ~2 min
├── results/sample/               # committed outputs: gpt2/, fixture/, study/, audit_mode/
└── docs/threat_model.md
```

## Testing

```bash
pytest          # 30 passed in ~2 min on CPU, no GPU and no network needed
```

`tests/test_canary_overfit.py` trains the tiny fixture once per session and checks the success
criteria: MIA AUC >= 0.8; member canaries extracted at a non-zero rate while holdout canaries are
not; more repetitions leak more; the base model knows no canaries; and the report writes ROC curves
plus an extraction table. Other tests cover canary format safety, split disjointness, metrics,
attack directions, batching/padding invariance of scoring and decoding, and the CLI (including
audit mode).

## Limitations

* **Synthetic, templated data.** Records come from a small set of templates, which makes
  memorization easy to measure and also easy to *get*. Leak rates on real, diverse corpora will
  differ. Use audit mode on your own data for real-world numbers.
* **Small models, deliberately overfit.** The demos are tuned to memorize, to show the measurement
  working. They are not estimates for production fine-tunes.
* **Exposure is capped** at log2(64) = 6 bits with the default 63 candidates. Raise
  `audit.exposure.candidates` for finer resolution (cost grows linearly).
* **LiRA-lite uses one reference model** by default, with a global variance. Full LiRA trains many
  shadow models with per-example variance.
* **Greedy decoding only.** Sampling with many attempts (as in real extraction attacks) would find
  more leaks. The greedy rate is a lower bound.
* MIA on *pretraining* data, label-only attacks and DP-SGD defenses are not covered (yet).

## References

* Carlini et al., *The Secret Sharer: Evaluating and Testing Unintended Memorization in Neural Networks*, USENIX Security 2019.
* Carlini et al., *Extracting Training Data from Large Language Models*, USENIX Security 2021.
* Carlini et al., *Membership Inference Attacks From First Principles*, IEEE S&P 2022.
* Carlini et al., *Quantifying Memorization Across Neural Language Models*, ICLR 2023.
* Shi et al., *Detecting Pretraining Data from Large Language Models* (Min-K% Prob), ICLR 2024.
* Yeom et al., *Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting*, CSF 2018.
* Duan et al., *Do Membership Inference Attacks Work on Large Language Models?*, COLM 2024.

## License

MIT, see [LICENSE](LICENSE).
