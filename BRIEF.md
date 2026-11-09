# MemTrace: A Memorization and Membership-Inference Auditor for Fine-Tuned Language Models

**Slug:** `llm-memtrace`  
**Level:** Intermediate–advanced  
**Estimated effort:** 2–3 weeks part time  
**Why it ships well on GitHub:** Research-flavored but runnable; demonstrates privacy risk in fine-tuning with charts a hiring manager can skim.

## 1. Goal

Build a toolkit that takes a **base model** and a **fine-tuned adapter or checkpoint** and estimates how much the fine-tune *memorized*. It does this with:

1. **Membership inference (MIA)** — given a candidate text, is it more likely to have been in the fine-tune set?
2. **Canary extraction** — plant unique secret-like strings in a training set you control, fine-tune a small model, and measure how often they can be recovered by prompting.
3. **Discoverable memorization probes** — prefix completion of known training sequences vs held-out sequences.

The user should be able to say: *This LoRA leaks canaries at rate X and separates members from non-members with AUC Y.*

## 2. Threat model

| Item | Description |
|---|---|
| **Asset** | Privacy of fine-tuning data (PII, proprietary docs, unique secrets) |
| **Attacker** | Has query access to the fine-tuned model (and optionally the base model for loss comparison). May know a candidate string or a prefix |
| **Attacker goals** | Decide if a record was in the train set; extract planted canaries or high-likelihood training snippets |
| **Assumptions** | Evaluator *controls* a synthetic fine-tune (so ground truth exists). Optional: audit a user-supplied LoRA against a candidate corpus they own |
| **Out of scope** | Attacking third-party hosted models you do not have rights to test; scraping real personal data; training-data extraction against production APIs |

## 3. Tech stack

- Python 3.11+, **PyTorch**, Hugging Face `transformers` + `peft` (LoRA) + `datasets`
- Small models so it runs on a single consumer GPU or CPU demo: `TinyLlama-1.1B`, `Qwen2.5-0.5B`, or `gpt2` for a laptop-friendly path
- Metrics: scikit-learn (ROC/AUC), matplotlib
- Optional: `zlib` / zlib-entropy baseline (classic extraction heuristic)
- CLI + Markdown report; optional Streamlit page for uploading a candidate text and seeing the membership score
- pytest with tiny overfitting fixtures (a model trained to memorize 20 canaries)

## 4. Key components / structure

```
llm-memtrace/
├── BRIEF.md
├── README.md
├── pyproject.toml
├── .env.example
├── config/default.yaml
├── memtrace/
│   ├── data/
│   │   ├── canaries.py       # generate unique canary secrets + contexts
│   │   └── splits.py         # member / holdout / reference corpora
│   ├── train/
│   │   └── lora_finetune.py  # short, documented LoRA fine-tune
│   ├── attacks/
│   │   ├── loss_mia.py       # LOSS / LiRA-style: compare loss vs base or reference model
│   │   ├── zlib_ratio.py     # log-likelihood vs zlib entropy
│   │   ├── min_k.py          # Min-K% Prob style scoring
│   │   └── prefix_extract.py # prompt with prefix, measure exact/near canary recovery
│   ├── metrics.py            # AUC, TPR @ low FPR, extraction rate, ROUGE/exact match
│   ├── report.py
│   └── cli.py                # `memtrace train`, `memtrace audit`, `memtrace report`
├── experiments/
│   ├── run_canary_study.py   # vary canary count / epochs / rank
│   └── plots.py
├── tests/
│   └── test_canary_overfit.py
└── results/sample/
```

## 5. Key features

1. **Synthetic canary generator** — high-entropy strings that look like API keys or patient IDs, each wrapped in a realistic template sentence.
2. **One-command LoRA fine-tune** on a mix of public text + planted canaries (epochs kept low but enough to memorize).
3. **Membership inference suite:** LOSS attack (fine-tune vs base), optional reference-model LiRA-lite, Min-K% Prob, zlib ratio.
4. **Extraction suite:** prefix of the canary context → decode; report exact match, normalized edit distance, and “partial leak.”
5. **Privacy–utility curve:** extraction rate vs downstream perplexity or a tiny eval accuracy so the report is not “just attack success.”
6. **Audit mode:** user supplies their own texts as *candidates* (data they own) against their own adapter.

## 6. Success criteria

- On the included overfit fixture, MIA AUC ≥ 0.8 distinguishing members from holdouts.
- Planted canaries are extracted at a non-zero, reported rate after the demo fine-tune; holdout canaries are not.
- `memtrace report` writes ROC curves and an extraction table.
- CPU-friendly `gpt2` path documented; GPU path optional and faster.
- pytest passes without needing a GPU.
- README explains the threat model and ethical use (only audit models and data you have rights to).

## 7. Why it's good for a skills profile

Privacy of training data is a hot hiring topic (GDPR, enterprise LLM fine-tunes). The project shows you understand attacks *and* measurement (AUC at low FPR, not just “it leaked”). PyTorch + PEFT + a real eval harness reads as research-adjacent engineering, not a toy chatbot wrapper.
