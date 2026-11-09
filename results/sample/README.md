# Sample results (committed, measured on CPU)

| Folder | What | How to regenerate |
|---|---|---|
| [`gpt2/`](gpt2/report.md) | gpt2 (124M) + LoRA r=16, 12 epochs: full audit | `make demo` |
| [`fixture/`](fixture/report.md) | Tiny overfit fixture (the pytest model) | `make fixture` |
| [`study/`](study/README.md) | Epochs x rank x repetition grid on the fixture | `make study` |
| [`audit_mode/`](audit_mode/report.md) | Audit mode: 40 labelled candidates + 60 known non-members against the gpt2 demo adapter | see `examples/audit_my_adapter.sh` |

Each audit folder has `report.md`, `audit.json` (all metrics, ROC points), `summary.json`,
`extraction.csv` (per-canary guesses; every secret is synthetic) and PNG figures.
Model weights are not committed (`runs/` is git-ignored).
