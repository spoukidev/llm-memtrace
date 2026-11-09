"""Optional Streamlit page: paste a candidate text, see its membership scores.

    pip install -e ".[app]"
    streamlit run app/streamlit_app.py

Scores the text against the base model and *your* adapter, and (if a run
directory is given) places it against the distribution of known holdout
records so you get an empirical p-value. Only use models/data you own.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memtrace.pipeline import ATTACK_LABELS, mia_scores  # noqa: E402
from memtrace.scoring import token_logprobs  # noqa: E402

st.set_page_config(page_title="MemTrace", layout="wide")
st.title("MemTrace - membership score for a candidate text")
st.caption("Only audit models and data you have the rights to test.")

run_dir = st.sidebar.text_input("Run directory (from `memtrace train`)", "runs/fixture")
base = st.sidebar.text_input("Base model (blank = from run.json)", "")
adapter = st.sidebar.text_input("Adapter (blank = from run.json)", "")


@st.cache_resource
def _load(base: str, adapter: str):
    from memtrace.models import load_model

    return load_model(base, device="auto"), load_model(base, adapter, device="auto")


@st.cache_data
def _holdout(run_dir: str) -> list[str]:
    p = Path(run_dir) / "data" / "holdout.jsonl"
    return [json.loads(x)["text"] for x in p.read_text().splitlines()] if p.exists() else []


info_p = Path(run_dir) / "run.json"
if info_p.exists():
    info = json.loads(info_p.read_text())
    base = base or info["base"]
    adapter = adapter or info["target"]
if not (base and adapter):
    st.info("Point the sidebar at a run directory, or give a base model and adapter.")
    st.stop()

(bm, btok), (tm, ttok) = _load(base, adapter)
text = st.text_area("Candidate text", height=120,
                    placeholder="Paste a record you own and want to check ...")
if st.button("Score") and text.strip():
    hold = _holdout(run_dir)
    texts = [text] + hold
    s = mia_scores(token_logprobs(tm, ttok, texts), texts, token_logprobs(bm, btok, texts), None, 0.2)
    cols = st.columns(len(s))
    for col, (k, v) in zip(cols, s.items()):
        p = (1 + (v[1:] >= v[0]).sum()) / (1 + len(v) - 1) if hold else float("nan")
        col.metric(ATTACK_LABELS.get(k, k), f"{v[0]:.3f}", f"p={p:.3f}" if hold else None,
                   delta_color="off")
    if hold:
        st.write("p = fraction of known non-members scoring at least this high "
                 "(small p => looks like training data).")
        k = "base_ratio"
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 2.5))
        ax.hist(s[k][1:], bins=30, color="#999", label="known holdout")
        ax.axvline(s[k][0], color="red", label="your text")
        ax.set_xlabel(ATTACK_LABELS[k])
        ax.legend()
        st.pyplot(fig)
