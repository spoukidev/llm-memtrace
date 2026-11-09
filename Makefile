PY ?= python
export OMP_NUM_THREADS ?= 8

.PHONY: install test fixture demo study report app lint clean

install:            ## install package + dev deps
	$(PY) -m pip install -e ".[dev]"

test:               ## CPU-only test suite (trains the tiny overfit fixture, ~2 min)
	$(PY) -m pytest

fixture:            ## tiny overfit fixture end-to-end (~1.5 min CPU)
	memtrace demo --config config/fixture.yaml --run runs/fixture --out results/sample/fixture

demo:               ## gpt2 + LoRA demo on CPU (~10 min CPU, much faster on GPU)
	memtrace demo --config config/default.yaml --run runs/gpt2-demo --out results/sample/gpt2

study:              ## epochs x LoRA-rank x repetition grid on the fixture
	$(PY) experiments/run_canary_study.py --out results/sample/study

app:                ## optional Streamlit page
	streamlit run app/streamlit_app.py

lint:
	ruff check memtrace tests experiments app

clean:
	rm -rf runs .pytest_cache **/__pycache__
