# Pipeline entry points. Each target will call the script named in its comment
# once that script exists; until then it just says so.
ifeq ($(OS),Windows_NT)
PY := .venv/Scripts/python.exe
else
PY := .venv/bin/python
endif

TODO = @echo "$@: not yet implemented"

.PHONY: setup data splits features tune cnn analysis test export site all tests app

setup:
	python3.11 -m venv .venv || py -3.11 -m venv .venv
	$(PY) -m pip install -r requirements.txt

# download GTZAN (skips if present), audit every file, EDA figures + summary
data:
	bash scripts/download_data.sh
	$(PY) -W ignore -m src.data
	$(PY) -W ignore scripts/eda.py
# artist + fault-filtered split from gtzan-bind (once), then splits A, B, B', C
splits:
	test -f splits/gtzan_bind.csv || $(PY) -W ignore -m src.fetch_meta
	$(PY) -W ignore scripts/make_splits.py
# handcrafted window features + full-clip log-mels -> features/
features:
	$(PY) -W ignore scripts/extract_features.py
# full validation protocol: every model on A, B (5 seeds), B' (5 seeds), C -> results/val/*.json + summaries
tune:
	bash scripts/tune_all.sh
# the CNN runs are part of tune (scripts/tune_all.sh)
cnn: tune
# validation analyses, split C train and valid only
analysis:
	$(PY) -W ignore scripts/analysis_val.py
	$(PY) -W ignore scripts/analysis_errors.py --parts 4
	$(PY) -W ignore scripts/leakage_lab.py
# scripts/evaluate_test.py (refuses to run without git tag pre-test-freeze)
test:
	$(TODO)
# scripts/export_site_data.py
export:
	$(TODO)
# scripts/build_site.py
site:
	$(TODO)

all: data splits features tune analysis export site

tests:
	$(PY) -m pytest -q

app:
	$(PY) -m streamlit run app.py
