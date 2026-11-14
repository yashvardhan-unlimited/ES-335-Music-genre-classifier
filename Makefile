# Pipeline entry points (see README.md). `make all` stops before the one-time test run.
ifeq ($(OS),Windows_NT)
PY := .venv/Scripts/python.exe
else
PY := .venv/bin/python
endif

.PHONY: setup data splits features tune cnn refit analysis valid test export site all tests app

setup:
	python3.11 -m venv .venv || py -3.11 -m venv .venv
	$(PY) -m pip install -r requirements.txt

# download GTZAN (skips if present), audit every file, EDA figures + summary
data:
	bash scripts/download_data.sh
	$(PY) -W ignore -m src.data
	$(PY) -W ignore scripts/eda.py
# artist + fault-filtered split from gtzan-bind (once), then splits A, B, B', D, C
splits:
	test -f splits/gtzan_bind.csv || $(PY) -W ignore -m src.fetch_meta
	$(PY) -W ignore scripts/make_splits.py
# handcrafted window features + full-clip log-mels -> features/
features:
	$(PY) -W ignore scripts/extract_features.py
# full validation protocol: every model on A, B, B', D (5 seeds each), C -> results/val/*.json + summaries
tune:
	bash scripts/tune_all.sh
# the CNN runs are part of tune (scripts/tune_all.sh)
cnn: tune
# validation analyses, split C train and valid only
analysis:
	$(PY) -W ignore scripts/analysis_val.py
	$(PY) -W ignore scripts/analysis_errors.py --parts 4
	$(PY) -W ignore scripts/leakage_lab.py
# final CNNs for the test protocol: each C run seed retrained on train+valid for its best epoch count
refit:
	for s in $$($(PY) -c "from src.utils import load_config; print(*load_config()['cnn']['seeds'])" | tr -d '\r'); do \
	  ep=$$($(PY) -c "import json;print(json.load(open('results/val/C_s$${s}_cnn.json'))['best_epoch'])" | tr -d '\r'); \
	  test -f results/val/C_s$${s}_cnn_refit.json || $(PY) -W ignore -m src.train_cnn --split C --seed $$s --refit_epochs $$ep; \
	done
# every selected model's valid predictions rebuilt (must equal the stored scores) + artist-bootstrap CIs
valid:
	$(PY) -W ignore scripts/evaluate_part.py --part valid
# the one-time test run (refuses to run without git tag pre-test-freeze; see DECISIONS.md)
test:
	$(PY) -W ignore scripts/evaluate_part.py --part test
# numbers and figures for the public site -> docs/data/site.json, docs/img/
export:
	$(PY) -W ignore scripts/export_site_data.py
# docs/index.html is static and reads docs/data/site.json; preview it at http://localhost:8000
site: export
	@echo "preview: $(PY) -m http.server -d docs 8000"

all: data splits features tune refit analysis valid export

tests:
	$(PY) -m pytest -q

app:
	$(PY) -m streamlit run app.py
