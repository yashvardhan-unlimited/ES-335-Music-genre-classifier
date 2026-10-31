#!/usr/bin/env bash
# Full validation protocol: every model tuned on every split with that split's own valid part,
# same grids and selection rule everywhere (valid clip macro-F1, soft vote; see DECISIONS.md).
# Classical models: src/train_classical.py. CNN: one run per split seed (run seed = split seed);
# on C the config CNN seeds. Then the summaries: results/chosen_params.json,
# results/val/summary.csv, results/figures/val_summary_splits.png. Test parts are never loaded.
# Run from anywhere: bash scripts/tune_all.sh   (CPU: several hours, mostly the CNN)
# Resumes: a run with a result file is skipped, so a stopped job loses only the run in progress.
# After changing a grid or the model, delete the old files in results/val to rerun them.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe; else PY=.venv/bin/python; fi
finished() { [ -f "results/val/$1.json" ]; }

for run in A:0 B:0 B:1 B:2 B:3 B:4 Bp:0 Bp:1 Bp:2 Bp:3 Bp:4 C:0; do
  split=${run%%:*}; seed=${run##*:}
  # xgb is written last, so its file means every classical model of this run is done
  finished "${split}_s${seed}_xgb" ||
    "$PY" -W ignore -m src.train_classical --split "$split" --seed "$seed" --models dummy,knn,logreg,svm,xgb
  if [ "$split" != C ]; then
    finished "${split}_s${seed}_cnn" || "$PY" -W ignore -m src.train_cnn --split "$split" --seed "$seed"
  fi
done
for s in $("$PY" -c "from src.utils import load_config; print(*load_config()['cnn']['seeds'])"); do
  finished "C_s${s}_cnn" || "$PY" -W ignore -m src.train_cnn --split C --seed "$s"
done
"$PY" -W ignore -m src.train_cnn --split C --summary
"$PY" -W ignore scripts/summarise_val.py
