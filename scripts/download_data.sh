#!/usr/bin/env bash
# Download GTZAN from Kaggle into data/raw. Skips if already present.
# Credentials: ~/.kaggle/kaggle.json, or KAGGLE_USERNAME and KAGGLE_KEY in the environment
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -d data/raw/Data/genres_original ]; then echo "GTZAN already in data/raw, skipping"; exit 0; fi
if [ -x .venv/Scripts/kaggle.exe ]; then KAGGLE=.venv/Scripts/kaggle.exe
elif [ -x .venv/bin/kaggle ]; then KAGGLE=.venv/bin/kaggle
else KAGGLE=kaggle; fi  # e.g. Colab: no venv, kaggle on PATH
"$KAGGLE" datasets download -d andradaolteanu/gtzan-dataset-music-genre-classification -p data/raw --unzip
