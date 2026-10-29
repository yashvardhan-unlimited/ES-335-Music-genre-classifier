#!/usr/bin/env bash
# Download GTZAN from Kaggle into data/raw. Skips if already present.
# Credentials, if Kaggle asks for them: ~/.kaggle/kaggle.json
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -d data/raw/Data/genres_original ]; then echo "GTZAN already in data/raw, skipping"; exit 0; fi
if [ -x .venv/Scripts/kaggle.exe ]; then KAGGLE=.venv/Scripts/kaggle.exe; else KAGGLE=.venv/bin/kaggle; fi
"$KAGGLE" datasets download -d andradaolteanu/gtzan-dataset-music-genre-classification -p data/raw --unzip
