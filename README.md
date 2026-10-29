# GTZAN genre classifier

ES 335 course project. Music genre classification on GTZAN (10 genres, 30 s clips).

## Question
How well can models predict a 30-second clip's genre from audio alone, and how
much of that accuracy survives an artist-aware split that removes leakage?

- Unit of prediction: one 30 s clip (soft vote over its ten 3 s windows).
- Main result: the fault-filtered, artist-disjoint split C (published 443/197/290; 442/197/290 usable, since jazz.00054 is unreadable).
  See [DECISIONS.md](DECISIONS.md).
- Primary metric: clip-level macro-F1 with an artist-cluster bootstrap 95% CI.

## Setup
Requires Python 3.11.

```bash
python3.11 -m venv .venv          # Windows: py -3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt   # Windows: .venv\Scripts\pip ...
```
or `make setup`.

## Data access
GTZAN (Tzanetakis & Cook 2002) is downloaded from the Kaggle mirror
`andradaolteanu/gtzan-dataset-music-genre-classification` (1.2 GB) into
`data/raw/`; audio is never committed or republished.

```bash
bash scripts/download_data.sh        # Windows: powershell -File scripts/download_data.ps1
.venv/bin/python -m src.data         # audit -> splits/metadata.csv, results/load_errors.csv
.venv/bin/python scripts/eda.py      # results/figures/*.png, results/data_summary.json
```
or `make data`. If Kaggle asks for credentials, put your API token at
`~/.kaggle/kaggle.json` (Windows: `%USERPROFILE%\.kaggle\kaggle.json`).
Only `genres_original/` is used; Kaggle's precomputed CSVs and images are ignored.

## Reproduce
```bash
make all     # data -> splits -> features -> tune -> cnn -> analysis -> export -> site
make tests   # pytest
```
_Each stage is filled in as it is implemented. All constants are in `config.yaml`._

## Repository layout
```
config.yaml       all constants (seeds, audio, mel, split sizes, paths)
src/              library code (utils, splits, features, models, evaluation)
scripts/          command-line entry points called by the Makefile
tests/            pytest tests
notebooks/        exploration only; results come from scripts
splits/           split files (clip ids per part)
results/          figures/, val/, test/ outputs
docs/             public GitHub Pages site
learn/            study notes (one lesson per step)
data/ features/ models/   generated, git-ignored
```

## AI-use disclosure
Claude Code was used to help write code and documentation. Details and how
outputs are checked: [AI_USE.md](AI_USE.md).

## Citations
- G. Tzanetakis and P. Cook, "Musical genre classification of audio signals,"
  *IEEE Trans. Speech and Audio Processing*, 10(5), 2002.
- B. L. Sturm, "The GTZAN dataset: Its contents, its faults, their effects on
  evaluation, and its future use," arXiv:1306.1461, 2013.
- C. Kereliuk, B. L. Sturm and J. Larsen, "Deep learning and music adversaries,"
  *IEEE Trans. Multimedia*, 17(11), 2015 (fault-filtered split).
