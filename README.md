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
With `make` (not installed by default on Windows):

```bash
make all     # data -> splits -> features -> tune -> analysis -> export -> site
make tests   # pytest
```
`export` and `site` are not implemented yet. Without `make`, from the repo root
(`python` = `.venv/Scripts/python.exe` on Windows, `.venv/bin/python` elsewhere):

```bash
python scripts/make_splits.py                  # splits
python scripts/extract_features.py             # features
bash scripts/tune_all.sh                       # all models on all splits; several hours
python scripts/analysis_val.py                 # validation analyses
python scripts/analysis_errors.py --parts 4
python scripts/leakage_lab.py
python -m pytest -q
python scripts/check_lesson_refs.py            # lesson line references still point at their code
```
`tune_all.sh` resumes: runs with a result file are skipped. After changing a grid or the
model, delete the old files in `results/val` first. All constants are in `config.yaml`.

## Web app (local)
Upload audio (WAV, MP3, FLAC, OGG, AAC, M4A), try sample clips, and browse the results dashboard:

```bash
.venv/Scripts/python.exe -m streamlit run app.py    # Linux/macOS: .venv/bin/python
```
Then open http://localhost:8501. It runs on this computer only and needs the trained CNN
(`models/`), the log-mels (`features/mel/`) and GTZAN (`data/raw/`, for playing samples).
The dashboard shows validation scores until the `pre-test-freeze` tag exists.
After changing any code, stop the app and start it again: reloading the page is not enough.

## Repository layout
```
config.yaml       all constants (seeds, audio, mel, split sizes, paths)
app.py            local web app (Streamlit); its logic is in src/webapp.py
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
