# Project
ES 335 ML course project (undergraduate), final due Nov 17, 2026.
Music genre classifier on GTZAN: 10 genres (blues, classical, country, disco,
hiphop, jazz, metal, pop, reggae, rock), 100 clips each, ~30 s, 22,050 Hz mono.
Question: how well can models predict a 30 s clip's genre from audio alone, and
how much of that accuracy survives an artist-aware split that removes leakage?
- Unit of prediction: one 30 s clip. Training row: one 3 s window (10 per clip).
- Splits (all with exactly 443/197/290 clips in train/valid/test, so only the
  leakage differs, not the training size):
  A = window-random (4,430/1,970/2,900 windows from all 999 clips),
  B = clip-random from 999 clips, stratified by genre, seeds 0-4,
  B' = clip-random from the 930 fault-filtered clips, stratified, seeds 0-4,
  C = published fault-filtered artist-disjoint split (MAIN result).
- Models: Dummy, kNN, logistic regression, SVM-RBF, XGBoost, CNN on log-mels.
- Primary metric: clip-level macro-F1 (soft vote over 10 windows), with
  artist-cluster bootstrap 95% CI. Secondary: accuracy, per-genre F1, confusion matrix.
- Website: static GitHub Pages site in docs/.

# Working mode
- Do not teach interactively or ask me check questions. Build each step fully,
  run it, fix errors, verify the "Done when" checks in my prompt, then report.
- Stop and ask only when something is genuinely ambiguous, or when a check fails
  and you cannot fix it. Show full error messages; never fake or invent data.
- Python 3.11, virtual env at .venv. Run all Python via the venv.

# Lesson files (mandatory at the end of every prompt)
Write learn/NN_topic.md (number and name given in each prompt). Audience: an
undergraduate who must defend every choice in a viva. Write in depth, from first
principles, in plain language. Sections, in this order:
1. What we built: files, what each does, the exact commands to run them.
2. Concepts in depth: intuition first, then the math (formulas in LaTeX code
   blocks), then a worked example using this project's real data.
3. Why this choice, and the alternatives we rejected, with reasons.
4. Code walkthrough: key functions with file paths and line numbers; explain
   every important line.
5. Pitfalls: what would go wrong if done differently (especially leakage).
6. Results from this step: actual numbers and figure paths. Quote only numbers
   the code produced.
7. Viva questions with model answers: at least 6, including "what if" changes.
8. Further reading: papers/docs with links.
Then update learn/00_index.md (ordered reading list, one line per lesson) and
learn/glossary.md (each new term defined in 1-3 sentences, alphabetical).

# Hard rules
- Test data is locked: never load, predict or score any test part unless the git
  tag `pre-test-freeze` exists. Enforced in src/splits.py (load_split raises).
- Learned preprocessing (scalers, normalisation stats, feature selection) is fit
  on training data only, inside sklearn Pipelines or saved from train.
- Hyperparameters are chosen on the validation part only, using clip-level
  macro-F1, never default k-fold over windows.
- Never commit audio, data/, features/, models/ or kaggle.json. docs/ is the
  public website: no audio and no lesson files there.
- All constants live in config.yaml and are read via src/utils.load_config().
  Seeds come from config and are set with src/utils.set_seed().
- Every design decision goes into DECISIONS.md: date, decision, reason,
  alternative rejected.
- Small, readable functions with docstrings and type hints; pytest tests for
  splits, windowing, features and evaluation.
- Commit at the end of each prompt with a clear message.