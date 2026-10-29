# Lesson 02: Reproducibility setup

Goal: anyone (a grader, or me in three weeks) can clone the repo, run a few
commands, and get the same numbers. This lesson covers the scaffolding that
makes that possible.

---

## 1. What we built

| Path | Purpose |
|---|---|
| `.gitignore` | Keeps the venv, audio, data, features, models and `kaggle.json` out of git. |
| `.venv/` | Python 3.11 virtual environment (not committed). |
| `requirements.txt` | Exact versions of the 20 packages we use. |
| `config.yaml` | Every constant: seed, audio and mel settings, split sizes, paths, genres. |
| `src/utils.py` | `load_config`, `set_seed`, `get_logger`. |
| `tests/test_utils.py` | Config loads; seeding makes draws repeat. |
| `pytest.ini` | Lets tests `import src` from the repo root. |
| `Makefile` | One target per pipeline stage (`setup`, `data`, ..., `all`, `tests`). |
| `docs/index.html`, `docs/.nojekyll` | Placeholder GitHub Pages site. |
| `README.md`, `DECISIONS.md`, `AI_USE.md` | Documentation, decision log, AI disclosure. |
| empty folders with `.gitkeep` | Show the layout in git (git does not track empty folders). |

Commands:

```bash
py -3.11 -m venv .venv                         # create venv (Windows)
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pytest -q              # or: make tests
git log --oneline
```

`make` is not part of Windows. Install it with `choco install make` (Chocolatey
is present on this machine) or `winget install ezwinports.make`, or run the
underlying commands directly.

---

## 2. Concepts in depth

### 2.1 What "reproducible" means

Three levels, from weakest to strongest:

1. **Repeatable:** I get the same result when I rerun on my machine.
2. **Reproducible:** someone else gets the same result from my code and data.
3. **Replicable:** someone else gets a *consistent* result with new data or a
   new implementation.

Graders can only check level 2, so that is the target. It needs four things:
the same code (git), the same environment (venv + pinned versions), the same
settings (config), and the same randomness (seeds).

### 2.2 Virtual environment and pinned versions

A venv is a private copy of Python with its own `site-packages`, so this
project's packages cannot clash with other projects'. Pinning (`librosa==X.Y.Z`)
matters because libraries change behaviour between versions: a different
default in librosa's mel filterbank, or a changed default in scikit-learn,
silently changes features or models.

We pin only the packages we import directly, not the full `pip freeze`. Their
own dependencies are resolved by pip. Trade-off: a sub-dependency could change
under us, but the file stays readable and installable on other operating
systems (a full Windows freeze contains Windows-only packages).

### 2.3 Where randomness lives in an ML pipeline

| Source | Example in this project | Controlled by |
|---|---|---|
| Data splitting | random clip split B, seeds 0-4 | `split_seeds` in config |
| Model initialisation | CNN weights, XGBoost column sampling | `set_seed(cfg["seed"])`, `random_state=` |
| Data order | shuffling batches in a DataLoader | `torch.manual_seed`, generator |
| Augmentation / dropout | dropout masks in the CNN | `torch.manual_seed` |
| Resampling for statistics | bootstrap CIs | a seeded `np.random.default_rng` |
| Hash order | Python `set` iteration order of strings | `PYTHONHASHSEED` |
| Hardware nondeterminism | some GPU kernels (atomic adds) | cuDNN flags; may still differ slightly |

*Intuition.* A pseudo-random generator is a deterministic function: from a
starting state (the seed) it produces a fixed sequence. Same seed, same
sequence.

*Math (simplest example, a linear congruential generator):*

```latex
x_{n+1} = (a\,x_n + c) \bmod m, \qquad u_n = x_n / m
```

Here $x_0$ is the seed. numpy actually uses PCG64 and torch uses Philox /
Mersenne Twister, but the principle is identical: the output is a function of
the seed only.

*Worked example from our test.* `set_seed(42)` then `np.random.rand(5)` and
`torch.rand(5)`; reseed with 42 and draw again. The test asserts the two arrays
are exactly equal. It passes (see Section 6).

**Note on `PYTHONHASHSEED`.** Python reads it once at interpreter start, so
setting it inside `set_seed` affects only child processes started afterwards
(for example DataLoader workers). To fix it for the main process, set it in the
shell before running Python.

**Note on GPUs.** Even with seeds, some CUDA operations are nondeterministic,
so CNN results on GPU may differ in late decimal places. Seeds make runs
*nearly* identical; the bootstrap CI tells us whether such small differences
matter (they should not).

### 2.4 Configuration file

All constants live in `config.yaml` and code reads them through
`load_config()`. Benefits: one place to see every setting, no "magic numbers"
scattered across scripts, and a diff of `config.yaml` in git shows exactly
which setting changed between two results.

### 2.5 Makefile

A Makefile names each stage and the command that runs it, so "how do I
reproduce this?" has a one-line answer: `make all`. Each target (`data`,
`splits`, `features`, `tune`, `cnn`, `analysis`, `export`, `site`) currently
prints "not yet implemented" and will call its script as later steps add them.
`test` (evaluate on the test set) is deliberately not part of `all`; it runs
once, after the `pre-test-freeze` tag. `tests` runs pytest.

### 2.6 Git commits as evidence

Each commit records what changed, when, and why. For grading, the history
shows the work was done step by step, that decisions came before results (in
particular that the test set was untouched until the freeze tag), and lets
anyone check out the exact code behind a reported number.

### 2.7 Why graders care

The ES 335 rubric gives **5 points to reproducibility**. A result nobody else
can rerun cannot be checked, so it carries little weight.

---

## 3. Why this choice, and the alternatives we rejected

| Choice | Alternative rejected | Why |
|---|---|---|
| venv | conda | venv ships with Python; all our packages have pip wheels; one less tool for graders. |
| Python 3.11 | 3.12+ | Mature wheel support for torch, librosa and their dependencies at project start. |
| Pin our 20 direct packages | Full `pip freeze` | Readable, portable across OSes; full freeze includes Windows-only packages. |
| YAML config | Constants in code / argparse flags | One reviewable file; diffable; comments allowed. |
| Makefile | Shell script / `run_all.py` | Standard, named stages, runs a single stage easily. |
| One global seed + per-split seeds | Unseeded runs | Exact reruns; split seeds 0-4 give variance across splits deliberately. |

---

## 4. Code walkthrough

`src/utils.py`

- Lines 11-14, `load_config`: opens the file as UTF-8 (Windows defaults to a
  different encoding) and uses `yaml.safe_load`, which builds only plain
  dicts/lists/numbers and cannot execute arbitrary objects, unlike `yaml.load`.
- Lines 17-26, `set_seed`:
  - line 23 sets `PYTHONHASHSEED` for child processes (see 2.3);
  - line 24 seeds Python's `random`;
  - line 25 seeds numpy's global generator (used by older APIs and libraries);
  - line 26 `torch.manual_seed` seeds the CPU generator and every CUDA device.
- Lines 29-37, `get_logger`: attaches a handler only if the logger has none, so
  calling it twice does not print every message twice; INFO level with
  timestamps, so script logs show when each stage ran.

`tests/test_utils.py`

- `test_config_loads`: checks seed 42, sample rate 22,050, split sizes summing
  to 930, and 10 genres.
- `test_set_seed_repeats_draws`: draws from numpy and torch twice with the same
  seed and requires exact equality (`np.array_equal`, `torch.equal`), not
  approximate equality.

`pytest.ini`: `pythonpath = .` puts the repo root on `sys.path`, so plain
`pytest` can import `src`.

---

## 5. Pitfalls

1. **Committing data or secrets.** Audio bloats the repo and may break
   licensing; `kaggle.json` is an API key. `.gitignore` blocks both. Check
   `git status` before every commit.
2. **Unpinned packages.** "Works on my machine" because of a newer librosa;
   features silently change for the grader.
3. **Seeding only numpy.** torch and Python's `random` have their own
   generators; unseeded, the CNN differs on every run.
4. **Global seed used everywhere for splits.** We want 5 different random
   splits (seeds 0-4) to see variance; reusing 42 would give one split five times.
5. **Notebooks as the source of results.** Out-of-order cell execution makes
   results unreproducible. Notebooks here are for exploration; numbers come
   from scripts.
6. **Hard-coded paths.** `C:\Users\...` fails on any other machine; paths are
   relative and live in `config.yaml`.

---

## 6. Results from this step

- venv: Python 3.11.9 (`py -3.11 -m venv .venv`).
- `requirements.txt` pins: numpy 2.4.6, pandas 3.1.0, scipy 1.17.1,
  scikit-learn 1.9.1, xgboost 3.2.0, librosa 0.11.0, soundfile 0.14.0,
  torch 2.15.0, torchaudio 2.11.1, matplotlib 3.11.2, seaborn 0.13.2,
  pyyaml 6.0.3, joblib 1.6.0, tqdm 4.70.1, kaggle 2.2.5, datasets 5.1.0,
  pyarrow 26.0.0, pytest 9.2.0, nbconvert 7.17.2, ipykernel 7.4.0.
- `pip check`: "No broken requirements found."
- `pytest -q`: `2 passed in 15.96s`.
- `torch.cuda.is_available()` → `False`: pip installed the CPU build of torch,
  so the CNN will train on CPU unless a CUDA build is installed later.

---

## 7. Viva questions with model answers

**Q1. Why pin versions?**
Library defaults change. Pinning means a grader installs the same librosa,
scikit-learn and torch, so features and models are computed the same way.

**Q2. Why not pin the full `pip freeze`?**
It is long and includes OS-specific packages, so it may not install on Linux or
macOS. Pinning our direct dependencies is portable; the risk (a transitive
package changing behaviour) is small for this project.

**Q3. What randomness exists in this pipeline?**
Random splits, model initialisation, batch shuffling, dropout, XGBoost
subsampling, bootstrap resampling, hash ordering, and GPU nondeterminism.

**Q4. What if you set the seed once at the top of a script but then call a
library that creates its own generator?**
That generator is not seeded by the global seed. Hence we also pass
`random_state=` to sklearn/XGBoost objects and use explicit seeded generators.

**Q5. Does `set_seed` make GPU training exactly reproducible?**
Not fully: some CUDA kernels are nondeterministic. Results should agree closely,
and the bootstrap CI shows such differences are negligible.

**Q6. What if `config.yaml` changed between two runs?**
The git diff of `config.yaml` shows exactly which setting changed, and the
commit hash identifies the code + config behind each result.

**Q7. Why is `make test` not part of `make all`?**
Test evaluation must happen once, after all choices are frozen (git tag
`pre-test-freeze`). Running it routinely would invite tuning on test.

**Q8. Why does `get_logger` check for existing handlers?**
`logging.getLogger(name)` returns the same object every time; adding a handler
on each call would print each message multiple times.

---

## 8. Further reading

- Python docs, *venv*. https://docs.python.org/3/library/venv.html
- pip docs, *Requirements file format*. https://pip.pypa.io/en/stable/reference/requirements-file-format/
- PyTorch docs, *Reproducibility*. https://pytorch.org/docs/stable/notes/randomness.html
- NumPy docs, *Random sampling*. https://numpy.org/doc/stable/reference/random/index.html
- Pineau et al. (2021), *Improving reproducibility in machine learning research*,
  JMLR. https://jmlr.org/papers/v22/20-303.html
- GNU make manual. https://www.gnu.org/software/make/manual/
