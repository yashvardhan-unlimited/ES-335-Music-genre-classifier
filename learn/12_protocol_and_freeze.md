# Lesson 12: One protocol for every split, and freezing it before test

Until now every model was tuned on split C, our main split. This step runs
**the identical tuning protocol on every split**: A (window-random, seed 0),
B (clip-random, seeds 0-4), B' (clip-random from fault-filtered clips, seeds
0-4) and C (artist-disjoint). It collects every choice the protocol made
into one file and every validation score into one table. Then it freezes
everything with a git tag, `pre-test-freeze`, before any test part is ever
opened.

This lesson is about the *roles* of the data parts and the *discipline*
around them: why each split gets the same protocol, why the test part is
used once, what "post-hoc" means, why peeking is a multiple-comparisons
problem, and how a git tag serves as evidence that the choices came first.

---

## 1. What we built

| file | what it does |
|---|---|
| `scripts/tune_all.sh` | The whole validation protocol in one script: for A s0, B s0-4, B' s0-4 and C, tune dummy, kNN, logistic regression, SVM and XGBoost (`src/train_classical.py`) and train the CNN (`src/train_cnn.py`); on A, B, B' one CNN run per split seed, on C the three config seeds; then the C CNN summary and `scripts/summarise_val.py`. |
| `Makefile` target `tune` | `make tune` runs `bash scripts/tune_all.sh`. |
| `scripts/summarise_val.py` | Reads every expected result JSON (fails if any is missing) and writes `results/chosen_params.json`, `results/val/summary.csv` and `results/figures/val_summary_splits.png`; prints the table. |
| `results/chosen_params.json` | split -> split seed -> model -> what valid chose: grid parameters; XGBoost also its number of trees; the CNN its best epoch (on C per run seed, plus the median). |
| `results/val/summary.csv` | One row per (split, split seed, model, CNN run seed): valid clip macro-F1 (soft and majority vote), clip and window accuracy, number of valid clips, chosen parameters. 86 rows. |
| `results/figures/val_summary_splits.png` | Valid clip macro-F1 per model across A -> B -> B' -> C, mean +/- sd over seeds, with the stratified dummy as floor. |
| `results/val/{A,B,Bp,C}_s*_*.json` | One result per split seed and model (written by the two training scripts). |

Commands (from the repo root, inside `.venv`):

```bash
make tune                              # = bash scripts/tune_all.sh, everything below
python -W ignore -m src.train_classical --split B --seed 3 --models dummy,knn,logreg,svm,xgb
python -W ignore -m src.train_cnn --split B --seed 3     # one CNN run, split seed 3
python scripts/summarise_val.py        # collect, table, figure (seconds)
```

{{RUNTIME}}

Nothing in this step loads a test part. `src/splits.py` would refuse: until
the tag exists, `load_split(..., "test")` raises `PermissionError`.

---

## 2. Concepts in depth

### 2.1 Three parts, three jobs

Every split divides the clips into three parts with fixed sizes (442 / 197 /
290 clips). Each part has exactly one job:

| part | job | may influence |
|---|---|---|
| train | fit the model's parameters (weights, support vectors, trees, scaler means) | the model |
| valid | **choose**: hyperparameters, number of trees, CNN epoch, window length, vote rule | every decision |
| test | **estimate** the performance of the finished, frozen pipeline, once | nothing |

The intuition: a score is only an honest estimate of performance on new
data if nothing that produced the model has seen the data it is scored on.
Train is seen by the fitting. Valid is seen by the *choosing*: once we pick
the hyperparameters with the best valid score, the valid score of that pick
is no longer an unbiased estimate (2.3). So we need a third part that
neither fitting nor choosing has touched. That part is test, and it can play
this role only as long as we have not used it to choose anything, including
"we looked at test, did not like it, and changed something".

### 2.2 Why every split gets the identical protocol

Our research question is how much accuracy survives as leakage is removed:
A (windows of one clip spread across parts) -> B (clips random, but one
artist's songs across parts) -> B' (same, minus GTZAN's faulty clips) -> C
(artist-disjoint by Sturm's labels; 13 names still cross parts in gtzan-bind's
naming, see `DECISIONS.md`). This is a **controlled experiment**: the split is
the treatment, the change in score is the effect. A controlled experiment
holds everything else fixed:

- same clips per part (442 / 197 / 290), so training size is not a cause;
- same features, models, grids (`config.yaml`), seeds, metric;
- same selection rule: **each split's own valid part**, clip macro-F1, soft vote;
- same refit (on train only) and same CNN training settings.

Two consequences matter.

**Each split chooses on its own valid part, never C's.** The question for
split A is "what would a researcher who used a window-random split have
built and reported?" That researcher tunes on a leaky valid part, and the
leak shapes the choice. Our data shows it: XGBoost early-stops at **903
trees** on A, 100 to 318 on B, 197 to 376 on B', and **54** on C. On A, every
extra tree memorises more of the training windows, and since the valid
windows are siblings of training windows, memorisation keeps paying. On C
it stops paying almost at once. Reusing C's 54 trees on A would hide part of
the effect we want to measure.

**The difference between rungs is then caused by the split.** If A used a
bigger grid, or B a different metric, a score difference between them could
come from the grid or the metric. With one protocol, it can only come from
which clips (or windows) ended up in which part.

### 2.3 Why the score of the chosen model is optimistic

Suppose the grid has $m$ configurations with true scores
$F_1, \dots, F_m$ and we observe noisy valid scores
$\hat F_k = F_k + \varepsilon_k$. We report the best one:

```latex
\hat F_{\text{chosen}} = \max_k \hat F_k ,
\qquad
\mathbb{E}\bigl[\max_k \hat F_k\bigr] \;\ge\; \max_k \mathbb{E}\bigl[\hat F_k\bigr] = \max_k F_k .
```

The maximum of noisy values is biased upwards (Jensen's inequality for the
convex function max). In the worst case all configurations are equally good,
$F_k = F$, and the noise is independent normal with sd $\sigma$:

```latex
\mathbb{E}\bigl[\max_{k \le m} \hat F_k\bigr] = F + \sigma\, \mathbb{E}\bigl[\max_{k \le m} Z_k\bigr],
\qquad Z_k \sim \mathcal{N}(0,1),
\qquad \mathbb{E}\bigl[\max_{k \le m} Z_k\bigr] \approx \sqrt{2 \ln m}\ \text{(large } m\text{)} .
```

**Worked example with our numbers.** The SVM grid has $m = 16$
configurations (4 values of C x 4 of gamma). By simulation (200,000 draws),
the expected maximum of 16 standard normals is **1.765**. How big is
$\sigma$? A rough gauge is the spread of the SVM's valid score across the five
B split seeds, sd **0.022**. That spread mixes clip sampling with the
choice itself, so it only sets the scale. With these values, picking the best
of 16 equally good settings would inflate the reported score by about
$1.765 \times 0.022 \approx 0.039$. Valid
configurations are correlated, not independent, so the true inflation is
smaller, but the order of magnitude (a few hundredths of macro-F1) is why
we never quote a valid score as the final result. Test, untouched by the
choice, removes this bias, but **only if it is used once**.

### 2.4 Peeking is a multiple-comparisons problem

Using test more than once turns it into a second validation part. Each look
is one more comparison, and the best-looking result among many looks is
optimistic for the same reason as in 2.3.

A clean way to see it: say each look at test has a 5% chance of showing a
"significant" improvement by luck alone. Over $m$ independent looks, the
chance that at least one of them does (the **family-wise error rate**) is

```latex
P(\text{at least one false win}) = 1 - (1 - 0.05)^m .
```

For $m = 1$ it is 0.05; for $m = 6$ (one look per model) 0.265; for
$m = 16$ (one per SVM configuration) 0.56; for $m = 25$, 0.72. Peeking
does not have to be deliberate. "Try the 5 s windows on test, just to see",
"the CNN looked bad on test, so we trained one more seed", "drop the genre
that does badly": each is a look that can change what we report, so the
final number is no longer a one-shot estimate. Gelman and Loken call
the many unplanned choices an analyst *could* have made the **garden of
forking paths**. The problem exists even when each individual step looks
innocent.

Our defence is structural, not willpower: all choices are made on valid, they
are written into files and committed, the commit is tagged, and only then
does the code even allow test to be loaded.

### 2.5 "Post-hoc", and how to report it honestly

An analysis is **post-hoc** if it was decided *after* seeing the results it
is about. Post-hoc analyses are not forbidden, and they are often where the
interesting findings are. They are a different kind of evidence: they suggest
hypotheses, they do not confirm them. The honest pattern:

1. **Report the pre-specified result first, unchanged.** Here: the frozen
   pipeline's test macro-F1 with its artist-bootstrap CI, for every model
   and split, exactly as `chosen_params.json` and the tagged code define them.
2. **Label everything else post-hoc**, in the same words in the report,
   the website and the viva: "post-hoc: after seeing test, we also ...".
3. **Never let a post-hoc result replace the headline.** If a post-hoc
   variant scores better on test, that is a hypothesis for a *new* test set,
   not a new result.
4. **Say how many things you tried**, so a reader can judge the
   multiple-comparisons risk.

Example from lesson 11: 5 s windows scored 0.477 vs 0.458 on valid. Even
on valid, choosing 5 s after seeing that would have been selection on noise.
We kept 3 s (DECISIONS.md). Had we instead found it on *test*, the only
honest report would be "post-hoc, 5 s windows scored X on test; not part of
the frozen protocol".

### 2.6 The git tag as evidence

A git **commit** is identified by a hash computed from its content: the
whole file tree (every file's bytes), the parent commit(s), author, date
and message. Change one byte of `results/chosen_params.json` and the hash
changes. An **annotated tag** is a separate git object that names one
commit hash and records who created it (tagger), when, and why (message):

```bash
git tag -a pre-test-freeze -m "all choices frozen before test"
git show pre-test-freeze      # tagger, date, message, then the commit it points to
```

What this gives us:

- **A fixed reference.** The tagged commit contains the code, `config.yaml`,
  `DECISIONS.md`, `chosen_params.json` and the validation results. Anyone can
  run `git diff pre-test-freeze HEAD` to see exactly what changed after the
  freeze. If the test step changes nothing but adds `results/test/`, the diff
  shows that.
- **An enforced order.** `src/splits.py` checks for the tag before any test
  load (`is_test_unlocked`, L104-107; `load_split`, L114-119). The test
  script will also write the commit hash it ran on into
  `results/test/RUN.md`, so the run is tied to a specific frozen state.
- **A timestamp others can check**, once pushed: GitHub shows when the tag
  arrived, and that it came before the test results.

Its limits, stated honestly: a local tag can be deleted and recreated, and
a local date can be faked. The tag is evidence of an *ordering* we commit
to, strongest once it is pushed before the test results exist. It is not
cryptographic proof against a determined cheat. The real protection is the
combination: the lock in code, the tag, the commit hash in `RUN.md`, and
`DECISIONS.md` written as we went.

### 2.7 Reading the validation table

{{TABLE}}

---

## 3. Why this choice, and the alternatives we rejected

| choice | alternative rejected | reason |
|---|---|---|
| Each split tunes on its own valid part | Reuse C's chosen hyperparameters everywhere | A leaky split also makes leaky *choices* (903 vs 54 XGBoost trees); reusing C's would understate what a leaky protocol really reports. |
| Same grids, metric, refit, sizes on all splits | Tune each split "as well as possible" | Then score differences could come from the protocol, not the split. |
| CNN once per split seed on A/B/B', three run seeds on C | Three CNN seeds everywhere | 11 extra runs x 2 at 14 to 35 min each, about 5 to 13 h more on this CPU; the 5 split seeds of B and B' already include seed-to-seed variation (DECISIONS.md). |
| Mean +/- sd over the 5 split seeds of B and B' | One seed per split | One clip-random draw can be lucky; the sd shows how much. A and C have one fixed split each. |
| Freeze with an annotated tag before test | A note in the README; a plain commit | An annotated tag records tagger, date and message and pins one commit hash; the code can check for it. |
| Test once, everything else post-hoc and labelled | Look at test to "sanity check" choices | Every look is a comparison (2.4); a checked-then-changed pipeline has no unbiased estimate left. |
| `summarise_val.py` fails if any result is missing | Summarise what exists | A silently partial table would look complete. |

---

## 4. Code walkthrough

**`scripts/tune_all.sh`**

- **L8 `set -euo pipefail`**: stop at the first failing command, unset
  variable or failing pipe stage. A failed run must not be followed by a
  summary that looks complete.
- **L9 `cd "$(dirname "$0")/.."`**: run from the repo root wherever it is
  called from, because all paths in `config.yaml` are relative.
- **L10**: the venv Python on Windows (`Scripts/python.exe`) or Linux/macOS
  (`bin/python`).
- **L12-16**: the loop over (split, seed) pairs. L14 tunes all classical
  models with `src/train_classical.py`; L15 trains one CNN run per split seed
  for A, B, B' (the run seed is the split seed, so the torch seed is
  42 + split seed).
- **L17-19**: the C CNN runs, with seeds read from `config.yaml`
  (`cnn.seeds`), not hard-coded.
- **L20-21**: the C CNN summary (median best epoch) and the overall
  summary.

**`src/train_classical.py`** (unchanged in this step; the protocol it
implements is the same for every `--split`):

- **L73-103 `tune_and_score`**: L76-78 put train and valid side by side
  with a `PredefinedSplit` so the grid search has exactly one fold, train ->
  valid; L79-81 score every configuration with `ClipMacroF1Scorer` (clip
  macro-F1, soft vote); L83 takes the best (first of ties); L87-88 refit on
  **train only** with one BLAS thread; L89 scores valid with that refit
  model. The grid's own score is kept as `grid_valid_macro_f1` (L90-95: on
  A and B, exact-duplicate files can make kNN ties break differently; see
  DECISIONS.md).
- **L285 `if args.split == "C"`**: tuning figures only for the main split,
  so running A/B/B' does not overwrite C's lesson figures.

**`scripts/summarise_val.py`**

- **L23 `SPLIT_SEEDS`**: which split seeds belong to the protocol; B and B'
  read `split_seeds` from config.
- **L29-37 `runs`**: the list of every *expected* result file, including
  the CNN run seeds (config seeds on C, the split seed elsewhere).
- **L40-46 `model_params`**: what valid chose for each model; XGBoost's
  tree count and the CNN's best epoch are choices too, because valid decided
  them by early stopping.
- **L51-53**: refuse to summarise if any expected file is missing.
- **L56-69**: one row per result for `summary.csv`, and the nested
  `chosen_params.json`; on C the CNN gets its best epoch per run seed and the
  median (the epoch count a refit on train+valid would use).
- **L76-81**: mean +/- sd per model and split, printed as a table.
- **L85-116 `plot`**: one line per model across the four splits (offset
  sideways so error bars do not overlap), the stratified dummy as a dashed
  floor, values printed at the C end.

**`src/splits.py`, the lock**

- **L104-110 `is_test_unlocked`**: runs `git tag -l pre-test-freeze` and
  returns True only if git prints exactly that name. With no git installed
  it returns False, so the lock stays shut.
- **L117-124 `load_split`**: a test part is returned only if the caller
  passes `allow_test=True` **and** the tag exists; otherwise
  `PermissionError("test set locked until pre-test-freeze tag")`. Every
  loader in the project goes through this function.

---

## 5. Pitfalls

1. **Tuning A or B with C's hyperparameters.** Understates the leaky
   splits, because a leaky valid part also leads to leaky *choices* (more
   trees, less regularisation: SVM C = 100 on A).
2. **Different protocols per split** (bigger grid on C "because it is the
   main result"). The A -> C gap would then mix leakage with tuning effort.
3. **Choosing on valid with a different metric** (window accuracy, as in
   lesson 09's third mistake). The chosen settings would differ from what
   our metric would choose.
4. **Quoting valid scores as results.** They are the maximum over a grid
   (2.3) and therefore optimistic. They are for choosing; test is for
   reporting.
5. **"Just a quick look" at test.** It cannot be undone. After one look,
   any change is post-hoc.
6. **Tagging a dirty working tree.** A tag points to a commit, not to
   uncommitted files. Results or code that were not committed are not
   frozen. Check `git status` is clean first.
7. **Moving the tag** (`git tag -f`) after seeing test. It destroys the
   evidence the tag exists for. If something must change after the freeze,
   keep the tag, commit the change, label it post-hoc, and report both.
8. **Reading B vs B' differences of 0.01 to 0.02 as real.** The sd over
   split seeds is about 0.01 to 0.03 (2.7). Smaller differences are noise.

---

## 6. Results from this step

{{RESULTS}}

---

## 7. Viva questions with model answers

**Q1. Why not tune once on C and apply the same hyperparameters to all
splits? It would be faster.**
Because the leaky splits would then not show what a leaky protocol really
reports. Leakage also changes the *choices*: XGBoost keeps 903 trees on A
and 54 on C, and the SVM picks C = 100 on A. A researcher who used split A
would tune on A's valid part and get A's choices. To measure the leakage of
the whole protocol, each split must run the whole protocol.

**Q2. What are the roles of valid and test, in one sentence each?**
Valid chooses: every hyperparameter, epoch and design option is picked by
its valid clip macro-F1. Test estimates: the frozen pipeline is scored on
it once, and that number is the result.

**Q3. Why may test be used only once?**
Each use is a comparison. If we look several times and keep the best or
change something after looking, the reported test score is a maximum over
looks and is optimistic, exactly like a valid score after tuning. With 16
independent looks, the chance of at least one lucky "win" at the 5% level
is 0.56.

**Q4. What does "post-hoc" mean, and what do you do with a post-hoc result?**
Decided after seeing the results it is about. We report the pre-specified
result first and unchanged, label the post-hoc analysis as such, say how
many things we tried, and never promote it to the headline. A post-hoc
finding is a hypothesis for new data, not a confirmed result.

**Q5. How does a git tag prove you did not peek?**
It does not prove it on its own. It pins one commit hash, and that hash
covers every committed file: code, config, `chosen_params.json`, valid
results. It also records tagger, date and message. The code refuses to load
test without the tag, and the test run writes its commit hash into
`RUN.md`. Pushed to GitHub before the test results, it gives a public
ordering. A determined cheat could fake a local tag; the evidence is the
combination of tag, lock, hash and the decision log.

**Q6. Your valid table shows SVM 0.937 on A and 0.458 on C. Is the drop
of 0.479 "the leakage"?**
It is the drop in *valid* scores under one identical protocol, so the split
is the only thing that differs. Strictly, both numbers are optimistic maxima over the grid (2.3),
and A is one seed. The test step will measure the same contrast on parts
nothing was chosen on, with artist-bootstrap CIs.

**Q7. What if you had used three CNN seeds per split seed?**
The CNN points on A, B, B' would have smaller noise, at about 5 to 13 extra
hours on this CPU. The B and B' sds over five split seeds already include
CNN seed variation. Lesson 10 showed one seed alone can move valid F1 by up
to 0.055, so we treat single CNN differences below that as noise.

**Q8. Logistic regression drops much less from A to B than kNN or the
SVM. Why?**
A linear model cannot memorise individual training windows. kNN (3
neighbours, distance-weighted, on A) can find a valid window's own siblings
from the same clip, and an SVM with C = 100 or a 903-tree XGBoost can carve
out tiny regions around them. Window-level leakage rewards the capacity to
memorise, so the most flexible models gain most from it.

**Q9. What if, after the test step, you find a bug in feature extraction?**
Fix it, commit, keep the tag, and report the frozen result *and* the
corrected one, clearly labelled as post-freeze and post-hoc. Do not move the
tag. `git diff pre-test-freeze HEAD` then shows exactly what changed.

---

## 8. Further reading

- G. C. Cawley, N. L. C. Talbot (2010). *On over-fitting in model selection
  and subsequent selection bias in performance evaluation*. JMLR 11.
  https://jmlr.org/papers/v11/cawley10a.html
- A. Gelman, E. Loken (2013). *The garden of forking paths*.
  http://www.stat.columbia.edu/~gelman/research/unpublished/p_hacking.pdf
- C. Dwork et al. (2015). *The reusable holdout: preserving validity in
  adaptive data analysis*. Science 349. https://doi.org/10.1126/science.aaa9375
- B. A. Nosek et al. (2018). *The preregistration revolution*. PNAS 115.
  https://doi.org/10.1073/pnas.1708274114
- B. Recht et al. (2019). *Do ImageNet classifiers generalize to ImageNet?*
  ICML. https://arxiv.org/abs/1902.10811
- B. L. Sturm (2013). *The GTZAN dataset: its contents, its faults, their
  effects on evaluation, and the future use of GTZAN*.
  https://arxiv.org/abs/1306.1461
- Git book, *Tagging*: https://git-scm.com/book/en/v2/Git-Basics-Tagging;
  *Git objects*: https://git-scm.com/book/en/v2/Git-Internals-Git-Objects
