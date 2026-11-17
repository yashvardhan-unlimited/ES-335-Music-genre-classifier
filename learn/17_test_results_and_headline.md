# Lesson 17: The test run, the headline number and what it means

The test set was scored exactly once, after the annotated git tag `pre-test-freeze`. This lesson
explains what was run, which number is the headline and why, how test compares with validation, and
what you can and cannot claim from it in a viva.

---

## 1. What we built

Nothing new in the code: this step is the *execution* of the protocol fixed in `DECISIONS.md`
(2026-11-14, "Test protocol"). The steps, in order:

| step | command | result |
|---|---|---|
| retrain the nine CNN runs still on the old stopping rule | `bash scripts/tune_all.sh` | all 19 ladder CNN runs use `min_epochs: 20`, on the GPU |
| refit the three final CNNs on train + valid | `python -m src.train_cnn --split C --seed S --refit_epochs E` (S = 0, 1, 2; E = 14, 14, 25) | `models/cnn_C_sS_refit.pt` |
| rebuild validation tables and analyses | `scripts/evaluate_part.py --part valid`, `scripts/analysis_val.py` | `results/val/headline.json`, `scores.csv` |
| freeze | `git tag -a pre-test-freeze -m "..."` | `src/splits.py` stops refusing the test part |
| score the test set **once** | `python scripts/evaluate_part.py --part test` | `results/test/headline.json`, `scores.csv`, `predictions/` |
| export the website data | `python scripts/export_site_data.py` | `docs/data/site.json`, `docs/img/` |

One run was retrained a second time: `B_s3`. Its stored GPU score (0.8146) differed from the score
rebuilt on the CPU (0.8196), because one validation clip had a top-two margin of 0.0005 and the
GPU and CPU round differently. `evaluate_part.py` refuses any run whose rebuilt score differs from
its stored one, so the run was retrained (new best epoch 26, 0.7903) instead of editing a number
or loosening the check.

---

## 2. Concepts in depth

### 2.1 Why the test set may be used only once

Every choice made on the validation part (a grid value, a stopping rule, which model to serve)
slightly fits the validation clips. The validation score of the chosen setting is therefore a bit
optimistic: you took the best of many tries (the winner's curse, lessons 12 and 13). The test part
has no such selection, so it is the only unbiased estimate. If you look at it, change the models and
look again, it becomes a second validation set. The tag is the evidence that the protocol was fixed
before any test number existed.

### 2.2 Two kinds of "final" model

The protocol has two sets of models, with different jobs:

- **Ladder models** are trained on *train only*, with the settings chosen on valid. They give the
  leakage ladder (A, B, B', D, C): the same recipe on five splits.
- **Final models** use the same settings but are refit on *train + valid*, so they have seen 639
  clips instead of 442. The refit CNN trains for the best epoch found on valid (14, 14, 25), because
  there is no validation part left to stop on.

The headline is the **final CNN, mean of its three run seeds, on split C test, with an artist
bootstrap interval**. This choice was written down before the test, so it is not picked after seeing
which number looks best.

### 2.3 The bootstrap and paired differences

Clips by one artist are not independent, so we resample whole artist clusters, not clips. For a
metric $m$ and clusters $g_1,\dots,g_K$:

```
draw K clusters with replacement  ->  a resample R_b
score_b = m(y[R_b], pred[R_b])      for b = 1 ... 2000
interval = 2.5th and 97.5th percentile of score_b
```

To compare two models $A$ and $B$ on the same clips, use the *same* resamples and take the
difference inside each resample:

```
diff_b = m(y[R_b], predA[R_b]) - m(y[R_b], predB[R_b])
```

This is a paired bootstrap. It is tighter than comparing two separate intervals, because
resample-to-resample luck cancels. If the interval of `diff` contains 0, the data do not separate
the two models.

### 2.4 Worked example from this project

The final CNN's three test scores on split C are 0.550, 0.576 and 0.530. Their mean is 0.552 and
the standard deviation over the three runs is 0.023. The artist bootstrap interval of the mean is
0.467 to 0.583. The final logistic regression scores 0.568. The paired difference CNN minus logistic
regression is -0.016 with interval -0.106 to +0.056: it contains zero, so the two are tied.

---

## 3. Why this choice, and the alternatives we rejected

- **Headline = final CNN mean over three seeds**, not the best seed. Reporting the best of three
  runs is selection on the test set. The mean is what a fresh run would give on average.
- **Refit on train + valid** for the final models: more data usually helps, and the epoch count is
  fixed by the rule, so nothing is tuned on test. Rejected: reporting only train-only models, which
  would waste 197 labelled clips.
- **Keeping the ladder on train only** so every rung of the ladder changes only the split, never
  the amount of training data (all splits have 442 training clips).
- **Retraining `B_s3` instead of editing its stored score or widening the tolerance.** The check
  exists to catch real mismatches; weakening it to pass would hide them.
- **Not trying any new model after the tag.** The MERT idea and the seed ensemble were left out of
  the protocol. If they are added now, they must be reported as done *after* seeing the test.

---

## 4. Code walkthrough

`src/splits.py`, L202-208 `is_test_unlocked`: runs `git tag -l pre-test-freeze` and is true only if
the output is exactly the tag name. If git is missing it returns False: it fails safe (locked).
`load_split` (`src/splits.py`) raises `PermissionError` for the test part unless this is true.

`scripts/evaluate_part.py`:

- L42-50 `runs`: lists every expected run (split, split seed, model, run seed). On C the CNN has
  three run seeds, all on split seed 0.
- L60-62 `boot_seed`: one bootstrap seed per clip set, the same for every model on it. That is
  what makes the differences paired.
- L70-100 `score_runs`: predicts each run, writes `predictions/*.csv`, and on `--part valid`
  raises `SystemExit` if the rebuilt score differs from the stored one by more than 1e-9.
- L118-127 `drops`: the leakage drops between neighbouring rungs, with paired intervals.
- L130-144 `paired_on_C`: CNN minus each other model on C.

---

## 5. Pitfalls

- **Tagging before the models are final.** The test run cannot be repeated. Stale refit models
  (the old CPU ones, 8/26/13 epochs) would have produced the headline. That is why the retrain came
  first.
- **Quoting the best seed.** The same CNN scores between 0.570 and 0.659 on validation depending on
  the seed.
- **Reading the interval as a verdict.** The C intervals are about 0.1 wide on each side; hip-hop
  in the C validation part is a single artist.
- **Comparing a validation number with a test number as if they were the same quantity.** Valid was
  used to choose settings; test was not.

---

## 6. Results from this step

Test set, clip macro-F1, ladder models (train only), from `results/test/headline.json`:

| split | kNN | logreg | SVM | XGBoost | CNN |
|---|---|---|---|---|---|
| A | 0.945 | 0.767 | 0.949 | 0.931 | 0.819 |
| B | 0.698 | 0.707 | 0.739 | 0.727 | 0.776 |
| B' | 0.674 | 0.705 | 0.732 | 0.722 | 0.748 |
| D | 0.505 | 0.560 | 0.569 | 0.539 | 0.654 |
| C (main) | 0.453 | 0.515 | 0.488 | 0.482 | 0.594 |

Split C, CNN train-only: 0.594, interval 0.491 to 0.663 (three runs, standard deviation 0.023).
On validation the same models scored CNN 0.620, logistic regression 0.468, SVM 0.458.

Final models on split C test (train + valid):

| model | macro-F1 |
|---|---|
| **CNN (headline)** | **0.552** (0.467 to 0.583) |
| logistic regression | 0.568 |
| SVM | 0.541 |
| XGBoost | 0.518 |
| kNN | 0.433 |

Paired differences, CNN minus the model (interval): train-only, SVM +0.107 (+0.022 to +0.197),
logistic regression +0.079 (-0.008 to +0.174). Final models: logistic regression -0.016
(-0.106 to +0.056), SVM +0.011 (-0.083 to +0.076), kNN +0.119 (+0.028 to +0.190).

Leakage drops on test (train-only models, with intervals): SVM A to B +0.210 (+0.189 to +0.249);
SVM B' to C +0.244 (+0.154 to +0.342); CNN B' to C +0.153 (+0.071 to +0.252); CNN A to B +0.043
(-0.002 to +0.091). Split C with the stricter filter (265 clips): CNN 0.579, logistic regression
0.501.

Figures: `docs/img/` (copied from `results/figures/`). Website data: `docs/data/site.json`.

What the numbers say, and what they do not:

- Leakage inflates the classical models a lot: the SVM falls from 0.949 (A) to 0.488 (C).
  The CNN falls less, from 0.819 to 0.594.
- Train-only, the CNN is clearly above SVM, XGBoost and kNN on C, but not clearly above logistic
  regression (interval contains 0).
- After refit on train + valid, the CNN did not improve (0.552 against 0.594). The runs differ by
  0.023 between seeds, so this may be noise; it is reported as it is.

---

## 7. Viva questions with model answers

1. **Why is the headline 0.552 and not 0.594?** The protocol (fixed before the test) names the final
   models, refit on train + valid, as the headline. Choosing 0.594 afterwards because it is higher
   would be selecting on the test set.
2. **Why did more training data not help the CNN?** We do not know. The refit uses the epoch count
   picked on validation, the runs differ by 0.023 between seeds, and the intervals overlap, so it
   may be noise. We report it without explaining it away.
3. **Is the CNN better than logistic regression?** Train-only it scores higher (+0.079) but the
   interval contains zero; on the final models they are tied (-0.016, interval contains zero). The
   honest statement is "no clear difference".
4. **What if you had looked at the test set before choosing the epochs?** The test would no longer
   be unbiased; its score would be optimistic and the interval too narrow in meaning. The tag
   and the lock in `load_split` exist to prevent this.
5. **What if the test score had been much lower than validation?** We would still report it. Only
   a bug found by a check (for example a stale model or a wrong file) justifies re-running, and
   that must be documented; changing the model to raise the score would be forbidden.
6. **Why resample artists and not clips?** Clips by one artist share style and recording
   conditions. Resampling clips treats them as independent, makes the interval too narrow and the
   differences look more certain than they are.
7. **Why retrain `B_s3` rather than relax the check?** The check compares the stored GPU score with
   the CPU rebuild. A one-clip flip from float rounding is real but rare; loosening the tolerance
   would also hide genuine mismatches such as a wrong checkpoint.
8. **Some genres in C have one to three artists. Does it matter?** Yes: scores for those genres
   mostly measure recognising those artists. It is stated as a limitation in the report.

---

## 8. Further reading

- Cawley & Talbot (2010), *On over-fitting in model selection and subsequent selection bias in
  performance evaluation*, JMLR: <https://jmlr.org/papers/v11/cawley10a.html>
- Efron & Tibshirani (1993), *An Introduction to the Bootstrap* (percentile intervals, paired
  differences).
- Sturm (2013), *The GTZAN dataset: its contents, its faults, their effects on evaluation, and its
  future use*: <https://arxiv.org/abs/1306.1461>
- Kereliuk, Sturm & Larsen (2015), *Deep learning and music adversaries* (the split C used here):
  <https://arxiv.org/abs/1507.04761>
