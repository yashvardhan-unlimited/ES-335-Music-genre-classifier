# Lesson 06: Evaluation metrics and the bootstrap

We fix **how** models will be judged before training any of them. This lesson
covers what we measure (clip-level macro-F1), how window predictions become a
clip prediction, and how sure we can be of a number computed from a few hundred
clips by a limited number of artists.

---

## 1. What we built

| Function / class in `src/evaluate.py` | Purpose |
|---|---|
| `macro_f1(y_true, y_pred)` | Convenience metric for the bootstrap. |
| `aggregate_clips(window_probs, clip_ids, method)` | 10 window probability vectors → one clip prediction (`soft` or `majority`). |
| `clip_scores(y_true, y_pred, labels)` | macro-F1, accuracy, per-genre F1, confusion matrix (counts and row-normalised). |
| `window_accuracy(y_true, y_pred)` | Window-level accuracy, diagnostic only. |
| `make_predefined_split(parts)` | One fixed train/valid fold for `GridSearchCV`. |
| `ClipMacroF1Scorer` | `scoring=` object for `GridSearchCV`: soft vote by clip, clip macro-F1. |
| `cluster_bootstrap_ci(df, metric_fn, ...)` | Point estimate and 95% CI, resampling artists. |
| `paired_bootstrap_diff(df, a, b, metric_fn, ...)` | Difference between two models with a CI, same resamples for both. |

`tests/test_evaluate.py`: 14 tests on tiny hand-made cases (Section 4).

```bash
.venv/Scripts/python -m pytest -q tests/test_evaluate.py
```

---

## 2. Concepts in depth

### 2.1 Why the evaluation is designed before any model is compared

If you first train ten models and then decide how to score them, every choice
(metric, aggregation, which split) can drift towards whatever makes your
favourite look best: a "garden of forking paths". Fixing the evaluation first
makes the comparison fair and the claims checkable. It is also the single
largest item in the ES 335 rubric: **evaluation design, 20 points**.
Everything here is recorded in DECISIONS.md with today's date, before any
model exists.

### 2.2 Confusion matrix, precision, recall, F1

For $K$ classes the **confusion matrix** $C$ is $K \times K$: $C_{ij}$ = number
of clips of true class $i$ predicted as class $j$. The diagonal holds correct
predictions. For class $g$:

```latex
TP_g = C_{gg}, \qquad FP_g = \sum_{i \ne g} C_{ig}, \qquad FN_g = \sum_{j \ne g} C_{gj}
```

```latex
P_g = \frac{TP_g}{TP_g + FP_g} \quad\text{(of clips called } g\text{, how many are } g\text{)}
\qquad
R_g = \frac{TP_g}{TP_g + FN_g} \quad\text{(of real } g\text{ clips, how many we found)}
```

```latex
F1_g = \frac{2 P_g R_g}{P_g + R_g} = \frac{2\,TP_g}{2\,TP_g + FP_g + FN_g}
```

F1 is the harmonic mean: it is high only if **both** precision and recall are
high. A genre the model never predicts has $R_g = 0$ and $F1_g = 0$.

The **row-normalised** confusion matrix divides each row by its total, so row
$g$ shows where the clips of genre $g$ went; the diagonal is each genre's recall.

### 2.3 Worked example (the test `test_three_class_macro_f1_by_hand`)

True: `[0, 0, 0, 1, 1, 2]`, predicted: `[0, 0, 1, 1, 2, 2]`.

```
              pred 0  pred 1  pred 2
true 0           2       1       0
true 1           0       1       1
true 2           0       0       1
```

| class | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|
| 0 | 2 | 0 | 1 | 1.00 | 0.667 | 0.800 |
| 1 | 1 | 1 | 1 | 0.50 | 0.500 | 0.500 |
| 2 | 1 | 1 | 0 | 0.50 | 1.000 | 0.667 |

Accuracy $= 4/6 = 0.667$.

### 2.4 Macro vs micro vs weighted

With $n_g$ = number of true clips of class $g$ and $N = \sum n_g$:

```latex
\text{macro-F1} = \frac{1}{K}\sum_g F1_g
\qquad
\text{weighted-F1} = \sum_g \frac{n_g}{N} F1_g
\qquad
\text{micro-F1} = \frac{2\sum_g TP_g}{2\sum_g TP_g + \sum_g FP_g + \sum_g FN_g}
```

For single-label classification every error is one FP (for the predicted
class) and one FN (for the true class), so micro-F1 equals accuracy.

On the worked example:
- macro-F1 $= (0.800 + 0.500 + 0.667)/3 = \mathbf{0.656}$
- weighted-F1 $= (3 \cdot 0.800 + 2 \cdot 0.500 + 1 \cdot 0.667)/6 = \mathbf{0.678}$
- micro-F1 $= 2 \cdot 4 / (2 \cdot 4 + 2 + 2) = \mathbf{0.667}$ = accuracy.

Weighted and micro lean towards the big class 0 (which does well); macro gives
the small class 1 (which does badly) a full third of the vote.

### 2.5 Why macro-F1 is primary even with balanced classes

GTZAN is balanced overall, but our test parts are not exactly: C's test part
has 26 (reggae) to 32 (rock) clips per genre (lesson 04). More importantly:
1. Macro-F1 punishes a model that gives up on one genre; accuracy barely
   notices if the other nine are fine.
2. It combines precision and recall: a model that dumps uncertain clips into
   "rock" gets low precision for rock even if rock recall is high.
3. It is the standard reported on this split, so we stay comparable.
On balanced data macro-F1 and accuracy are usually close; when they diverge,
that is a signal worth investigating (some genre is absorbing errors).

### 2.6 Window level vs clip level

Models are trained on 3 s windows, but users and labels deal with clips. We
always score clips:
- Window accuracy counts a clip 10 times, so clips aren't weighted like
  independent examples, and it measures a task nobody performs (genre of a
  random 3 s snippet).
- Window and clip numbers can differ a lot: in
  `test_clip_scorer_aggregates_through_the_index` window accuracy is 2/4 =
  0.5, yet both clips are classified correctly after averaging (clip macro-F1
  = 1.0).

`window_accuracy` exists only as a diagnostic ("how much does aggregation
help?").

### 2.7 Soft vote vs majority vote

Each window gives a probability vector $p(\cdot \mid w_i)$.

- **Soft vote**: average the vectors, then argmax.
  $\hat{y} = \arg\max_g \frac{1}{10}\sum_i p(g \mid w_i)$.
- **Majority vote**: each window votes for its own argmax; most votes wins;
  ties broken by the mean probability.

They can disagree. `test_soft_and_majority_can_disagree`: 2 windows say
class 0 with 0.99, 3 windows say class 1 with 0.6. Majority: class 1 (3 votes
to 2). Soft: mean $p_0 = (2 \cdot 0.99 + 3 \cdot 0.4)/5 = 0.636$, so class 0.
Soft voting lets confident windows outweigh hesitant ones; majority throws
confidence away. Ties are real with 10 windows (5-5, 4-4-2): in
`test_majority_tie_broken_by_mean_probability` votes are 2-2 and the mean
probability (0.45 vs 0.40) picks class 1.

We use soft voting (DECISIONS.md) because it uses more information and has no
ties in practice; majority is computed for comparison.

### 2.8 Sampling distributions and the bootstrap from first principles

Our test macro-F1 is computed from one particular set of 290 clips. A
different (equally valid) set of clips would give a different number. The
**sampling distribution** of the statistic is the distribution of values we
would get over many such sets. Its spread is the uncertainty of our one
number.

We cannot collect many test sets. The **bootstrap** (Efron 1979) pretends the
sample is the population: draw a new sample of the same size **with
replacement** from the data we have, recompute the statistic, repeat $B$
times. The spread of the $B$ values approximates the sampling distribution.

For $B$ resamples with values $\hat\theta^{*}_1,\dots,\hat\theta^{*}_B$, the
95% **percentile interval** is

```latex
\left[\ \hat\theta^{*}_{(2.5\%)},\ \hat\theta^{*}_{(97.5\%)}\ \right]
```

We use $B = 2000$ (`config.yaml` `bootstrap.n`) so the 2.5th and 97.5th
percentiles are estimated from 50 resamples in each tail.

*Why "with replacement"*: each resample must be a plausible alternative dataset
of the same size; drawing without replacement would just reshuffle the same
clips and give the same score every time.

### 2.9 Why resample artists: clustered, non-independent data

The bootstrap assumes the things you resample are independent. Clips are not:
clips by the same artist are similar, and a model that fails on an artist
tends to fail on all their clips. Resampling clips treats 18 Public Enemy
clips as 18 independent pieces of evidence; really they are closer to one.

So we resample **artists (clusters)**: draw as many artists as there are, with
replacement, and take all clips of each drawn artist. A clip whose artist is
unknown is its own cluster (we do not know who it shares an artist with, and
lumping all unknowns together would glue unrelated clips).

*Real structure, C's valid part* (computed with `_cluster_rows`): 197 clips
form **79 clusters**; the biggest are Public Enemy (18 clips), John Lee Hooker
(11) and Magic Slim & The Teardrops (9); 54 clusters are single clips; 1 clip
has no artist name.

*Synthetic illustration on that structure* (not a model result): we made each
artist entirely right with probability 0.7 or entirely wrong, then bootstrapped
accuracy both ways, 2,000 resamples:

| Resampling | point | 95% CI | width |
|---|---|---|---|
| clips | 0.655 | [0.589, 0.721] | 0.132 |
| artists | 0.655 | [0.471, 0.825] | **0.354** |

Same data, same point estimate; the honest (artist) interval is **2.7×
wider**. When errors cluster by artist, a clip bootstrap overstates our
certainty. With real models errors cluster partly, so the true gap will be
somewhere in between, but the artist bootstrap is the safe choice.

The *effective* sample size is closer to the number of clusters than to the
number of clips. One rough formula (the design effect) for equal cluster size
$m$ and within-cluster correlation $\rho$:

```latex
n_{\text{eff}} = \frac{n}{1 + (m - 1)\rho}
```

With $\rho = 1$ (artists are all-right or all-wrong), $n_{\text{eff}}$ equals the
number of clusters.

### 2.10 Paired comparisons and reading a CI on a difference

To compare model A and model B, compute both on **the same** resampled
clusters in each bootstrap round and record the difference
$d^{*}_b = M_A^{*} - M_B^{*}$. Some test sets are harder than others for every
model; pairing cancels that shared difficulty, so the difference is measured
much more precisely than either score alone.

Reading the output `(diff, low, high, share_pos)`:
- **CI excludes 0** (e.g. [0.02, 0.09]): A is better with this test set and
  this uncertainty model.
- **CI includes 0** (e.g. [−0.03, 0.05]): no evidence of a difference; *not*
  evidence that they are equal.
- **share_pos**: fraction of resamples where A beat B. 0.97 means A won on
  97% of plausible test sets. It is not a p-value, but it is easy to read.

Do **not** compare models by checking whether their separate CIs overlap: two
overlapping CIs can still hide a clearly positive paired difference, because
the separate CIs both contain the shared test-set noise.

`test_paired_bootstrap` checks the extremes: a model that is always right
vs one always wrong gives (1, 1, 1, 1); a model vs itself gives (0, 0, 0, 0).

### 2.11 What these tools cannot tell us

- **Not generalisation beyond GTZAN.** The bootstrap describes uncertainty
  from *which GTZAN artists/clips* were sampled, not how the model would do on
  other datasets, eras or label definitions.
- **Not label noise.** Mislabelled test clips bias every model's score the
  same way; no resampling fixes that.
- **Not leftover leakage.** C still has 7.6% of test clips sharing an artist
  name with train (lesson 04); the CI does not correct for it.
- **Not the effect of tuning many models.** A CI on the winner of 20 models is
  optimistic; the test lock and tuning only on valid limit this.
- **Not "why".** A higher macro-F1 says nothing about what the model listens
  to (loudness, artist, actual musical cues).
- **Small-sample limits.** With ~79-100 clusters per part, the percentile
  bootstrap is approximate, especially near 0 or 1.

---

## 3. Why this choice, and the alternatives we rejected

| Choice | Alternative rejected | Why |
|---|---|---|
| Clip-level macro-F1 primary | Accuracy, window accuracy | Equal weight per genre; scores the real task. |
| Soft vote | Majority vote | Uses confidence; no ties. Majority kept as comparison. |
| `PredefinedSplit` on valid | k-fold over windows | k-fold over windows recreates split A inside tuning. |
| Clip-aggregating scorer in `GridSearchCV` | Default window accuracy scorer | Tunes for the metric we report. |
| Artist-cluster bootstrap | Clip bootstrap | Clips by an artist are correlated; clip CIs too narrow (2.7× in our synthetic check). |
| Missing artist = own cluster | One "unknown" cluster | Unknown ≠ same artist. |
| Percentile interval, B = 2000 | Normal approximation | No normality assumption for a bounded statistic. |
| Paired bootstrap | Overlap of separate CIs | Cancels shared test-set difficulty. |

---

## 4. Code walkthrough (`src/evaluate.py`)

- Lines 21-45, `aggregate_clips`:
  - `np.unique(..., return_inverse=True)` gives the sorted clip ids and, for
    each window, the index of its clip (`inv`).
  - Line 35: `np.add.at(mean_probs, inv, window_probs)` sums window rows into
    their clip row (`add.at` handles repeated indices correctly, unlike
    `mean_probs[inv] += ...`). Dividing by window counts gives the mean.
  - Line 42: votes are counted the same way at `(clip, argmax class)`.
  - Line 44: classes that are not tied for most votes get `-inf`, so the
    argmax over mean probabilities chooses only among the tied classes; with no
    tie it picks the single winner.
- Lines 48-66, `clip_scores`: `f1_score(..., average=None, labels=labels)` gives
  per-class F1 in a fixed order; macro-F1 is their mean, so a class with no
  true and no predicted clips counts as 0 when it is in `labels` (test
  `test_absent_class_counts_as_zero_and_empty_row_is_zero`). Row normalisation
  uses `np.divide(..., where=rows > 0)` so an empty row gives zeros, not NaN.
- Lines 74-80, `make_predefined_split`: maps `train → -1` (never a validation
  row) and `valid → 0` (fold 0); anything else, notably `test`, raises.
- Lines 83-102, `ClipMacroF1Scorer.__call__(estimator, X, y)`: the signature
  `GridSearchCV` expects. sklearn slices the DataFrame `X` by rows but keeps its
  index, so the clip id of every validation row is recovered through
  `X.index` (or a `clip_ids` Series indexed like `X`). It soft-votes, checks
  that every clip's windows share one label, maps predicted column indices back
  to `estimator.classes_` and returns macro-F1 over all classes.
- Lines 105-110, `_cluster_rows`: missing artist → a unique key `__solo_<row>`.
- Lines 113-117, `_resamples`: each round draws `len(groups)` cluster indices
  with replacement and concatenates their rows; seeded `default_rng` makes
  intervals reproducible.
- Lines 120-131, `cluster_bootstrap_ci`; lines 134-146, `paired_bootstrap_diff`:
  the paired version evaluates both prediction columns on the **same** `r` in
  each round.

`tests/test_evaluate.py` highlights:
- 3-class macro-F1 = 0.6556, accuracy 4/6, exact confusion matrices.
- Soft vs majority disagreement; a 2-2 majority tie resolved by mean probability.
- The scorer gives the same answer when rows are shuffled (alignment by index,
  not position) and runs inside a real `GridSearchCV`.
- Two artists, one all-right, one all-wrong: point 0.5, interval [0, 1].

---

## 5. Pitfalls

1. **Scoring windows in `GridSearchCV`** (the default scorer): tunes for the
   wrong task and, with k-fold, for leakage.
2. **Losing the clip alignment**: if X were a NumPy array, sklearn's slices
   would not carry clip ids; the scorer needs a DataFrame index.
3. **Macro-F1 over only the labels present in a small sample**: a genre missing
   from a bootstrap resample changes the denominator. `clip_scores` takes a fixed
   `labels` list for reported numbers.
4. **Clip bootstrap on clustered data**: intervals too narrow.
5. **"CIs overlap, so no difference"**: use the paired difference.
6. **Reading a CI as covering everything**: it covers sampling of clips/artists
   only (2.11).

---

## 6. Results from this step

No model results yet (by design). From code run in this step:
- `pytest -q`: **45 passed** in total, **14** in `tests/test_evaluate.py`.
- C valid part: 197 clips in **79 artist clusters** (largest 18, 54 singletons,
  1 missing artist).
- Synthetic check on that structure: clip-bootstrap CI width 0.132 vs artist
  0.354 (**2.7×**).

---

## 7. Viva questions with model answers

**Q1. Why design the evaluation before training models?**
So that the metric, aggregation and uncertainty method cannot be chosen to
flatter a particular model. It also makes the comparison reproducible and is
graded explicitly (20 points).

**Q2. Classes are balanced. Why not just accuracy?**
Test parts are not exactly balanced (C test 26-32 per genre), and accuracy can
hide one failing genre. Macro-F1 gives every genre equal weight and combines
precision and recall. We report accuracy as a secondary number.

**Q3. Compute macro-, weighted- and micro-F1 for true [0,0,0,1,1,2] and
predicted [0,0,1,1,2,2].**
Per-class F1 0.8, 0.5, 0.667. Macro 0.656, weighted 0.678, micro = accuracy
0.667.

**Q4. When do soft and majority voting disagree?**
When a few confident windows face more, less confident ones; e.g. two windows
at 0.99 for class 0 and three at 0.6 for class 1: majority picks 1, soft picks 0.

**Q5. Why resample artists rather than clips?**
Clips by one artist are not independent; errors cluster by artist. Resampling
clips overstates the number of independent observations and gives intervals
that are too narrow (2.7× too narrow in our synthetic check).

**Q6. What if a clip has no artist?**
It is its own cluster. Grouping all unknown clips together would assume they
share an artist, which we have no reason to believe.

**Q7. Model A has 0.62 [0.55, 0.69], model B 0.58 [0.51, 0.65]. Is A better?**
Can't tell from overlapping separate CIs. Compute the paired bootstrap
difference on the same resamples: if its CI excludes 0 (say [0.01, 0.07]), A is
better on this test distribution.

**Q8. What does share_pos = 0.9 mean?**
In 90% of resampled test sets A scored higher than B. It is a descriptive
summary of the bootstrap distribution, not a p-value.

**Q9. Why a `PredefinedSplit` and not 5-fold CV for tuning?**
Our splits are designed to control leakage; k-fold over windows would mix
windows of one clip and artists across folds. One fixed train/valid split
matches the evaluation exactly.

**Q10. What can a CI not tell you?**
Generalisation to other datasets, the effect of label noise, leftover leakage,
the optimism from picking the best of many models, or why the model works.

---

## 8. Further reading

- Efron & Tibshirani (1993), *An Introduction to the Bootstrap*, Chapman & Hall.
- Field & Welsh (2007), *Bootstrapping clustered data*, J. R. Stat. Soc. B
  69(3). https://doi.org/10.1111/j.1467-9868.2007.00593.x
- Sokolova & Lapalme (2009), *A systematic analysis of performance measures
  for classification tasks*, Information Processing & Management 45(4).
  https://doi.org/10.1016/j.ipm.2009.03.002
- Berg-Kirkpatrick, Burkett & Klein (2012), *An empirical investigation of
  statistical significance in NLP* (paired bootstrap), EMNLP.
  https://aclanthology.org/D12-1091/
- scikit-learn: *Model evaluation* (precision/recall/F1, averaging) and
  `PredefinedSplit`. https://scikit-learn.org/stable/modules/model_evaluation.html
- Gelman & Loken (2013), *The garden of forking paths*.
  http://www.stat.columbia.edu/~gelman/research/unpublished/p_hacking.pdf
