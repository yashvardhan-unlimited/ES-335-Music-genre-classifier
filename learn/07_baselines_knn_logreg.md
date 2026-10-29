# Lesson 07: Floors and baselines: dummy classifiers, kNN, logistic regression

Before any "real" model, we set out what a number like 0.47 means. A **floor**
says what a model with no information scores under our metric. A **simple
baseline** says what an ordinary, well-understood model scores. Every later
model (SVM, XGBoost, CNN) is judged against both. In the ES 335 rubric the
baseline is worth **15 points**, and it is what makes any later claim of
improvement checkable.

All numbers here come from the **valid** part of split C (442 train clips =
4,420 windows; 197 valid clips = 1,970 windows). The test part was never
loaded.

---

## 1. What we built

| File | What it does |
|---|---|
| `src/models.py` | Factories returning sklearn objects: `dummy_most_frequent`, `dummy_stratified`, `knn_pipeline`, `logreg_pipeline`, `svm_pipeline`, `xgb_model`. kNN, logreg and SVM are `Pipeline([StandardScaler, clf])`. `MODELS` maps CLI names to (factory, grid key). |
| `src/train_classical.py` | CLI. Loads train and valid window features through the locked `load_split`, tunes each model with `GridSearchCV(cv=PredefinedSplit, scoring=ClipMacroF1Scorer)`, refits the best setting on train, scores valid per clip, writes one JSON per model, plots kNN valid F1 against k, prints a summary table. |
| `config.yaml` (`grids`, `logreg_max_iter`) | The hyperparameter grids (all constants live in config). |
| `tests/test_models.py` | Scaler sits inside each pipeline and is fit on the training data only; every model returns probabilities; feature rows line up with `features/index.csv`; the loader refuses the test part. |
| `results/val/C_s0_{model}.json` | Best params, valid clip macro-F1 (soft and majority), accuracy, window accuracy, per-genre F1, confusion counts, the full grid. |
| `results/figures/val_knn_k.png` | Valid clip macro-F1 against k, both weightings, with the stratified-dummy floor. |

```bash
.venv/Scripts/python -W ignore -m src.train_classical --split C --seed 0 --models dummy,knn,logreg
# or
make tune
# tests
.venv/Scripts/python -m pytest -q tests/test_models.py
```

`--models dummy` expands to both dummy classifiers. `--seed` picks the split
file for B and B' (`B_seed{seed}.csv`); split C has one file, so for C it only
labels the output. The whole run takes about 30 s on a laptop.

---

## 2. Concepts in depth

### 2.1 Why a floor and a baseline are both required

A score means nothing on its own. "Macro-F1 0.47" could be excellent (if
chance is 0.02) or poor (if a one-line rule gets 0.60). Two reference points
fix that:

- **Floor (dummy classifier)**: uses no features at all. Anything that cannot
  clearly beat it has learned nothing. It also shows how our metric behaves at
  the bottom: with 10 genres, macro-F1 for a constant predictor is far below
  its accuracy (Section 2.2).
- **Simple baseline (kNN, logistic regression)**: standard, cheap models
  with few hyperparameters, fully understood. If an SVM or a CNN only matches
  logistic regression, the extra complexity has bought nothing. This is the
  "is the fancy model worth it?" test.

There is also a debugging reason. If the pipeline had a bug (labels shuffled,
features misaligned with labels, test leaking into train), it would show up
as a baseline that is impossibly good or exactly at the floor. Our
`load_windows` test (Section 4) exists for the same reason.

### 2.2 Dummy classifiers

**Intuition.** `DummyClassifier` ignores X completely.

- `most_frequent`: always predicts the genre that is most common in train.
  Its `predict_proba` returns the one-hot vector of that genre for every row.
- `stratified`: for each row it draws a random genre with probability equal to
  that genre's share of the training rows. `predict_proba` returns the one-hot
  vector of the drawn genre, so soft voting over the 10 windows of a clip
  becomes a majority vote of 10 random draws.

**Math: most frequent.** If the model always predicts genre $m$, then for
$m$: $TP = n_m$ (all clips of $m$), $FP = N - n_m$, $FN = 0$; every other
genre has $TP = 0$ and F1 = 0. So

```latex
F1_m = \frac{2\,n_m}{2\,n_m + (N - n_m)} = \frac{2\,n_m}{N + n_m},
\qquad
\text{macro-F1} = \frac{1}{K}\,\frac{2\,n_m}{N + n_m},
\qquad
\text{accuracy} = \frac{n_m}{N}.
```

**Worked example (our data).** The most common train genre in C is
classical (48 of 442 train clips; blues 46, hiphop 47). C valid has
$N = 197$ clips, of which $n_m = 20$ are classical:

```latex
F1_{\text{classical}} = \frac{2 \cdot 20}{197 + 20} = \frac{40}{217} = 0.184,
\quad \text{macro-F1} = 0.184 / 10 = 0.018,
\quad \text{accuracy} = 20 / 197 = 0.102.
```

These match the JSON exactly (`per_genre_f1.classical` 0.184, macro-F1
0.018, accuracy 0.102). Accuracy says "10%, roughly chance", while macro-F1
says "almost nothing". Macro-F1 penalises a model that ignores nine genres,
and that is why it is our primary metric.

**Math: stratified.** If predictions are drawn independently of the truth
with probabilities $q_g$, and genre $g$ makes up a share $p_g$ of valid,
then in expectation precision$_g = p_g$ and recall$_g = q_g$, so

```latex
\mathbb{E}[F1_g] \approx \frac{2\,p_g\,q_g}{p_g + q_g} \approx 0.1 \quad\text{when } p_g \approx q_g \approx 0.1.
```

So the expected macro-F1 is about 0.1. Our draw gave 0.128 (accuracy 0.137,
window accuracy 0.110). The gap from 0.1 is luck in the random draws: one
seed, 197 clips.

### 2.3 Standardisation (why scaling matters)

**Intuition.** Our 57 features are in wildly different units. Measured on the
train windows, the standard deviations run from 0.002 (`harm_rms_var`) to
about 1.5 million (`rolloff_var`, which is in Hz²). Any method built on
distances or on one penalty for all weights silently treats a 1-unit change
as equally important in every feature. Without scaling, "distance" means
"difference in roll-off variance" and nothing else.

**Math.** For each feature $j$, using **training rows only**:

```latex
\mu_j = \frac{1}{n}\sum_{i \in \text{train}} x_{ij},
\qquad
\sigma_j = \sqrt{\frac{1}{n}\sum_{i \in \text{train}} (x_{ij} - \mu_j)^2},
\qquad
z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}.
```

Valid and test rows are transformed with the **train** $\mu, \sigma$.
Putting `StandardScaler` inside the `Pipeline` guarantees this: each
`fit(X_train)` refits the scaler on exactly the rows the classifier is
trained on.

**Worked example.** Take two train windows, `blues.00029` window 0 and
`blues.00039` window 0. In raw units, 99.08% of their squared Euclidean
distance comes from one feature, `rolloff_var`. The other 56 features share
the remaining 0.92%. On C valid, kNN with k = 3 scores clip macro-F1
**0.336** unscaled against **0.424** scaled (window accuracy 0.232 against
0.376). Scaling alone is worth nearly 0.09 macro-F1.

### 2.4 k-nearest neighbours

**Intuition.** To classify a new window, find the $k$ training windows closest
to it and let them vote. There is no model to fit: "training" just stores the
data. That is why kNN is called **lazy learning** (it puts off all the work
until a prediction is needed). It assumes that windows close in feature space
tend to share a genre.

**Distance.** sklearn's default is Minkowski with $p = 2$, i.e. Euclidean:

```latex
d(\mathbf{x}, \mathbf{x}') = \Big(\sum_{j=1}^{57} |z_j - z'_j|^p\Big)^{1/p},
\qquad p = 2 \;\Rightarrow\; \text{Euclidean}, \quad p = 1 \;\Rightarrow\; \text{Manhattan}.
```

**Voting.** With $N_k(\mathbf{x})$ the $k$ nearest training windows:

```latex
\text{uniform:}\quad \hat p(g \mid \mathbf{x}) = \frac{1}{k}\sum_{i \in N_k(\mathbf{x})} \mathbb{1}[y_i = g]
\qquad
\text{distance:}\quad \hat p(g \mid \mathbf{x}) = \frac{\sum_{i \in N_k} w_i\,\mathbb{1}[y_i = g]}{\sum_{i \in N_k} w_i},\;\; w_i = \frac{1}{d(\mathbf{x}, \mathbf{x}_i)}
```

These window probabilities are what our soft vote averages over a clip's
10 windows. With k = 3 uniform, each window's probability vector holds only
values in {0, 1/3, 2/3, 1}.

**Bias and variance as k changes.** For a query point, the expected squared
error of a kNN estimate (regression form, the cleanest to write down) splits
into

```latex
\mathbb{E}\big[(y - \hat f(\mathbf{x}))^2\big]
= \underbrace{\sigma^2}_{\text{noise}}
+ \underbrace{\Big(f(\mathbf{x}) - \tfrac{1}{k}\textstyle\sum_{i \in N_k} f(\mathbf{x}_i)\Big)^2}_{\text{bias}^2}
+ \underbrace{\sigma^2 / k}_{\text{variance}}.
```

Small $k$: low bias (only the closest points), high variance (one noisy
neighbour decides). Large $k$: the vote averages over more points (variance
falls as $1/k$), but those points are farther away and may belong to other
genres (bias grows). The best $k$ balances the two.

**Worked example (our data).** Clip macro-F1, train vs valid (window
accuracy in brackets):

| k (uniform) | train | valid |
|---|---|---|
| 1 | 1.000 (1.000) | 0.409 (0.352) |
| 3 | 1.000 (0.971) | 0.424 (0.376) |
| 25 | 0.940 (0.852) | 0.412 (0.383) |

k = 1 is perfect on train, because every train window's nearest neighbour is
itself. That is the textbook picture of overfitting (zero bias on train, huge
variance). As k grows, train performance drops and window accuracy on valid
rises slightly. Clip macro-F1 on valid hardly moves, though (Section 6).

**A leakage fact this exposes.** In scaled train space, for **81.2%** of
train windows the nearest *other* window comes from the **same clip**.
Windows of one clip are near-copies of each other in feature space. So under
split A (window-random), a valid window's nearest neighbour would usually be
a sibling window from the same clip in train, and kNN would look excellent
for the wrong reason. Lesson 04's ladder exists to measure this.

**Curse of dimensionality.** In high dimensions "near" stops meaning much.
One way to see it: to capture a fraction $r$ of uniformly spread data inside
a cube around a point, the cube's edge must be $r^{1/d}$ of the full range.
For $r = 1\%$ and $d = 57$:

```latex
e = 0.01^{1/57} = e^{\ln(0.01)/57} = e^{-0.0808} \approx 0.92,
```

so the "local" neighbourhood spans 92% of every feature's range. A related
effect is **distance concentration**: all pairwise distances become similar.
In our scaled train data the median nearest-other-window distance is 3.37,
the median of all pairwise distances is 9.88, and for the median window the
nearest/farthest ratio is 0.102. Real audio features lie close to a
lower-dimensional surface, so it is not hopeless, but the neighbours are less
"local" than the word suggests. This is one reason kNN tops out early.

**Cost.** No fitting, but every prediction compares the query with all
$n = 4{,}420$ train windows in 57 dimensions, so the cost is $O(n\,d)$ per
query. That is fine here, and painful for millions of rows.

### 2.5 Multinomial logistic regression

**Intuition.** Give each genre $g$ a weight vector $\mathbf{w}_g$ and a bias
$b_g$. Score a window by the dot product (a weighted sum of its 57 scaled
features), then turn the 10 scores into probabilities. Training adjusts the
weights so the true genre gets high probability.

**Softmax.**

```latex
z_g = \mathbf{w}_g^\top \mathbf{z} + b_g \quad(\text{logit}),
\qquad
p(g \mid \mathbf{z}) = \frac{e^{z_g}}{\sum_{h=1}^{K} e^{z_h}}.
```

Every $p_g$ is positive and they sum to 1. Adding the same constant to every
logit changes nothing, so only differences between logits matter.

*Small example.* Logits $(2, 1, 0)$ give $e^z = (7.389, 2.718, 1)$, with sum
$11.107$, so $p = (0.665, 0.245, 0.090)$.

**Cross-entropy loss.** For one window with true genre $y$ the loss is
$-\log p(y \mid \mathbf{z})$. In the example, if the truth is class 1, the
loss is $-\ln 0.665 = 0.408$. If the truth were class 3, it would be
$-\ln 0.090 = 2.41$: confident and wrong costs a lot. Averaged over the data,
this is the negative log-likelihood of a categorical model.

**L2 penalty and what C means.** sklearn minimises

```latex
\min_{W,\,\mathbf{b}} \;\; C \sum_{i=1}^{n} -\log p(y_i \mid \mathbf{z}_i) \;+\; \tfrac12 \lVert W \rVert_2^2 .
```

$C$ is the **inverse** regularisation strength. Small $C$ means the penalty
dominates: weights are pushed towards 0, so the model is simpler and
higher-bias. Large $C$ means the data term dominates: weights can grow to fit
the training data closely, so variance is higher. The gradient for class $g$
is

```latex
\frac{\partial}{\partial \mathbf{w}_g} = C \sum_i \big(p(g \mid \mathbf{z}_i) - \mathbb{1}[y_i = g]\big)\,\mathbf{z}_i + \mathbf{w}_g ,
```

i.e. "predicted minus actual", weighted by the input. The problem is convex,
so lbfgs (a quasi-Newton method) finds the single optimum. Note that the L2
penalty treats all weights alike, which is a second reason to standardise:
otherwise a feature in Hz² would need a tiny weight and one in [0, 1] a
large one, and the penalty would punish them very differently.

**Linear decision boundaries.** The model picks $g$ over $h$ when
$z_g > z_h$, i.e. when

```latex
(\mathbf{w}_g - \mathbf{w}_h)^\top \mathbf{z} + (b_g - b_h) > 0 ,
```

which is one side of a hyperplane. So logistic regression carves the 57-d
space into convex regions with flat walls. It cannot represent "genre X if
feature 1 is high *and* feature 2 is low, or both are extreme" unless such
combinations happen to be linearly separable. kNN can draw any shape, but it
needs enough nearby data to do so.

**Worked example (our data).** Across the grid, as $C$ grows the weight norm
roughly doubles each step, the solver needs more iterations, train F1 rises
steadily, and valid F1 peaks at C = 0.1:

| C | ‖W‖ | lbfgs iterations | train clip F1 | valid clip F1 |
|---|---|---|---|---|
| 0.01 | 4.11 | 41 | 0.843 | 0.447 |
| 0.1 | 9.27 | 65 | 0.902 | **0.468** |
| 1 | 18.3 | 122 | 0.931 | 0.452 |
| 10 | 37.5 | 247 | 0.950 | 0.432 |

This is the bias-variance curve in miniature: too much penalty underfits,
too little overfits. Note the large train/valid gap even at the best C (0.90
vs 0.47). Some of it is ordinary overfitting. Some is the artist-disjoint
split: the model learns artist and recording traits that do not transfer.

*Softmax on one real clip.* For valid clip `blues.00000` (C = 0.1), the mean
window probabilities are disco 0.261, rock 0.187, country 0.175, jazz 0.132,
blues 0.107 (others lower). Soft vote: **disco**, which is wrong. The
per-window top choices were rock, disco, jazz, disco, rock, disco, disco,
country, disco, country, so the majority vote also says disco (5 of 10).

### 2.6 Why windows are scored per clip here

Every model is trained on windows but scored on clips (lesson 06):
`ClipMacroF1Scorer` averages a clip's 10 window probabilities and takes the
argmax. Our window accuracies (0.376 kNN, 0.434 logreg) are lower than the
clip accuracies (0.452, 0.482). Voting smooths over individual bad windows,
which is exactly why the clip is the unit of prediction.

---

## 3. Why this choice, and the alternatives we rejected

| Choice | Rejected | Why |
|---|---|---|
| Two dummy floors | Only "random guessing" stated in words | A measured floor under *our* metric and *our* valid set; most-frequent shows how macro-F1 punishes ignoring classes; stratified shows chance. |
| kNN + logistic regression as baselines | Jumping straight to SVM/XGBoost/CNN | Without a simple baseline we cannot tell whether complexity helps. kNN (non-parametric, local) and logreg (parametric, linear) are opposite ends of the bias-variance spectrum. |
| `StandardScaler` inside a `Pipeline` | Scaling the whole dataset once up front | Fitting the scaler on train+valid(+test) uses their means and SDs, which is a learned statistic leaking from held-out data. The pipeline refits it on each fit's training rows. |
| `PredefinedSplit` (train → valid, one fold) | k-fold over windows | k-fold over windows puts sibling windows in train and validation folds (81% of nearest neighbours are siblings), so it rewards memorisation and would choose k = 1. |
| `refit=False`, refit best on train only | `refit=True` | `refit=True` retrains on train+valid, so per-genre F1 and the confusion matrix reported "on valid" would come from a model that saw valid. |
| Grid ties → first in grid order | Hand-picked "simplest" rule | Avoids inventing a rule after seeing results; all grid values are saved. |
| `threadpool_limits(1)` for the refit | Default threading | lbfgs results changed with the BLAS thread count (C = 1: valid F1 0.457 multi-threaded vs 0.452 single); one thread makes results machine-independent (see DECISIONS.md). |
| Grids in `config.yaml` | Hard-coded in the script | Project rule: all constants in config. |
| Grid k ∈ {1,3,5,9,15,25} (odd, roughly log-spaced) | Every k from 1 to 50 | Odd k avoids two-way vote ties; log spacing covers small and large neighbourhoods cheaply; a finer grid would mostly fit valid noise (Section 6). |

---

## 4. Code walkthrough

### `src/models.py`

- **L20–27 `dummy_most_frequent`, `dummy_stratified`**: thin wrappers;
  stratified takes `random_state` from config seed (42) so its random draws
  are reproducible.
- **L30–31 `_scaled`**: the single place where `StandardScaler` is chained
  in front of a classifier. The step names `scale` and `clf` are what the
  grid keys refer to (`clf__C`, `clf__n_neighbors`).
- **L34–36 `knn_pipeline`**: `KNeighborsClassifier(n_neighbors, weights)`,
  default Euclidean metric.
- **L39–41 `logreg_pipeline`**: `LogisticRegression(C, max_iter=2000)`.
  sklearn's default for this is L2 penalty, lbfgs, multinomial softmax for
  more than two classes. `max_iter` comes from config. All four fits
  converged (at most 247 iterations).
- **L44–48 `svm_pipeline`**, **L51–53 `xgb_model`**: written now for the
  next step and not yet tuned. `probability=True` is needed for soft voting
  and is deprecated in sklearn 1.9 (noted in the code).
- **L57 `MODELS`**: CLI name → (factory, key of its grid in config, or
  `None` for no grid).

### `src/train_classical.py`

- **L31–41 `load_windows`**:
  - L34: `index.csv` row numbers become a `row` column, since row *i* of
    `index.csv` is row *i* of `handcrafted.npy`.
  - L36–37: `load_split` returns this part's windows. It raises
    `PermissionError` for `test` unless the `pre-test-freeze` tag exists, so
    the test lock covers this loader too. `merge(..., validate="1:1")` fails
    loudly if a window appears twice.
  - L38: every split row must have features.
  - L39–40: the feature matrix is indexed by `clip_id`. `ClipMacroF1Scorer`
    reads clip ids from `X.index`, and sklearn keeps a DataFrame's index when
    it slices rows inside `GridSearchCV`.
- **L44–47 `param_grid`**: turns `{n_neighbors: [...]}` from config into
  `{clf__n_neighbors: [...]}`. Dummies get `{}`, which `GridSearchCV` treats
  as a single candidate.
- **L50–84 `tune_and_score`**:
  - L53–55: stack train then valid. The predefined split marks train rows
    `-1` (always trained on) and valid rows `0` (the one test fold).
  - L56–58: grid search; every candidate is fit on the 4,420 train windows
    and scored by clip macro-F1 on the 197 valid clips. `n_jobs=-1` runs
    candidates in parallel.
  - L60: `rank_test_score == 1` may hold for several candidates; `argmin`
    takes the first in grid order.
  - L64–65: refit the winner on train only, with one BLAS thread (Section 3).
  - L66–71: window probabilities → soft-vote clip predictions → clip scores
    with labels fixed to the 10 genres in config order.
  - L72: **self-check**: the refit model's valid macro-F1 must equal the
    grid search's score for the same setting. If they differed, the
    reported diagnostics would not belong to the selected model.
  - L73–84: the JSON content; majority-vote F1 and window accuracy are
    diagnostics only.
- **L87 `plot_knn_k`**: one line per weighting, log-scaled k axis, y from 0 so
  small differences are not blown up, dashed stratified-dummy floor.
- **L119 `main`**: parses the CLI, expands `dummy`, seeds, loads the two parts
  once, loops over models, writes JSONs, plots, prints the table.

### `tests/test_models.py`

- `test_scaler_is_inside_pipeline_and_fit_on_train_only`: after `fit`, the
  scaler's `mean_` equals the mean of the rows it was fit on.
- `test_every_model_gives_probabilities_and_grid_keys_match`: every model
  in `MODELS` accepts the first value of its config grid and returns valid
  probability rows.
- `test_load_windows_rows_match_features_index`: the first valid row's
  features equal the right row of `handcrafted.npy`; all windows of a clip
  share a label.
- `test_load_windows_refuses_test`: the test part stays locked.

---

## 5. Pitfalls

1. **Fitting the scaler outside the pipeline on all data.** Valid and test
   means and SDs leak into training. The effect is small for a scaler but
   large for feature selection or PCA, and the habit is what matters.
2. **Tuning with k-fold over windows.** Sibling windows in both folds mean
   the nearest neighbour is usually a copy of the same clip. CV would pick
   k = 1 and report scores far above what new clips get.
3. **Using split A to "show" kNN works.** Same reason: under a window-random
   split, valid windows have siblings in train.
4. **`refit=True` and then reporting "valid" confusion matrices.** The model
   would have trained on valid.
5. **Over-reading the k plot.** All 12 kNN settings lie between 0.402 and
   0.424. With 197 valid clips, a couple of clips flipping moves macro-F1 by
   about 0.01, so "k = 3 is best" is a tie-break, not a finding. A plot with
   the y-axis zoomed to 0.40–0.43 makes this look dramatic. Ours starts at 0
   for that reason.
6. **Reporting only accuracy for the dummy.** 0.102 accuracy looks like
   "chance level"; macro-F1 0.018 shows it is useless.
7. **Ignoring thread-dependent numerics.** Without the single-thread refit,
   the same code gave 0.452 or 0.457 for logreg C = 1 depending on the core
   count, and the self-check could fail on another machine.
8. **Treating windows as independent samples in the loss.** Logistic
   regression's likelihood assumes independent rows. The 10 windows of a
   clip are not, so the model is more confident than it should be. This is
   harmless for ranking classes, but its probabilities are not calibrated
   clip-level probabilities.
9. **Forgetting `max_iter`.** sklearn's default of 100 iterations would stop
   C = 1 and C = 10 early (122 and 247 needed), and the model would not be
   the optimum the math describes.

---

## 6. Results from this step

Split C, valid part (197 clips), from `results/val/C_s0_*.json`:

| Model | Best params | Clip macro-F1 (soft) | Majority-vote macro-F1 | Clip accuracy | Window accuracy |
|---|---|---|---|---|---|
| Dummy most-frequent | n/a | 0.018 | 0.018 | 0.102 | 0.102 |
| Dummy stratified | n/a | 0.128 | 0.128 | 0.137 | 0.110 |
| kNN | k = 3, uniform | 0.424 | 0.424 | 0.452 | 0.376 |
| Logistic regression | C = 0.1 | **0.468** | 0.468 | 0.482 | 0.434 |

(Soft and majority differ only in the 4th decimal: 0.4681 vs 0.4676 for
logreg, where they disagree on 3 clips; 0.4241 vs 0.4244 for kNN, 21 clips.)

**Figure:** `results/figures/val_knn_k.png`. Valid macro-F1 is flat across
k: uniform 0.409, 0.424, 0.412, 0.410, 0.402, 0.412 for k = 1, 3, 5, 9, 15, 25;
distance weighting 0.409, 0.416, 0.413, 0.413, 0.402, 0.412. The two
weightings are identical at k = 1 (one neighbour, nothing to weight) and
nearly identical elsewhere. Reading: the neighbourhood size barely matters,
so kNN's limit is the feature space (distances that mix genres), not the
choice of k.

**Logistic regression grid:** C = 0.01 → 0.447, 0.1 → 0.468, 1 → 0.452,
10 → 0.432 (inverted U, Section 2.5).

**Per-genre F1 (valid):**

| Genre | kNN | Logreg |
|---|---|---|
| blues | 0.000 | 0.323 |
| classical | 0.800 | 0.976 |
| country | 0.222 | 0.278 |
| disco | 0.516 | 0.585 |
| hiphop | 0.000 | 0.167 |
| jazz | 0.650 | 0.632 |
| metal | 0.542 | 0.516 |
| pop | 0.720 | 0.424 |
| reggae | 0.326 | 0.353 |
| rock | 0.465 | 0.429 |

**Interpretation.**
- Both baselines are far above the floor (0.42–0.47 vs 0.13), so the
  handcrafted features carry real genre information even when no artist is
  shared between train and valid.
- Classical is easy (logreg 20 of 20 classical clips correct). Blues and
  hiphop are hard. kNN never predicts blues at all: 13 of 23 blues clips go
  to country. Its 7 hiphop predictions are all wrong, and the 18 real hiphop
  clips go to metal (11) and reggae (7). Logistic regression recovers some
  (5 blues, 3 hiphop correct) but still sends 10 hiphop clips to metal.
  These two genres pull macro-F1 down, which accuracy alone would hide.
- Logistic regression beats kNN by 0.044 macro-F1 on valid. That has **not**
  been tested: valid is also the set we tuned on, so both numbers are
  slightly optimistic, and no paired bootstrap has been run. The real
  comparison comes on test, after the freeze.
- Train vs valid gaps are large (logreg 0.902 vs 0.468; kNN k = 3 1.000 vs
  0.424).

Test data: not loaded (no `pre-test-freeze` tag; `load_split` would raise).

---

## 7. Viva questions with model answers

**Q1. Why do you need a dummy classifier if you know chance is 10%?**
"Chance" depends on the metric and the class balance. Under our metric and
valid set, always predicting classical gives macro-F1 0.018 (not 0.1),
accuracy 0.102. Random stratified guessing gives 0.128. A measured floor
also catches pipeline bugs: a model at the floor has learned nothing.

**Q2. Derive the most-frequent dummy's macro-F1.**
Only the predicted genre has non-zero F1: $2n_m/(N+n_m) = 40/217 = 0.184$,
divided by 10 genres = 0.018.

**Q3. Why must kNN features be standardised, and why inside a Pipeline?**
Euclidean distance adds squared differences, so features with large units
dominate. For two train windows, 99% of the squared distance came from
`rolloff_var`. Unscaled kNN scored 0.336 vs 0.424. Inside a Pipeline, the
mean and SD are learned from the training rows of each fit only, so valid
and test never influence them.

**Q4. What if you had tuned kNN with 5-fold cross-validation over windows?**
Sibling windows of the same clip would fall in both training and
validation folds. 81% of windows have a sibling as nearest neighbour, so
k = 1 would look nearly perfect and would be chosen. The reported score
would reflect recognising clips, not genres.

**Q5. What does C do in logistic regression? What if C = 1,000?**
C multiplies the data loss against a fixed ½‖W‖² penalty, so it is the
inverse regularisation strength. Larger C means bigger weights and closer
fit to train. In our grid, ‖W‖ went from 4.1 (C = 0.01) to 37.5 (C = 10) and
valid F1 fell after C = 0.1. At C = 1,000 the penalty is almost gone, so
expect higher train F1, lower valid F1, and many more lbfgs iterations
(possibly hitting `max_iter`).

**Q6. Why is logistic regression's decision boundary linear even though it
outputs probabilities?**
It predicts $g$ over $h$ when $(\mathbf{w}_g-\mathbf{w}_h)^\top\mathbf{z} +
(b_g-b_h) > 0$, which is a hyperplane. The softmax is monotone, so it only
rescales the logits into probabilities without changing which is largest.

**Q7. Your k plot shows k = 3 is best. Is k = 3 really better than k = 15?**
No evidence for it. All settings are within 0.402–0.424, and with 197
clips a couple of clips flipping moves macro-F1 by about 0.01. k = 3 won the
tie-break on this valid set; on another valid set another k could win. The
honest reading is that k does not matter much here.

**Q8. Why does k = 1 give train macro-F1 1.0?**
Each train window's nearest neighbour is itself (distance 0), so it always
predicts its own label. Training accuracy for k = 1 says nothing about
generalisation.

**Q9. What if you used `refit=True`?**
GridSearchCV would retrain the best setting on train+valid. Any "valid"
score computed from that model would be on data it trained on. We keep
`refit=False` and refit on train only, then assert that the refit score
equals the grid score.

**Q10. Logreg beats kNN by 0.044. Can you claim logistic regression is
better?**
Not yet. Valid was used to pick hyperparameters, so both numbers are
optimistic. There is no confidence interval, and clips by one artist are
correlated. The claim needs a paired artist-cluster bootstrap on the test
part after the freeze.

**Q11. Why is kNN called a lazy learner, and what does that cost?**
It builds no model at fit time; it stores the training windows. All the
work happens at prediction: distances to all 4,420 windows in 57 dimensions
per query. Training is instant and prediction is slow, and the whole training
set must be kept.

**Q12. What is the curse of dimensionality, concretely here?**
In 57 dimensions, a neighbourhood holding 1% of uniformly spread data spans
about 92% of each feature's range ($0.01^{1/57}$). Distances also
concentrate: the median nearest-window distance (3.37) is a third of the
median pairwise distance (9.88). "Nearest" neighbours are not very near,
which caps what kNN can do.

---

## 8. Further reading

- Cover & Hart (1967), *Nearest neighbor pattern classification*, IEEE
  Trans. Information Theory 13(1). https://doi.org/10.1109/TIT.1967.1053964
- Beyer, Goldstein, Ramakrishnan & Shaft (1999), *When is "nearest neighbor"
  meaningful?*, ICDT. https://doi.org/10.1007/3-540-49257-7_15
- Hastie, Tibshirani & Friedman (2009), *The Elements of Statistical
  Learning*, 2nd ed., ch. 2.3 and 2.5 (kNN, curse of dimensionality), ch. 4.4
  (logistic regression), ch. 7 (bias-variance). https://hastie.su.domains/ElemStatLearn/
- Bishop (2006), *Pattern Recognition and Machine Learning*, §4.3.4
  (multiclass logistic regression, softmax).
- scikit-learn user guide: *Nearest Neighbors*
  https://scikit-learn.org/stable/modules/neighbors.html ;
  *Logistic regression* https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression ;
  *DummyClassifier* https://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators ;
  *Pipelines and composite estimators* https://scikit-learn.org/stable/modules/compose.html
- Tzanetakis & Cook (2002), *Musical genre classification of audio
  signals*, IEEE Trans. Speech and Audio Processing 10(5) (the original
  GTZAN baselines, including kNN). https://doi.org/10.1109/TSA.2002.800560
- Sturm (2013), *The GTZAN dataset: Its contents, its faults, their effects
  on evaluation, and the future use of GTZAN*. https://arxiv.org/abs/1306.1461
