# Lesson 08: Support vector machines and gradient-boosted trees

Lesson 07 set the reference points: a floor (stratified dummy, valid clip
macro-F1 0.128) and two baselines (kNN 0.424, logistic regression 0.468).
This step adds the two strongest "classical" models for tabular features:

- an **RBF support vector machine (SVM)**, a maximum-margin classifier whose
  kernel lets it draw curved boundaries, and
- **XGBoost**, an ensemble of small decision trees built one after another
  by gradient boosting.

Both see exactly the same inputs as the baselines: the 57 handcrafted
features of each 3 s window, split C (fault-filtered, artist-disjoint),
442 train clips = 4,420 windows, 197 valid clips = 1,970 windows. Both are
scored the same way: soft vote over a clip's 10 windows, then clip macro-F1
on valid. The test part was never loaded.

The short answer of this step: **neither beats logistic regression on
valid**. The SVM reaches 0.458, XGBoost 0.419, logistic regression 0.468.
Most of this lesson explains why that is a plausible result and not a bug,
and what the tuning plots tell us about our data.

---

## 1. What we built

| File | What it does |
|---|---|
| `config.yaml` (`grids.svm`, `grids.xgb`, `xgb`) | The SVM grid C x gamma, the XGBoost grid max_depth x learning_rate, and XGBoost's fixed settings (2000 trees max, subsample 0.8, colsample_bytree 0.8, early stopping patience 50 on `mlogloss`). |
| `src/models.py` | `svm_pipeline` (StandardScaler + `SVC(kernel="rbf", probability=True, random_state=42)`) is now registered in `MODELS` as `"svm"`, so it is tuned by the same `GridSearchCV` code as kNN and logreg. `xgb_model` returns an `XGBClassifier`; it is deliberately *not* in `MODELS` (Section 4). |
| `src/train_classical.py` | New `valid_report` (shared clip scoring), `tune_xgb` (manual loop with early stopping), `plot_svm_heatmap`, `plot_xgb_curve`, `summary_table` (reads every result JSON for the split, so it lists all models even if you only re-ran some). |
| `tests/test_models.py` | New `test_tune_xgb_early_stops_and_picks_by_valid_clip_f1`; the existing pipeline tests now also cover the SVM through `MODELS`. |
| `scripts/lesson08_examples.py` | Re-computes every worked-example number in this lesson (kernel values, Gini split, support vectors, scaling test, first-round tree splits). Uses train and valid only. |
| `results/val/C_s0_svm.json`, `results/val/C_s0_xgb.json` | Best params, valid clip macro-F1 (soft and majority), accuracy, window accuracy, per-genre F1, confusion, the full grid. The SVM file also stores the numeric value of `gamma="scale"`; the XGBoost file stores the best iteration of every setting and the train/valid loss curve of the chosen one. |
| `results/figures/val_svm_heatmap.png` | Valid clip macro-F1 for every (C, gamma). |
| `results/figures/val_xgb_curve.png` | Train and valid window mlogloss against boosting rounds, chosen XGBoost setting. |

```bash
# all classical models on split C (about 80 s on a 16-thread laptop)
.venv/Scripts/python -W ignore -m src.train_classical --split C --seed 0 --models dummy,knn,logreg,svm,xgb
# or
make tune
# only the new two (the summary table still lists all models from their JSONs)
.venv/Scripts/python -W ignore -m src.train_classical --models svm,xgb
# worked examples quoted below
PYTHONPATH=. .venv/Scripts/python -W ignore scripts/lesson08_examples.py
# tests
.venv/Scripts/python -m pytest -q tests/test_models.py
```

Re-running gives byte-identical `C_s0_svm.json` and `C_s0_xgb.json` (checked),
and the dummy, kNN and logreg JSONs are unchanged in content from the
previous commit (checked by comparing the parsed JSON).

---

## 2. Concepts in depth

### 2.1 Maximum margin: the idea behind the SVM

**Intuition.** Take two genres and imagine their windows as two clouds of
points in feature space. Logistic regression places a line (a hyperplane in
57 dimensions) that makes the training data *probable*. Many lines separate
the clouds equally well on training data, though. The SVM picks the one
that stays **as far as possible from the nearest points of both classes**:
the widest "street" between them. The reasoning is that a boundary with a
wide safety gap is less likely to flip when new points jitter a little, so
it should generalise better.

**Math.** A linear classifier is $f(x) = w^\top x + b$, predicting class
$+1$ if $f(x) > 0$ and $-1$ otherwise. The signed distance of a point to
the hyperplane $f(x)=0$ is $f(x)/\lVert w\rVert$. Rescaling $w$ and $b$
does not move the hyperplane, so we can fix the scale by demanding that the
nearest training points have $y_i f(x_i) = 1$. The street then has width
$2/\lVert w \rVert$, and maximising it is the same as minimising
$\lVert w \rVert^2$:

```latex
\min_{w,b}\ \tfrac12 \lVert w\rVert^2
\quad\text{subject to}\quad y_i\,(w^\top x_i + b) \ge 1 \quad \text{for all } i .
```

The points that sit exactly on the edge of the street ($y_i f(x_i) = 1$)
are the **support vectors**. Only they determine the solution: delete any
other training point and the boundary does not move.

### 2.2 Soft margin, hinge loss, and C

Real data, ours included, cannot be separated perfectly (a blues window and
a rock window can have almost the same features). The **soft margin** lets
point $i$ violate the street by a slack $\xi_i \ge 0$ at a price:

```latex
\min_{w,b,\xi}\ \tfrac12 \lVert w\rVert^2 + C \sum_{i=1}^{n} \xi_i
\quad\text{s.t.}\quad y_i\,(w^\top x_i + b) \ge 1 - \xi_i,\ \ \xi_i \ge 0 .
```

At the optimum each slack equals $\xi_i = \max(0,\ 1 - y_i f(x_i))$, so the
problem is the same as

```latex
\min_{w,b}\ \tfrac12 \lVert w\rVert^2 + C \sum_{i=1}^{n} \underbrace{\max\bigl(0,\ 1 - y_i f(x_i)\bigr)}_{\text{hinge loss}} .
```

The **hinge loss** is zero for points that are correct *and* outside the
street ($y_i f(x_i) \ge 1$), and grows linearly as a point moves into the
street and across the boundary. Compare logistic regression's loss
$\log(1 + e^{-y f(x)})$: it never reaches exactly zero, so every point
pulls a little on the boundary. With the hinge, well-classified points
pull nothing; that is why only support vectors matter.

**C** is the price of a violation (sklearn's C plays the same role as in
logistic regression: an *inverse* regularisation strength).

- **Small C**: violations are cheap, so the optimiser prefers a wide, simple
  street and tolerates many misclassified training points. High bias, low
  variance. Many points end up inside the street, so many support vectors.
- **Large C**: violations are expensive, so the optimiser bends or narrows
  the street to fit the training points. Low bias, high variance.

### 2.3 Kernels and the kernel trick

**Intuition.** A straight boundary in the original 57 features may be too
simple. One fix is to add new features (squares, products, ...) and draw a
straight boundary in the bigger space, which is curved when seen in the
original space. The **kernel trick** lets the SVM do this in an enormous,
even infinite, feature space without ever computing those features.

**Math.** Solving the soft-margin problem with Lagrange multipliers
$\alpha_i$ gives the **dual problem**:

```latex
\max_{\alpha}\ \sum_i \alpha_i - \tfrac12 \sum_i\sum_j \alpha_i\alpha_j\, y_i y_j\, x_i^\top x_j
\quad\text{s.t.}\quad 0 \le \alpha_i \le C,\ \ \sum_i \alpha_i y_i = 0 ,
```

with the classifier

```latex
f(x) = \sum_{i} \alpha_i y_i\, x_i^\top x + b .
```

Training points with $\alpha_i > 0$ are the support vectors. Notice that the
data appear **only through dot products** $x_i^\top x_j$. If we map each
point through some $\phi(x)$ into a richer space, we only ever need
$\phi(x_i)^\top\phi(x_j)$. A **kernel** is a function that returns this dot
product directly, $K(x, x') = \phi(x)^\top \phi(x')$, without building
$\phi$. Replacing every $x_i^\top x_j$ by $K(x_i, x_j)$ is the kernel trick.
The upper bound $\alpha_i \le C$ is where C enters the dual: a single point
can only push on the boundary up to strength C.

### 2.4 The RBF kernel and gamma

The radial basis function (Gaussian) kernel is

```latex
K(x, x') = \exp\!\bigl(-\gamma\, \lVert x - x' \rVert^2\bigr), \qquad \gamma = \frac{1}{2\sigma^2}.
```

It is a **similarity**: 1 for identical points, falling towards 0 as they
move apart. Its feature map $\phi$ is infinite-dimensional, so with enough
support vectors it can draw almost any boundary. The classifier becomes

```latex
f(x) = \sum_{i \in \text{SV}} \alpha_i y_i \exp\!\bigl(-\gamma \lVert x_i - x\rVert^2\bigr) + b ,
```

a weighted sum of Gaussian "bumps" centred on the support vectors.
**gamma** sets the width of each bump:

- **small gamma** (wide bumps): every support vector influences points far
  away; the sum of many wide bumps is a smooth, almost linear surface.
- **large gamma** (narrow bumps): each support vector only influences its
  immediate neighbourhood; the boundary can wrap tightly around single
  training points, which means overfitting.

sklearn's `gamma="scale"` sets $\gamma = 1/(d \cdot \operatorname{Var}(X))$
with $d$ features and the variance of all entries of the training matrix.
After standardisation every column has variance 1, so on our data
$\gamma_\text{scale} = 1/57 = 0.0175$ (stored as `gamma_scale_value` in the
JSON).

**Worked example on our data** (`scripts/lesson08_examples.py`). Over 5,000
random pairs of standardised train windows, the median squared distance is
95.8 (70.4 for same-genre pairs, 98.5 for different-genre pairs); the 10th
percentile is 44.3. The kernel value at those distances:

| gamma | $K$ at median pair ($d^2 = 95.8$) | $K$ at a close pair ($d^2 = 44.3$) |
|---|---|---|
| 0.001 | 0.909 | 0.957 |
| 0.01 | 0.384 | 0.642 |
| scale = 0.0175 | 0.186 | 0.460 |
| 0.1 | 0.0000692 | 0.0119 |

At gamma = 0.1 even a close pair has similarity 0.01: every training window
is an island, and the model behaves like a lookup table of its training
points. At gamma = 0.001 almost all pairs look similar (0.91 to 0.96), so the
bumps blend into a gentle surface. Note also that same-genre windows are
only a little closer than different-genre windows (70.4 vs 98.5): genres
overlap heavily in this feature space, which already hints that no clever
boundary will separate them cleanly.

Distances are computed on **standardised** features, which is why the SVM
sits in a Pipeline after `StandardScaler`. Without it, `rolloff_var`
(standard deviation about 1.5 million, lesson 07) would dominate
$\lVert x - x'\rVert^2$, and any fixed gamma would mean something different
for every feature.

### 2.5 Reading our heatmap region by region

`results/figures/val_svm_heatmap.png` shows valid clip macro-F1 for every
(C, gamma). Columns are in numeric gamma order, with "scale" placed at its
value 0.0175. The colour scale runs from 0.30 to 0.50 so differences are
visible; the floor is 0.128 and logistic regression 0.468.

| C \ gamma | 0.001 | 0.01 | scale (0.0175) | 0.1 |
|---|---|---|---|---|
| 100 | **0.458** | 0.428 | 0.433 | 0.404 |
| 10 | 0.440 | 0.428 | 0.424 | 0.404 |
| 1 | 0.422 | 0.422 | 0.406 | 0.408 |
| 0.1 | 0.457 | 0.381 | 0.401 | 0.340 |

Region by region, with the support-vector counts and training window
accuracies from `scripts/lesson08_examples.py`:

- **Right column, top (C = 10 or 100, gamma = 0.1): overfitting.** Narrow
  bumps and expensive violations. At C = 100, gamma = 0.1 the model gets
  **100.0%** of training windows right, uses 3,901 of 4,420 windows as
  support vectors (88%), and scores 0.404 on valid. It has memorised the
  training windows. C = 10 and C = 100 give exactly the same valid score
  (0.40433), which suggests that at C = 10 the training set is already fit
  and raising the penalty changes nothing.
- **Bottom right (C = 0.1, gamma = 0.1): the worst cell, 0.340.** Narrow
  bumps, but violations are cheap, so the model cannot even fit the islands:
  training window accuracy 0.485, and 4,306 of 4,420 windows (97%) are
  support vectors (almost every point sits inside the street). It is too
  local to generalise *and* too constrained to fit: underfitting of the
  training set combined with no useful smoothness.
- **Left column (gamma = 0.001): the best region.** Wide bumps give an
  almost linear boundary. With C = 100 (best, 0.458) the model has room to
  use that smooth surface fully: training window accuracy 0.949, 2,113
  support vectors (48%). With C = 0.1 (0.457) it is smoother still and
  scores practically the same.
- **Middle (gamma = 0.01 and scale): in between**, 0.38 to 0.43. More
  curvature than the data supports, without being extreme.
- **The diagonal trade-off.** Moving down (smaller C) and moving left
  (smaller gamma) both make the model simpler. Up to a point, a larger C can
  compensate a smaller gamma and vice versa, so good settings often lie along
  a diagonal rather than at one spot.

**What the heatmap says about our data.** The best cells are those that make
the SVM *closest to linear*, and logistic regression, which is linear,
scores slightly higher (0.468). With 442 training clips from artist-disjoint
groups, the curvature the RBF kernel can add mostly fits artist-specific
detail that does not carry over to new artists.

**Caveat: the best cell is on the corner of the grid** (largest C, smallest
gamma). The true optimum may lie outside it (gamma below 0.001). The grid
was fixed in advance by the step specification and we did not widen it
after seeing valid scores, because widening a grid until a valid score looks
good is a mild form of tuning on valid by hand. It is recorded in
`DECISIONS.md`. Given that C = 0.1 and C = 100 at gamma = 0.001 differ by
only 0.0016, there is little reason to expect a large gain.

**Even the best SVM fits training far better than valid**: 0.949 window
accuracy on train vs 0.433 on valid (both from `predict`). This gap is the
artist-disjoint generalisation gap; it would look much smaller on split A.

### 2.6 One-vs-one multiclass

The SVM is a two-class method. For K = 10 genres sklearn's `SVC` trains
**one-vs-one (OvO)**: one binary SVM for every pair of genres,

```latex
\binom{K}{2} = \frac{K(K-1)}{2} = \frac{10 \cdot 9}{2} = 45 \text{ classifiers},
```

each trained only on the windows of its two genres (about 900 of our 4,420).
`predict` lets each of the 45 classifiers vote, and the genre with most
votes wins. The alternative, one-vs-rest, trains 10 classifiers of "genre g
vs all others"; each sees all 4,420 windows and a 1:9 class imbalance. OvO
trains more but smaller problems, and SVM training time grows faster than
linearly in the number of points, so OvO is usually faster for kernel SVMs.

### 2.7 Platt scaling: probabilities from an SVM

Soft voting needs window **probabilities**, but an SVM only outputs a
signed score $f(x)$ (how far, and on which side). **Platt scaling** fits a
sigmoid on top:

```latex
P(y = 1 \mid x) = \frac{1}{1 + \exp\bigl(A\, f(x) + B\bigr)} ,
```

with A and B chosen to minimise cross-entropy. If A and B were fit on the
same points the SVM was trained on, the scores would be over-optimistic
(training points sit on the right side of the street by construction), so
sklearn (via libsvm) uses an **internal 5-fold cross-validation**: train
on 4/5, collect $f(x)$ on the held-out 1/5, fit A and B on those
held-out scores. For 10 classes, each of the 45 pairwise sigmoids gives
$P(\text{g} \mid \text{g or h})$, and **pairwise coupling** (Wu, Lin & Weng
2004) solves for one 10-class probability vector that agrees best with all
45 pairwise estimates. `random_state=42` fixes the internal folds.

Two consequences matter for us:

1. **`predict` and `argmax(predict_proba)` can disagree**, because
   `predict` uses OvO votes and `predict_proba` uses the coupled
   probabilities. For our best SVM they disagree on **178 of 1,970** valid
   windows. Our reported numbers all come from `predict_proba` (window
   accuracy 0.419 in the JSON; `predict` would give 0.433), so the model we
   tune is the model we score.
2. Platt's internal folds split **windows**, not clips, so sibling windows
   of a clip can be on both sides. That makes the sigmoid a little
   over-confident, but it only changes the probability mapping, never which
   data the valid score is computed on. Recorded in `DECISIONS.md`.

### 2.8 Decision trees: splits and impurity

**Intuition.** A decision tree asks a sequence of yes/no questions about
single features ("is `perc_rms_var` at most 0.000025?") and follows the
answers to a leaf, which holds a prediction. Training chooses each
question greedily: the one that makes the resulting groups as **pure**
(single-genre) as possible.

**Math.** For a node whose windows have genre shares $p_1, \dots, p_K$,
the **Gini impurity** is

```latex
G = 1 - \sum_{k=1}^{K} p_k^2 ,
```

the probability that two windows drawn at random from the node have
different genres. It is 0 for a pure node and $1 - 1/K$ for an even mix.
(Entropy $-\sum_k p_k \log p_k$ is the other common choice and behaves
similarly.) A split sends $n_L$ windows left and $n_R$ right; its quality
is the impurity decrease

```latex
\Delta G = G_\text{parent} - \frac{n_L}{n} G_L - \frac{n_R}{n} G_R .
```

The tree tries every feature and every threshold between sorted values and
keeps the largest $\Delta G$, then recurses on each child until a stopping
rule (maximum depth, minimum node size) holds.

**Worked example on our data.** The 4,420 train windows have genre counts
from 410 (pop) to 480 (classical), so the root Gini is 0.8998, close to the
maximum 0.9. A depth-1 sklearn tree (`scripts/lesson08_examples.py`)
chooses `perc_rms_var <= 2.533e-05`:

- left: 368 windows, 313 of them classical (then 31 jazz, 7 country),
  Gini 0.2686;
- right: 4,052 windows, Gini 0.8959.

Weighted impurity after the split is
$(368 \cdot 0.2686 + 4052 \cdot 0.8959)/4420 = 0.8437$, a decrease of
0.0561. In words: windows whose percussive energy barely varies are almost
all classical. This makes musical sense (little drum activity) and is the
same feature XGBoost picks first, see 2.9.

**Why trees need no scaling.** A split only asks "is $x_j \le t$?". If you
rescale a feature by any increasing function (standardise it, take its log,
multiply by 1000), the order of the values is unchanged, so the same
partitions of the data are available, just with a translated threshold. The
tree, and its predictions, stay the same. We checked this:
XGBoost trained on raw and on standardised features gives valid
probabilities that differ by at most **0.00e+00**, with 0 changed
predictions. (XGBoost's histogram method bins each feature by quantiles,
which are also unchanged by an increasing transform.) For the same reason,
the huge differences in feature units (lesson 07) do not matter to trees.

### 2.9 Boosting as gradient descent in function space

A single deep tree overfits; a single shallow tree underfits. **Boosting**
adds many shallow trees, each correcting the current mistakes of the sum
of the previous ones.

**Intuition.** Ordinary gradient descent improves a *parameter vector* by
small steps against the gradient of the loss. Gradient boosting improves a
*function* $F(x)$ the same way: at each round it computes, for every
training point, which direction the prediction should move to reduce the
loss (the negative gradient), fits a small tree to those directions, and
adds a small multiple of that tree to $F$.

**Math.** For K classes the model keeps one score function per class,
$F_k(x)$, turned into probabilities by the softmax
$p_k(x) = e^{F_k(x)} / \sum_j e^{F_j(x)}$ and trained on cross-entropy
(XGBoost's `multi:softprob` objective, reported as **mlogloss**):

```latex
L = -\frac{1}{n}\sum_{i=1}^{n} \log p_{y_i}(x_i).
```

Round $m$ adds one tree per class, scaled by the **learning rate** $\eta$:

```latex
F_k^{(m)}(x) = F_k^{(m-1)}(x) + \eta\, h_{k,m}(x) .
```

The gradient of the loss with respect to the score $F_k(x_i)$ is the
familiar "predicted minus true":

```latex
g_{ik} = \frac{\partial L_i}{\partial F_k(x_i)} = p_k(x_i) - \mathbb{1}[y_i = k] .
```

Plain gradient boosting (Friedman 2001) fits $h_{k,m}$ to $-g_{ik}$ by least
squares. **XGBoost** (Chen & Guestrin 2016) uses a second-order (Newton)
step: with $h_{ik}$ the second derivative (XGBoost's softmax code uses
$2\,p_k(1-p_k)$), and a tree that sends point $i$ to leaf $q(i)$ with leaf
values $w_j$, it minimises

```latex
\sum_i \Bigl[g_i\, w_{q(i)} + \tfrac12 h_i\, w_{q(i)}^2\Bigr] + \gamma T + \tfrac12 \lambda \sum_{j=1}^{T} w_j^2 ,
```

where $T$ is the number of leaves, $\lambda$ an L2 penalty on leaf values
(default 1) and $\gamma$ a cost per leaf (default 0). With $G_j$, $H_j$ the
sums of $g_i$ and $h_i$ over leaf $j$, the best leaf value and the gain of a
split are

```latex
w_j^* = -\frac{G_j}{H_j + \lambda}, \qquad
\text{Gain} = \tfrac12\Bigl[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\Bigr] - \gamma .
```

This gain plays the role of the Gini decrease in 2.8, but it is computed
from the gradients of *our* loss, so every split directly targets the
current errors.

**Worked example on our data.** In round 1 every window has
$p_k = 0.1$ for all genres (all scores start equal), so for the classical
tree $g = 0.1 - 1 = -0.9$ for classical windows and $+0.1$ for the rest.
The ten round-1 root splits of the chosen model
(`scripts/lesson08_examples.py`) are:

| class tree | root split | gain |
|---|---|---|
| blues | `mfcc11_mean` < -19.6 | 232.5 |
| classical | `perc_rms_var` < 0.000025 | **872.8** |
| country | `mfcc2_mean` < 118.3 | 152.5 |
| disco | `bandwidth_mean` < 2401.6 | 138.0 |
| hiphop | `harm_rms_var` < 0.0039 | 202.2 |
| jazz | `perc_rms_mean` < 0.0299 | 148.4 |
| metal | `mfcc4_mean` < 51.4 | 408.4 |
| pop | `rolloff_mean` < 6250.3 | 468.3 |
| reggae | `mfcc1_var` < 3290.5 | 257.9 |
| rock | `mfcc7_mean` < -16.7 | 99.5 |

The classical tree's root is the same feature and essentially the same
threshold as the Gini stump in 2.8, and has by far the largest gain; rock's
root split has the smallest, matching rock's low F1 later. Thresholds are in
raw feature units (no scaling, 2.8).

### 2.10 The knobs: learning rate, depth, subsampling, early stopping

- **Learning rate (shrinkage) $\eta$.** Each tree's contribution is
  multiplied by $\eta < 1$. Small steps mean each tree corrects only part of
  the error, so later trees can correct differently; this regularises,
  at the price of needing more trees. On our data halving $\eta$ from 0.1 to
  0.05 roughly doubled the number of trees chosen by early stopping: 54 ->
  109 (depth 3), 33 -> 68 (depth 5), 26 -> 62 (depth 7).
- **max_depth.** A depth-d tree can model interactions between up to d
  features. Deeper trees fit faster and overfit sooner: the best valid
  mlogloss was 1.685 / 1.700 at depth 3, 1.728 / 1.739 at depth 5, 1.729 /
  1.760 at depth 7 (learning rate 0.05 / 0.1), and deeper models stopped
  after fewer trees.
- **subsample = 0.8, colsample_bytree = 0.8.** Each tree is grown on a
  random 80% of the training windows and a random 80% of the 57 features.
  This is **stochastic gradient boosting** (Friedman 2002): trees become less
  correlated, which reduces variance, and it is also faster. The random
  draws are fixed by `random_state=42`.
- **n_estimators = 2000 + early stopping (patience 50).** The number of
  trees is the main capacity knob. Rather than grid-searching it, we grow up
  to 2000 trees per class and, after every round, compute **valid window
  mlogloss**. When it has not improved for 50 rounds, training stops, and
  predictions use the trees up to the best round (`best_iteration`, counted
  from 0, so `n_trees = best_iteration + 1`). No setting came close to 2000
  trees (most was 109), so the cap never bound; the code warns if it does.

**Why stop on mlogloss but select on clip macro-F1?** mlogloss is smooth: it
changes a little every round, so "no improvement for 50 rounds" is a
meaningful signal. Clip macro-F1 on 197 clips moves in jumps (one clip
changes it by very roughly 0.003 to 0.005), so stopping on it would stop at
noise. But macro-F1 per clip is the project's metric, so it is what chooses
*between* the six settings, exactly as for kNN, logreg and SVM.

### 2.11 Reading the XGBoost curve

`results/figures/val_xgb_curve.png` (depth 3, learning rate 0.1):

- Round 1: valid mlogloss 2.214. A model knowing nothing scores
  $\ln 10 = 2.303$, so one round already learned something.
- Train loss keeps falling: 0.468 at the best round, 0.223 after 104 rounds.
  Boosting can drive training loss towards 0 with enough trees.
- Valid loss falls until round **54** (1.700), then creeps up (1.767 at
  round 104, where 50 rounds without improvement ended training). Rising
  valid loss while train loss falls is the textbook sign of overfitting;
  early stopping cuts at the turning point.
- The gap at the best round (0.468 vs 1.700) is large. As for the SVM, the
  model learns a lot that is specific to the training artists.

With 54 rounds and one tree per class per round, the chosen model has
54 x 10 = 540 trees of depth at most 3.

### 2.12 Interpreting our numbers

| model | best params | valid clip macro-F1 | majority-vote F1 | clip accuracy | window accuracy |
|---|---|---|---|---|---|
| dummy (most frequent) | | 0.018 | 0.018 | 0.102 | 0.102 |
| dummy (stratified) | | 0.128 | 0.128 | 0.137 | 0.110 |
| kNN | k = 3, uniform | 0.424 | 0.424 | 0.452 | 0.376 |
| logistic regression | C = 0.1 | **0.468** | 0.468 | 0.482 | 0.434 |
| RBF SVM | C = 100, gamma = 0.001 | 0.458 | 0.455 | 0.467 | 0.419 |
| XGBoost | depth 3, lr 0.1, 54 trees | 0.419 | 0.414 | 0.442 | 0.428 |

1. **All four real models are far above the floor** (0.128), so all of them
   learn genre information.
2. **The more flexible models do not win.** The SVM's best settings are its
   most linear ones, and it ends 0.010 below logistic regression. XGBoost is
   0.049 below. On 197 valid clips (13 to 24 per genre) a 0.010 difference is
   within noise (one clip moves macro-F1 by roughly 0.003 to 0.005); the
   artist-cluster bootstrap on the test part (lesson 06) is what will say
   whether any difference is real. On valid alone the honest statement is:
   **no evidence that non-linear boundaries help on these 57 features under
   an artist-disjoint split.**
3. **Why might that be?** Each training row is a 3 s window, but the 4,420
   windows come from only 442 clips and fewer artists, so the effective
   sample size is small. Flexible models spend their extra capacity on
   detail that is shared by windows of the same clip or artist, and that
   detail does not transfer to new artists (train window accuracy 0.949 for
   the best SVM vs 0.433 on valid). A linear model cannot fit that detail,
   which here is an advantage.
4. **XGBoost has the second-best window accuracy (0.428) but gains least
   from voting** (0.428 -> 0.442 clip accuracy, vs logreg 0.434 -> 0.482).
   A plausible reading is that its window errors are more consistent within
   a clip (several windows wrong the same way), so averaging fixes fewer of
   them; we have not tested this directly.
5. **The six XGBoost settings span only 0.400 to 0.419** and the best by
   mlogloss (depth 3, lr 0.05, 1.685) is not the best by clip F1 (depth 3,
   lr 0.1). The choice between them is mostly noise; shallow trees are
   consistently best by mlogloss.
6. **Per-genre F1** (valid clips):

   | genre | logreg | SVM | XGBoost |
   |---|---|---|---|
   | blues | 0.323 | 0.345 | 0.148 |
   | classical | 0.976 | 0.952 | 0.889 |
   | country | 0.278 | 0.250 | 0.298 |
   | disco | 0.585 | 0.510 | 0.475 |
   | hiphop | 0.167 | 0.190 | 0.133 |
   | jazz | 0.632 | 0.686 | 0.645 |
   | metal | 0.516 | 0.510 | 0.500 |
   | pop | 0.424 | 0.471 | 0.467 |
   | reggae | 0.353 | 0.300 | 0.412 |
   | rock | 0.429 | 0.368 | 0.229 |

   Classical is easy for every model; hiphop, country and blues are hard
   for all of them. The biggest single confusions on valid: SVM country ->
   pop (9 of 23 country clips), hiphop -> metal and hiphop -> reggae (7 of
   18 each), rock -> disco (7 of 24); XGBoost blues -> country (11 of 23
   blues clips), hiphop -> metal (10 of 18), rock -> disco (9 of 24). Valid
   genres have 13 (pop) to 24 (rock) clips, because split C is
   artist-disjoint rather than stratified. That the same hard pairs
   recur across three very different models suggests the limit is in the
   features or labels, not the classifier. We look at this in the error
   analysis step.

---

## 3. Why this choice, and the alternatives we rejected

| Choice | Alternative | Why we rejected it |
|---|---|---|
| RBF kernel | Linear SVM | Logistic regression already covers linear boundaries; the RBF kernel with small gamma includes near-linear behaviour as a special case, so the grid can find it (and did). |
| RBF kernel | Polynomial kernel | Extra hyperparameter (degree), numerically less stable on standardised data with large values, and no reason to expect polynomial structure. RBF is the standard default (Hsu, Chang & Lin). |
| Log-spaced grid C {0.1..100} x gamma {scale, 0.001..0.1} | Random search / Bayesian optimisation | 16 settings take 25 s and give a readable heatmap; a viva needs to explain each cell. Random search pays off with many hyperparameters, not two. |
| `probability=True` (Platt) | Majority vote of `predict` | Lesson 06 chose soft voting for all models so they are scored the same way; majority vote throws away confidence. (We still report majority-vote F1: 0.455, vs 0.458 soft.) |
| `probability=True` | `CalibratedClassifierCV(SVC(), ensemble=False)` | Same Platt sigmoid; needed only from sklearn 1.11, we pin 1.9.1 (noted in code). Isotonic calibration needs more data than ~900 windows per pair. |
| XGBoost | Random forest | Random forests average independent deep trees (variance reduction); boosting builds a sequence of shallow trees (bias reduction) and is usually stronger on tabular data. One tree ensemble is enough to answer "do trees help here?". |
| XGBoost | LightGBM / CatBoost | Same family, similar results on 4,420 rows; XGBoost was already pinned in `requirements.txt`. |
| Manual loop + early stopping | `GridSearchCV` with `n_estimators` in the grid | Would multiply fits by the number of tree counts tried and cannot cleanly pass a per-fit `eval_set`. Early stopping finds the tree count in one fit per setting. |
| Early stop on valid **window mlogloss** | Early stop on valid clip macro-F1 | Clip F1 jumps in steps of a few thousandths, so "no improvement for 50 rounds" would trigger on noise. |
| Early stop on valid | An inner hold-out carved from train | Would shrink the already small, artist-disjoint train set and break the equal-size design across splits A/B/B'/C. Cost: valid now picks both the tree count and the setting, so the XGBoost valid score is slightly optimistic. Test answers that. |
| No scaling for XGBoost | StandardScaler in front | Proven to change nothing (2.8); it would only add a step to explain. |
| Fixed grid from the spec, not widened | Extending gamma below 0.001 after seeing the corner | Expanding the grid in the direction valid looks good is tuning by hand on valid. Recorded as a caveat instead. |

---

## 4. Code walkthrough

### `config.yaml`

- `grids.svm`: `C: [0.1, 1, 10, 100]`, `gamma: [scale, 0.001, 0.01, 0.1]`.
  Gamma values are written as plain decimals because **PyYAML reads `1e-3`
  as the string `"1e-3"`** (YAML 1.1 needs a dot: `1.0e-3`). sklearn would
  then reject it, or worse, a string could slip into a comparison.
- `grids.xgb`: `max_depth: [3, 5, 7]`, `learning_rate: [0.05, 0.1]`.
- `xgb`: the fixed settings passed to every XGBoost fit, including
  `early_stopping_rounds: 50` and `eval_metric: mlogloss`.

### `src/models.py`

- **L44–48 `svm_pipeline`**: `_scaled(SVC(kernel="rbf", C, gamma,
  probability=True, random_state=seed))`. The scaler is inside the
  Pipeline, so in every grid fit it is fit on the 4,420 train windows only.
  `random_state` fixes Platt's internal folds.
- **L51–53 `xgb_model`**: `XGBClassifier(random_state=seed, n_jobs=-1,
  **params)`. All other settings come from config through `**params`.
  `n_jobs=-1` uses all cores; XGBoost's CPU histogram method gave
  byte-identical results on re-run.
- **L57–63 `MODELS`**: `"svm"` added, so `param_grid("svm")` builds
  `{clf__C: [...], clf__gamma: [...]}` and the existing `GridSearchCV` path
  tunes it.
- **L64–65**: comment explaining why XGBoost is not in `MODELS`: it needs
  integer labels (`MODELS` models get genre strings) and a per-fit
  `eval_set` for early stopping.

### `src/train_classical.py`

- **L53–67 `valid_report`**: extracted from the old `tune_and_score` so
  that the SVM (through `GridSearchCV`) and XGBoost (through its own loop)
  are scored by **the same code**: soft-vote clip predictions, clip
  macro-F1 and accuracy, majority-vote F1, window accuracy, per-genre F1,
  confusion counts in config genre order.
- **L70–94 `tune_and_score`**: unchanged logic: grid search on the
  predefined train -> valid fold, refit the winner on train only with one
  BLAS thread, assert that its valid F1 equals the grid-search score, then
  `valid_report`.
- **L97–132 `tune_xgb`**:
  - L105–106: grid and fixed settings default to config; the test passes
    small ones.
  - L107–110: genres become integer codes 0..9 **in config order**
    (`pd.Categorical(..., categories=GENRES).codes`); an unknown label would
    become -1, which the assert catches. `classes = np.array(GENRES)` maps
    probability column k back to genre k. If the encoding and decoding used
    different orders, every prediction would be silently mislabelled.
  - L113–114: `itertools.product` over the grid values, in config order
    (depth outer, learning rate inner), so "ties -> first" is well defined.
  - L115–116: fit with `eval_set=[(train), (valid)]`. XGBoost records
    mlogloss for both after every round and **early-stops on the last entry
    (valid)**. The train entry is only for the curve.
  - L117–118: `predict_proba` on valid automatically uses only the trees up
    to `best_iteration`; the probabilities go through `valid_report`.
  - L119–122: one grid row per setting with `best_iteration`, best valid
    mlogloss and valid clip macro-F1; logged as it runs.
  - L123–126: keep the setting with the highest clip macro-F1 (strict `>`,
    so the first of any tie stays) and its loss curves.
  - L128–129: warn if early stopping never triggered (cap reached).
  - L130–132: JSON content, same keys as other models plus
    `best_iteration`, `n_trees`, `fixed_params`, `curve`.
  - No refit step: the model that produced the valid probabilities *is*
    the trained-on-train model.
- **L167–204 `plot_svm_heatmap`**: L174–176 place `"scale"` at its numeric
  value so columns are in true gamma order; L183 sets the colour range to
  the grid's own range rounded out to 0.05 (with a 0 to 0.6 range every cell
  looked the same blue); L189–193 mark the best cell with a bold label and
  an orange frame. Sequential single-hue blue colour map, light = low.
- **L207–233 `plot_xgb_curve`**: train and valid mlogloss per round,
  dashed line at the best round (`best_iteration + 1`, since rounds are
  counted from 1 on the x axis), direct line labels instead of a legend.
- **L236–246 `summary_table`**: reads `results/val/{split}_s{seed}_{name}.json`
  for every model in `MODELS` plus `xgb`, so the printed table always
  covers all models that have results.
- **L249 `main`**: L267 routes `xgb` to `tune_xgb` and every other model to
  `tune_and_score`; L269–271 compute and store `gamma_scale_value` for the
  SVM; L276–282 draw the figures for the models that ran.

### `tests/test_models.py`

- `test_scaler_is_inside_pipeline_and_fit_on_train_only` already covered
  `svm_pipeline`; `test_every_model_gives_probabilities_and_grid_keys_match`
  now also fits the SVM with the first values of its config grid
  (C = 0.1, gamma = "scale") and checks its probabilities.
- **`test_tune_xgb_early_stops_and_picks_by_valid_clip_f1`** (new):
  synthetic data with 10 genres, 3 windows per clip, a 2-setting grid and a
  300-tree cap with patience 10. It checks that (a) both settings are in the
  grid, (b) the reported F1 is the grid maximum, (c) early stopping fired
  (fewer than 300 trees), (d) the stored curve has exactly
  `n_trees + 10` rounds and its minimum is at `best_iteration`. Point (d)
  would catch an off-by-one in `best_iteration` or a curve from the wrong
  setting.

---

## 5. Pitfalls

1. **SVM without scaling.** The RBF kernel works on Euclidean distance, so
   `rolloff_var` (SD about 1.5 million) would decide every similarity, and
   gamma would be meaningless. Scaling *outside* the Pipeline on train+valid
   would also leak valid statistics.
2. **Interpreting gamma without knowing the scale.** gamma = 0.1 is "narrow"
   only because standardised windows are about $\sqrt{95.8} \approx 9.8$
   apart. Change the features (say, add 100 more) and the same gamma means
   something else; `"scale"` adapts, fixed values do not.
3. **Early stopping on test.** If the test part were the `eval_set`, the
   number of trees would be chosen on test and the test score would be
   optimistic. Our `eval_set` is valid, and `load_split` refuses test anyway.
4. **Forgetting that early stopping uses valid.** Valid chose the tree count
   *and* the setting, so XGBoost's valid score carries two selections. Do
   not compare models by a valid difference of 0.01 as if it were a test
   result.
5. **Refitting XGBoost on train+valid with the same `n_trees`.** More data
   usually supports more trees; and there would be no held-out set left to
   early-stop on. Any final refit must be decided before the test freeze
   (lesson 07) and must fix the tree count by a rule, not by looking at test.
6. **Label encoding mismatch.** XGBoost returns probabilities in code order
   0..9. Decoding with a different order (e.g. `np.unique` of the labels
   seen in a subset) would silently swap genres. We encode and decode with
   the same config list.
7. **Using `predict` for one number and `predict_proba` for another** with
   an SVM. They disagree on 178 of 1,970 valid windows for our best SVM.
   Mixing them makes tables inconsistent.
8. **Platt's internal CV over windows.** Sibling windows of a clip can land
   in both internal folds, making the sigmoid over-confident. It does not
   leak into our valid score, but it is why we would not trust the SVM's
   probabilities as calibrated confidences.
9. **Tuning by k-fold over windows** (lesson 07): sibling windows in both
   folds would favour large C and large gamma, because memorising windows
   would score well. Our heatmap would then point at the overfitting corner.
   On split A, the same thing happens at evaluation time.
10. **Reading the best cell on a grid edge as "the optimum".** It is only
    the best of what we tried.
11. **YAML scientific notation.** `1e-3` in YAML 1.1 is a string. Always
    write `0.001` or `1.0e-3`.
12. **Older XGBoost versions** did not automatically predict with
    `best_iteration` (you had to pass `ntree_limit` / `iteration_range`).
    With our pinned 3.2.0 it is automatic; on an old version you would
    silently score the model 50 rounds past its best.

---

## 6. Results from this step

All numbers from `results/val/C_s0_svm.json`, `results/val/C_s0_xgb.json`,
the printed summary, and `scripts/lesson08_examples.py`. Split C, train
4,420 windows (442 clips), valid 1,970 windows (197 clips).

Printed summary table:

```
              model                              best_params  macro_f1  macro_f1_majority  accuracy  window_acc
dummy_most_frequent                                       {}     0.018              0.018     0.102       0.102
   dummy_stratified                                       {}     0.128              0.128     0.137       0.110
                knn {'n_neighbors': 3, 'weights': 'uniform'}     0.424              0.424     0.452       0.376
             logreg                               {'C': 0.1}     0.468              0.468     0.482       0.434
                svm               {'C': 100, 'gamma': 0.001}     0.458              0.455     0.467       0.419
                xgb   {'max_depth': 3, 'learning_rate': 0.1}     0.419              0.414     0.442       0.428
```

**SVM** (`results/figures/val_svm_heatmap.png`): best C = 100, gamma = 0.001,
valid clip macro-F1 0.458 (accuracy 0.467). Grid range 0.340 (C = 0.1,
gamma = 0.1) to 0.458. `gamma="scale"` = 0.0175 on our data. Best model:
2,113 support vectors of 4,420; train window accuracy 0.949; valid window
accuracy 0.433 by `predict`, 0.419 by `argmax(predict_proba)`.

**XGBoost** (`results/figures/val_xgb_curve.png`):

| max_depth | learning_rate | trees (best_iteration + 1) | valid window mlogloss | valid clip macro-F1 |
|---|---|---|---|---|
| 3 | 0.05 | 109 | 1.6849 | 0.411 |
| 3 | 0.1 | **54** | 1.6996 | **0.419** |
| 5 | 0.05 | 68 | 1.7281 | 0.413 |
| 5 | 0.1 | 33 | 1.7388 | 0.404 |
| 7 | 0.05 | 62 | 1.7290 | 0.418 |
| 7 | 0.1 | 26 | 1.7595 | 0.400 |

Chosen: depth 3, learning rate 0.1, 54 rounds (540 trees). Train mlogloss
0.468 at the best round, valid 1.700; first-round valid 2.214; the run
stopped at round 104 (train 0.223, valid 1.767). Raw vs standardised inputs:
maximum probability difference 0.00e+00.

**Ranking on valid** (clip macro-F1): logreg 0.468 > SVM 0.458 > kNN 0.424 >
XGBoost 0.419 >> stratified dummy 0.128 > most-frequent dummy 0.018.

---

## 7. Viva questions with model answers

**Q1. What is a support vector, and what happens if you delete a training
window that is not one?**
A support vector is a training point with $\alpha_i > 0$ in the dual: it
lies on or inside the margin, or on the wrong side. The decision function
is a sum over support vectors only, so deleting any other point leaves the
solution unchanged. Our best SVM uses 2,113 of 4,420 windows (48%), a high
share, which tells us the classes overlap a lot.

**Q2. Explain C and gamma, and read the corner cells of your heatmap.**
C is the price of a margin violation (large C = fit the training data
harder); gamma is the inverse width of the Gaussian bumps (large gamma =
very local influence). C = 100, gamma = 0.1: 100% training accuracy, 0.404
valid: overfitting. C = 0.1, gamma = 0.1: 0.485 training accuracy, 97% of
points support vectors, 0.340 valid: too local to generalise and too
constrained to fit. Both gamma = 0.001 corners (0.457, 0.458) are best: a
wide kernel gives a near-linear boundary, consistent with logistic
regression doing as well.

**Q3. Why is the kernel trick needed at all? Why not compute the features?**
The RBF kernel's feature map is infinite-dimensional, so the features cannot
be computed. The dual only needs dot products between pairs of points, and
the kernel gives those directly at the cost of one distance computation.

**Q4. Your SVM's `predict` and `predict_proba` disagree on 178 windows. Which
is right?**
Neither is "wrong": `predict` takes the OvO majority of 45 pairwise SVMs;
`predict_proba` fits Platt sigmoids on internal-CV scores and couples the
45 pairwise probabilities. They answer slightly different questions. We use
`predict_proba` everywhere because soft voting needs probabilities, and we
score the same thing we tuned on.

**Q5. Why does XGBoost not need a StandardScaler but the SVM does?**
Trees split on thresholds of single features. An increasing transform keeps
the order of values, so the same partitions exist and the tree is
unchanged. We verified it: raw vs standardised inputs gave valid
probabilities identical to the last digit. The SVM uses Euclidean distances,
which depend on each feature's units.

**Q6. Why early-stop on mlogloss but choose the setting by clip macro-F1?**
mlogloss is smooth and changes each round, so 50 rounds without improvement
is a reliable stopping signal. Clip macro-F1 on 197 clips moves in steps of
a few thousandths, which would trigger on noise. But clip macro-F1 is our
metric, so it decides between settings, as for every other model.

**Q7. What if you set learning_rate = 0.01?**
Each tree would add a tenth of the current step, so early stopping would
need roughly 5 to 10 times more trees (our data: halving it doubled the tree
count), still well under 2000 at depth 3. Valid mlogloss might improve
slightly (smaller steps regularise), but on our evidence the clip macro-F1
differences between settings are within noise. Training would take longer.

**Q8. What if you had used the test part as the `eval_set`?**
The number of trees would be chosen to minimise test loss, so the test
score would no longer be an unbiased estimate; it would be the best of up to
2000 candidate models evaluated on test. Here it cannot happen: `load_split`
raises before the `pre-test-freeze` tag exists.

**Q9. The SVM's best cell is on the edge of the grid. What would you do?**
Report it (we did, in `DECISIONS.md`) and not widen the grid now, because
expanding the grid in the direction valid looks good is manual tuning on
valid. If we wanted a wider search, we would fix the new grid in config
before running it, ideally justified by something other than the valid
score (e.g. "gamma = 0.0001 makes the kernel almost constant, see the
kernel table"). The gap between the two gamma = 0.001 cells (0.0016)
suggests little to gain.

**Q10. Is "boosting is gradient descent in function space" just a
metaphor?**
No. With loss $L(F)$, the negative gradient at each training point,
$-\partial L/\partial F(x_i)$, says how each prediction should move. A tree
fit to those values is the best direction available among trees; adding
$\eta$ times it is a gradient step with step size $\eta$. For softmax
cross-entropy the gradient is $p_k - \mathbb{1}[y = k]$, the same
"predicted minus true" as in logistic regression's gradient. XGBoost adds
second derivatives (a Newton step) and a penalty on leaf values.

**Q11. What if we ran these models on split A instead?**
Sibling windows of the same clip would be in train and valid. Large C and
large gamma would look best, because recognising a clip from its siblings
would score well (the SVM at C = 100, gamma = 0.1 already fits 100% of train
windows). We would expect valid scores to rise, and that rise would be
leakage, not genre skill (not yet measured for these models). The split ladder (lesson 04) measures exactly this later.

**Q12. Logistic regression beats both. Was this step wasted?**
No. A negative result with a clear reason is a finding: on these 57
handcrafted features and an artist-disjoint split, extra boundary
flexibility does not help, and the tuning plots show why (the best SVM is
the most linear one; XGBoost's valid loss turns up after 54 rounds while
train loss keeps falling). It also sets the bar for the CNN, which changes
the *features*, not just the classifier.

---

## 8. Further reading

- Boser, Guyon & Vapnik (1992), *A training algorithm for optimal margin
  classifiers*, COLT (kernel trick for SVMs). https://doi.org/10.1145/130385.130401
- Cortes & Vapnik (1995), *Support-vector networks*, Machine Learning 20
  (soft margin). https://doi.org/10.1007/BF00994018
- Platt (1999), *Probabilistic outputs for support vector machines and
  comparisons to regularized likelihood methods*, in Advances in Large
  Margin Classifiers. https://www.researchgate.net/publication/2594015
- Wu, Lin & Weng (2004), *Probability estimates for multi-class
  classification by pairwise coupling*, JMLR 5. https://www.jmlr.org/papers/v5/wu04a.html
- Hsu, Chang & Lin, *A practical guide to support vector classification*
  (scaling, RBF default, log grid over C and gamma).
  https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf
- Chang & Lin (2011), *LIBSVM: a library for support vector machines*, ACM
  TIST 2(3) (what sklearn's `SVC` runs). https://doi.org/10.1145/1961189.1961199
- Breiman, Friedman, Olshen & Stone (1984), *Classification and Regression
  Trees* (CART, Gini impurity).
- Friedman (2001), *Greedy function approximation: a gradient boosting
  machine*, Annals of Statistics 29(5). https://doi.org/10.1214/aos/1013203451
- Friedman (2002), *Stochastic gradient boosting*, Computational Statistics
  & Data Analysis 38(4). https://doi.org/10.1016/S0167-9473(01)00065-2
- Chen & Guestrin (2016), *XGBoost: a scalable tree boosting system*, KDD.
  https://arxiv.org/abs/1603.02754
- Prechelt (1998), *Early stopping: but when?*, in Neural Networks: Tricks of
  the Trade. https://doi.org/10.1007/3-540-49430-8_3
- Hastie, Tibshirani & Friedman (2009), *The Elements of Statistical
  Learning*, 2nd ed., ch. 9.2 (trees), ch. 10 (boosting), ch. 12 (SVMs).
  https://hastie.su.domains/ElemStatLearn/
- scikit-learn user guide: *Support Vector Machines*
  https://scikit-learn.org/stable/modules/svm.html ; *RBF SVM parameters*
  example https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html ;
  *Probability calibration* https://scikit-learn.org/stable/modules/calibration.html
- XGBoost docs: *Introduction to boosted trees*
  https://xgboost.readthedocs.io/en/stable/tutorials/model.html ; *Parameters*
  https://xgboost.readthedocs.io/en/stable/parameter.html ; *Early stopping*
  (Python API) https://xgboost.readthedocs.io/en/stable/python/python_intro.html#early-stopping
- Grinsztajn, Oyallon & Varoquaux (2022), *Why do tree-based models still
  outperform deep learning on typical tabular data?*, NeurIPS Datasets and
  Benchmarks. https://arxiv.org/abs/2207.08815
