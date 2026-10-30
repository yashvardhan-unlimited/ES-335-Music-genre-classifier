# Lesson 11: Validation analyses: learning curves, ablations, calibration, voting

By now every model has a tuned valid score. This lesson asks *why* the scores
are what they are, using five controlled experiments, all on **split C train
and valid only**:

1. **Window length**: is 3 s the right window?
2. **Learning curves**: would more training artists help, and does that
   tell us whether the models are limited by bias or by variance?
3. **Feature-group ablation**: which of the 57 handcrafted features does the
   SVM actually need?
4. **Calibration**: when a model says 80%, is it right 80% of the time? Can
   temperature scaling fix the CNN?
5. **Soft vs majority vote**: does the aggregation rule matter?

At the end, two design decisions are fixed and recorded in `DECISIONS.md`:
keep **3 s windows** and keep the **soft vote**.

---

## 1. What we built

| File | What it does |
|---|---|
| `scripts/analysis_val.py` | The five analyses as `part_window_length`, `part_learning_curves`, `part_feature_groups`, `part_calibration`, `part_voting`; `--parts` selects which run. |
| `src/features.py` | `windows(y, n_windows)` now cuts any number of equal windows; `clip_handcrafted(y, counts)` computes the 57 features for several window counts sharing one HPSS and tempo pass (the main 3 s features go through the same function); `feature_group(name)` maps a feature to its group (config `analysis.feature_groups`). |
| `src/evaluate.py` | `reliability` (per-bin confidence and accuracy), `ece`, `softmax_t` (softmax with temperature), `fit_temperature` (one-parameter NLL minimisation). |
| `src/splits.py` | `artist_groups` (artist per clip; a clip with no artist is its own group) and `sample_artists` (all clips of a random share of artist groups). |
| `config.yaml` (`analysis`) | Window lengths 1/3/5/10 s, artist fractions 0.25-1.0, 3 repeats, 10 calibration bins, the feature groups. |
| `tests/test_analysis.py` | Windows of every length tile the clip; every feature is in exactly one group; ECE by hand; temperature recovers a known over-confidence factor without changing predictions; artist sampling keeps whole artists. |
| `features/ablation/` | Handcrafted features at 1/3/5/10 s for the 639 train+valid clips (not committed). |
| `results/val/analysis_*.json` | `window_length`, `learning_curves`, `feature_groups`, `calibration`, `voting`. |
| `results/figures/analysis_*.png` | One figure per analysis. |

```bash
.venv/Scripts/python -W ignore scripts/analysis_val.py              # all five parts
.venv/Scripts/python -W ignore scripts/analysis_val.py --parts 1,3  # just some
.venv/Scripts/python -m pytest -q tests/test_analysis.py
```

Runtimes on this laptop's CPU: part 1 about 5 min (4 min of it extracting
features for 639 clips at four window lengths, done once), part 3 about
40 s, part 4 about 50 s, part 5 about 1 s, part 2 about 99 min (12 CNN
trainings). Parts 2 and 4 need the CNN results of lesson 10.

---

## 2. Concepts in depth

### 2.1 Why all of this uses validation, never test

Every analysis here can change a decision: the window length, the
aggregation rule, which features to keep, whether to calibrate. Any number
that can change a decision is part of model selection, and model selection
must happen on validation. If we looked at test to choose the window length,
the test score of the chosen length would be the best of four, an optimistic
number (lesson 09's selection bias), and the test set would no longer be an
independent check. Test is used once, after everything is frozen (the
`pre-test-freeze` tag, lesson 12). The code enforces it: every loader goes
through `load_split`, which refuses the test part.

The price: valid has only 197 clips (13 to 24 per genre), so one clip
changes macro-F1 by very roughly 0.003 to 0.005. Differences of 0.01 to 0.02
in this lesson are a few clips and must be read as "no clear difference",
not as rankings.

### 2.2 Ablation as a controlled experiment

An **ablation** removes or changes one component and measures the effect,
holding everything else fixed. It is the machine-learning version of a
controlled experiment: if only one thing changed, the change in score is
caused by that thing (up to noise).

What "everything else fixed" means here:

- same train and valid clips (split C);
- same model and **same hyperparameters** (SVM C = 100, gamma = 0.001, from
  lesson 08), not re-tuned per variant;
- same metric (clip macro-F1, soft vote).

Holding hyperparameters fixed isolates the effect of the ablated component,
but it has a cost we must state: the parameters were tuned for the full
setup (3 s windows, all 57 features), so the full setup has a small built-in
advantage, and a variant might do better with its own tuning. Re-tuning per
variant would answer a different question ("best achievable with X") and
would multiply the selection on valid.

### 2.3 Window length

**The trade-off.** Shorter windows give more windows per clip (30 at 1 s)
and more training rows (13,260), but each window's features are averages
over fewer frames, so they are noisier and may miss slower patterns
(a 1 s window may hold less than one bar of music). Longer windows give
stabler statistics but fewer votes per clip (3 at 10 s) and fewer training
rows (1,326).

**Our numbers** (`results/val/analysis_window_length.json`,
`results/figures/analysis_window_length.png`):

| window | windows per clip | train rows | valid clip macro-F1 (soft) | majority | window accuracy |
|---|---|---|---|---|---|
| 1 s | 30 | 13,260 | 0.435 | 0.449 | 0.424 |
| 3 s | 10 | 4,420 | 0.458 | 0.455 | 0.419 |
| 5 s | 6 | 2,652 | **0.477** | 0.466 | 0.443 |
| 10 s | 3 | 1,326 | 0.449 | 0.468 | 0.467 |

The recomputed 3 s features are identical to the main features (maximum
relative difference 0.0), so the 3 s row reproduces lesson 08's 0.458.

**Reading.** Window accuracy rises steadily with length (0.424 -> 0.467):
a longer window is individually easier to classify. Clip macro-F1 does not
follow, because longer windows also mean fewer votes per clip. The spread
over all four lengths is 0.042, a handful of clips. 5 s is highest, but by
0.019, with parameters tuned at 3 s, chosen after seeing the valid scores.
**Decision: keep 3 s** (DECISIONS.md). Switching would be selecting on
noise, and it would change every model's input.

### 2.4 Feature-group ablation

Drop one group at a time from the SVM's 57 features, retrain, re-score
(`results/val/analysis_feature_groups.json`, full model 0.458):

| dropped group | features | valid clip macro-F1 | change |
|---|---|---|---|
| MFCC | 40 | 0.416 | **-0.042** |
| HPSS (harmonic/percussive RMS) | 4 | 0.449 | -0.009 |
| chroma | 2 | 0.458 | +0.000 |
| tempo | 1 | 0.464 | +0.006 |
| ZCR + RMS | 4 | 0.468 | +0.009 |
| spectral shape (centroid, bandwidth, roll-off) | 6 | 0.471 | +0.013 |

**Reading.**

- Only the **MFCCs** clearly matter: dropping them costs 0.042. They are
  40 of the 57 features and describe the overall spectral envelope (timbre)
  compactly (lesson 05).
- Every other change is within +/-0.013, i.e. a few clips: no evidence that
  those groups help *or* hurt. A positive change ("the model got better
  without spectral shape") is not a reason to drop them: it is within noise,
  and centroid, bandwidth and roll-off are largely redundant with the low
  MFCCs, so the SVM loses little when they go.
- A subtle effect of fixed hyperparameters: the RBF kernel uses
  $\exp(-\gamma\lVert x-x'\rVert^2)$ summed over all features. Dropping 40
  MFCCs shrinks typical squared distances, which acts like a smaller
  gamma; dropping 2 chroma features barely changes them. So the MFCC row
  mixes "losing information" with "changing the kernel width".

**Ablation vs permutation importance.** Lesson 15 uses a different tool on
the same model: shuffle a group's values on valid *without retraining*.
Shuffling the MFCCs costs 0.208, five times the ablation's 0.042
(`results/val/permutation_importance.json`). The difference is the point:
permutation importance asks "how much does *this trained model* rely on the
group?", ablation asks "how much would *a model trained without it* lose?".
A retrained SVM leans on the remaining, partly redundant features; a fixed
model whose MFCC inputs are scrambled cannot.

### 2.5 Learning curves and the bias-variance diagnosis

A **learning curve** plots the validation score against the amount of
training data. Its shape is a diagnosis:

- **Still rising at 100%**: the model is limited by data (variance). More
  training data would help; so might stronger regularisation or
  augmentation, which make the most of the data you have.
- **Flat for the last steps**: more of the *same kind* of data will not
  help. If the score is also far below what you need, the model is limited
  by bias (its form, or its features) and needs to change, not grow.
- **Gap to the training score**: a large gap with a rising valid curve is
  variance; a small gap with a low plateau is bias.

**Why artists, not clips.** We subsample **artists** (all clips of a random
share of the 146 artist groups in C train; a clip without a known artist is
its own group), because split C's valid artists are new. Subsampling clips
would keep every training artist and only thin out their songs, which
answers a different question ("more songs by the same artists") and would
overstate how well small training sets generalise to new artists. Because
artists have very different numbers of clips (up to 26), the number of
training clips varies between repeats at the same fraction; it is recorded
and its mean is shown on the x axis.

The CNN is trained for a fixed **13 epochs** (the median best epoch from
lesson 10) with no early stopping, then scored once on valid, so valid does
not choose anything inside the learning curve. With fewer clips, 13 epochs
also mean fewer gradient steps; we keep epochs, not steps, fixed, as
specified, and note it as a caveat.

**Result** (`results/val/analysis_learning_curves.json`,
`results/figures/analysis_learning_curves.png`). Valid clip macro-F1, mean
+/- sd over the 3 artist subsets (seeds 42, 43, 44); the SVM keeps the
hyperparameters tuned on the full train part.

| artists | training clips per repeat | SVM | CNN (13 epochs) |
|---|---|---|---|
| 25% (36 of 146) | 91 / 124 / 86 | 0.304 +/- 0.060 | 0.233 +/- 0.027 |
| 50% (73) | 225 / 223 / 173 | 0.427 +/- 0.043 | 0.443 +/- 0.092 |
| 75% (110) | 290 / 360 / 279 | 0.444 +/- 0.008 | 0.468 +/- 0.038 |
| 100% (146) | 442 | 0.458 +/- 0.000 | 0.528 +/- 0.071 |

How to read it:

- **SVM: flattening.** +0.123 from 25% to 50%, then only +0.017 and
  +0.014. More artists of the kind we already have barely help the SVM. Its
  limit is the 57 handcrafted features (bias), not the amount of data. At
  100% all three repeats use the same clips and the SVM is deterministic,
  hence sd 0.
- **CNN: still rising.** It starts *below* the SVM at 25% (0.233 vs 0.304):
  241,898 parameters learned from about 100 clips, and 13 epochs over a
  quarter of the data are also a quarter of the gradient steps. It overtakes
  from 50% and gains +0.060 from 75% to 100%. A curve still rising at the
  right edge says the CNN is data-limited (variance): more training artists
  would very likely help it.
- **Noise.** The CNN's sd is 0.04 to 0.09 per point, so the steps between
  50%, 75% and 100% are each within about one sd. Only the overall trend and
  the jump from 25% are clear. Two of the three 25% subsets (124 and 86 clips)
  contain no clips of one genre, so that genre's F1 is 0 and macro-F1 loses
  up to 0.1 for that reason alone.
- **Why 0.528 at 100% and not lesson 10's 0.610?** Same 442 clips, same
  torch seeds 42 to 44, but (1) no early stopping: we score epoch 13, not the
  best epoch picked on valid, so there is no optimistic selection; and (2) a
  different random stream. In lesson 10 every epoch ends with a valid pass
  through a `DataLoader`, and creating a `DataLoader` iterator draws a number
  from torch's global RNG even with `shuffle=False`. We checked this: one
  pass over a tiny `DataLoader` changes the next `torch.rand` value. So from
  epoch 2 on, crops and dropout masks differ, and the two trainings diverge.
  Seed 44 shows how much this matters: 0.635 there (best epoch 13), 0.446 here
  (fixed epoch 13). For the final refit with a fixed epoch count this means
  one refit is one draw from a wide distribution.

### 2.6 Calibration and reliability diagrams

A classifier is **calibrated** if, among all predictions made with
confidence $p$, a fraction $p$ is correct. Confidence here is the largest
class probability. A **reliability diagram** groups predictions into bins of
confidence (10 equal-width bins on [0, 1]) and plots, per bin, mean
confidence (x) against accuracy (y). A calibrated model lies on the
diagonal; points **below** it mean **over-confidence** (the model says 80%
and is right 60% of the time), points above it mean under-confidence.

The **expected calibration error** summarises the diagram:

```latex
\text{ECE} = \sum_{b=1}^{B} \frac{n_b}{n}\, \bigl|\text{acc}(b) - \text{conf}(b)\bigr| ,
```

with $n_b$ predictions in bin $b$, $\text{acc}(b)$ their accuracy and
$\text{conf}(b)$ their mean confidence. Worked by hand (our unit test): two
predictions at confidence 0.95 with one correct, two at 0.65 both correct:
$\text{ECE} = \tfrac12|0.5-0.95| + \tfrac12|1.0-0.65| = 0.225 + 0.175 =
0.40$.

ECE has limits: it depends on the binning, empty or tiny bins are noisy
(our clip-level diagram has 197 clips over 10 bins, some with 2 clips), and
it only looks at the top class. It is a summary for comparison, not a
precise measurement.

We compute it at two levels: **window** (1,970 valid windows, each
window's probabilities) and **clip** (197 clips, the soft-vote mean
probabilities: the prediction we actually make).

### 2.7 Temperature scaling, and why it does not change accuracy

A neural network's logits $z$ can be rescaled by one number $T > 0$ before
the softmax (Guo et al. 2017):

```latex
p_k(x; T) = \frac{\exp(z_k / T)}{\sum_j \exp(z_j / T)} .
```

$T > 1$ flattens the probabilities (less confident), $T < 1$ sharpens them.
$T$ is fit on held-out data by minimising cross-entropy, with the network's
weights frozen; it is a single parameter, so it cannot overfit much.

**Why it does not change the predicted class.** Dividing all logits by the
same positive $T$ keeps their order: if $z_a > z_b$ then $z_a/T > z_b/T$,
and the softmax is increasing in each logit, so the arg-max is unchanged.
Window predictions, window accuracy and window F1 are identical before and
after. (Measured: 0 changed window predictions for all three seeds.)

**But clip predictions can change.** Our clip prediction is the arg-max of
the *mean* of 10 window probability vectors. Temperature reshapes each
window's vector non-linearly (it moves confident windows more than unsure
ones, and with 10 classes it redistributes mass among the losing classes
too), so the order of the *averaged* probabilities can change even though no
single window's arg-max does. We measured it rather than assume it: seed 2's
clip macro-F1 goes from 0.635 to 0.623 with its fitted temperature, with 0
window predictions changed.

**Fitting on valid, honestly.** Fitting $T$ on valid and measuring ECE on the
same valid data is in-sample. We also report a **cross-fitted** version: valid
artists are split into two random halves, $T$ fit on one half's windows is
applied to the other half's, and vice versa.

### 2.8 Soft vs majority vote

Soft vote: average the 10 window probability vectors, take the arg-max.
Majority vote: each window votes for its arg-max class; ties broken by the
mean probability (lesson 06). Soft voting uses confidence (an unsure window
counts less) and has no ties; majority voting is robust to one wildly
over-confident window. Which wins is an empirical question.

---

## 3. Why this choice, and the alternatives we rejected

| Choice | Alternative | Why |
|---|---|---|
| All analyses on valid | On test | Each can change a decision; test must stay independent (2.1). |
| Hyperparameters fixed in ablations | Re-tune per variant | Isolates the component; re-tuning answers "best achievable" and multiplies selection on valid. Stated as a caveat. |
| Artist-level subsampling | Clip-level subsampling | Valid artists are new; thinning out songs of the same artists would not measure what more *artists* bring. |
| CNN with fixed 13 epochs in the learning curve | Early stopping per subset | Early stopping would let valid choose inside each point and make the curve optimistic. |
| Equal-width 10-bin ECE, clip and window | Adaptive bins; Brier score | Standard and readable with the diagram; the Brier score also mixes in accuracy. Both levels shown because they disagree (2.7, 6). |
| Cross-fitted temperature | In-sample only | In-sample ECE after fitting on the same data is optimistic. |
| Keep 3 s, keep soft vote | Switch to 5 s / per-model choice | Within noise and post-hoc; see DECISIONS.md. |

---

## 4. Code walkthrough

### `src/features.py`

- **L27–34 `windows(y, n_windows)`**: `y.reshape(n, 661500 // n)` after
  asserting that `n` divides the clip (1, 3, 5, 10 s give 30, 10, 6, 3).
- **L51–58 `feature_group`**: strips `_mean`/`_var` and trailing digits
  (`mfcc12_var` -> `mfcc`) and looks the stem up in config; raises if it is
  in zero or several groups (tested: 40 + 2 + 6 + 4 + 4 + 1 = 57).
- **L94–101 `clip_handcrafted`**: one HPSS and one tempo per clip, then the
  57 features for each requested window count. **L104–107 `clip_features`**
  (the main pipeline) calls it with 10, so the ablation's 3 s features and
  the main features come from the same code.

### `src/evaluate.py`

- **L149–162 `reliability`**: confidence = max probability; bins
  $(0, 0.1], \dots, (0.9, 1]$ via `np.digitize(..., right=True)`; per bin:
  count, mean confidence, accuracy.
- **L165–169 `ece`**: the formula of 2.6 over non-empty bins.
- **L172–175 `softmax_t`**: `scipy.special.softmax(logits / T)`.
- **L178–187 `fit_temperature`**: minimises mean cross-entropy over
  $\log T \in [-3, 3]$ (so $T \in [0.05, 20]$) with a bounded scalar search;
  optimising $\log T$ keeps $T$ positive.

### `src/splits.py`

- **L122–126 `artist_groups`**, **L129–135 `sample_artists`**: draw
  `round(fraction x n_groups)` groups with `np.random.default_rng(seed)` from
  the sorted group list (so the draw does not depend on clip order), return
  their clips in input order.

### `scripts/analysis_val.py`

- **L64–68 `svm_probs`**: the tuned SVM refit on the given rows with one BLAS
  thread, as in tuning.
- **L91–106 `extract_ablation`**: runs `clip_handcrafted` for all four window
  counts in parallel over the 639 train+valid clips (no test clips) and
  saves one array plus index per length; skipped if already present.
- **L117–159 `part_window_length`**: checks the 3 s ablation features against
  the main features, then fits and scores the SVM per length.
- **L164–175 `cnn_valid_logits`, `window_labels`**: CNN logits on the 10 fixed
  windows of every valid clip, and the matching per-window labels.
- **L178–228 `part_learning_curves`**: for each fraction and repeat (seed
  42 + repeat): sample artists; SVM on their windows; CNN via
  `train_cnn.fit(..., valid=None, epochs=13)` (normalisation stats from the
  subset), then valid logits and `valid_report`.
- **L232–264 `part_feature_groups`**: full SVM, then one refit per dropped
  group.
- **L268–277 `calibration_entry`**: ECE and reliability tables at window and
  clip level, and clip macro-F1.
- **L280–291 `cross_fit_temperature`**: two halves of valid artists; $T$ from
  one half applied to the other.
- **L294–352 `part_calibration`**: SVM and XGBoost refit exactly as tuned
  (asserts the XGBoost refit reproduces its valid score), CNN checkpoints of
  the three seeds (asserts each reproduces its valid score), temperature
  in-sample and cross-fitted.
- **L356–384 `part_voting`**: reads `valid_macro_f1` and
  `valid_macro_f1_majority` from every model's result file.

---

## 5. Pitfalls

1. **Choosing the window length (or anything else) on test.** It would turn
   test into a second validation set (2.1).
2. **Reading +/-0.01 as a ranking.** On 197 clips that is a few clips.
3. **Dropping features because the ablation went up by 0.013.** Within
   noise, and a decision made on valid noise does not transfer.
4. **Subsampling clips instead of artists** for a learning curve on an
   artist-disjoint split: it overstates what small data achieves on new
   artists.
5. **Early-stopping inside the learning curve on valid**: each point would
   be the best of many epochs on valid; the curve would be optimistic.
6. **Calibrating windows and assuming clips are calibrated.** Soft voting
   already softens confidence; a window-level temperature can make clip
   probabilities under-confident (our seed 0: clip ECE 0.049 -> 0.077).
7. **Reporting in-sample ECE after temperature scaling** as if it were
   held-out.
8. **Saying "temperature scaling cannot change accuracy"** without the
   qualifier: true for each window, not for a soft-voted clip.
9. **Loading checkpoints with arbitrary pickled objects.** Our first CNN
   checkpoints stored the class names as numpy strings, which PyTorch's
   default safe loader (`weights_only=True`) refuses; they are now saved as
   plain strings.

---

## 6. Results from this step

**Window length**: see 2.3. Keep 3 s.

**Feature groups**: see 2.4. Only MFCCs clearly matter (-0.042).

**Learning curves**: see 2.5. SVM flattens (0.304, 0.427, 0.444,
0.458 at 25/50/75/100% of artists); CNN keeps rising (0.233, 0.443, 0.468,
0.528) and overtakes the SVM from 50%. SVM: feature-limited. CNN:
data-limited. Part 2 took 5,953 s, 5,925 s of it in the 12 CNN trainings
(2.7 to 13.8 min each, while the CPU was shared with the CNN tuning runs).

**Calibration** (`results/val/analysis_calibration.json`,
`results/figures/analysis_calibration.png`; ECE with 10 bins):

| model | window ECE | clip ECE | clip macro-F1 |
|---|---|---|---|
| SVM (Platt) | 0.267 | 0.112 | 0.458 |
| XGBoost | 0.110 | 0.105 | 0.419 |
| CNN seed 0 (T = 1.37) | 0.104 -> 0.033 / 0.029 | 0.049 -> 0.077 / 0.078 | 0.580 -> 0.580 / 0.580 |
| CNN seed 1 (T = 2.01) | 0.196 -> 0.038 / 0.031 | 0.100 -> 0.068 / 0.080 | 0.615 -> 0.614 / 0.614 |
| CNN seed 2 (T = 1.75) | 0.154 -> 0.034 / 0.032 | 0.077 -> 0.097 / 0.114 | 0.635 -> 0.623 / 0.627 |

(CNN: before -> after in-sample / after cross-fitted. Cross-fitted
temperatures: seed 0 1.34 and 1.41, seed 1 1.91 and 2.13, seed 2 1.50 and
2.03.)

- **All three CNNs are over-confident per window** (every fitted $T > 1$).
  Temperature scaling cuts window ECE by a factor of 3 to 6, and the
  cross-fitted numbers are as good as the in-sample ones, so this is not an
  artefact of fitting on the same data. This is the over-confidence behind
  the valid-loss spikes in lesson 10's curves.
- **At clip level the picture is mixed.** Soft voting averages 10
  over-confident windows, which already pulls clip probabilities towards
  the middle. Applying the window temperature as well over-corrects for
  seeds 0 and 2 (clip ECE rises: 0.049 -> 0.077, 0.077 -> 0.097) and helps
  seed 1 (0.100 -> 0.068). In the diagram, CNN seed 0 after $T$ sits above
  the diagonal (under-confident).
- **The SVM is the worst calibrated per window** (0.267): Platt's sigmoid is
  fit on internal CV folds over windows (lesson 08), and its clip-level
  diagram lies below the diagonal (over-confident).
- **XGBoost** is S-shaped: over-confident at low confidence, under-confident
  at high confidence.
- **Accuracy**: 0 window predictions changed by $T$ for any seed; clip
  macro-F1 changed by 0.000, -0.001 and -0.012 (in-sample) through the
  soft vote.

**Conclusion for the project:** we report probabilities as scores, not as
calibrated confidences, and do **not** apply temperature scaling to the
reported CNN: it does not improve the clip-level probabilities we actually
use, and it can only change clip predictions through the soft vote.

**Voting** (`results/val/analysis_voting.json`,
`results/figures/analysis_voting.png`):

| model | soft | majority | soft - majority |
|---|---|---|---|
| dummy (most frequent) | 0.018 | 0.018 | +0.000 |
| dummy (stratified) | 0.128 | 0.128 | +0.000 |
| kNN | 0.424 | 0.424 | -0.000 |
| logistic regression | 0.468 | 0.468 | +0.001 |
| SVM | 0.458 | 0.455 | +0.003 |
| XGBoost | 0.419 | 0.414 | +0.005 |
| CNN seed 0 | 0.580 | 0.568 | +0.013 |
| CNN seed 1 | 0.615 | 0.603 | +0.011 |
| CNN seed 2 | 0.635 | 0.642 | -0.008 |

Soft >= majority for 8 of 9 results; all differences within 0.013.
**Keep the soft vote** (DECISIONS.md).

---

## 7. Viva questions with model answers

**Q1. Why are none of these analyses run on test?**
Each could change a decision (window length, vote, features, calibration).
Anything that changes a decision is model selection, and selecting on test
would make the test score optimistic and no longer independent. The code
refuses the test part before the `pre-test-freeze` tag anyway.

**Q2. Your 5 s windows scored 0.477 vs 0.458 for 3 s. Why not switch?**
0.019 is a handful of the 197 valid clips; the SVM's parameters were tuned at
3 s; and choosing 5 s *after* seeing valid scores is selection on noise.
Switching would also change every other model's input and invalidate their
tuning. We record it as a finding, not a decision.

**Q3. What does a still-rising learning curve mean? A flat one?**
Rising at 100%: limited by data (variance); more training artists would
help, as would regularisation that makes better use of the data. Flat:
more of the same data will not help; if the level is too low, the model or
features must change (bias). Ours: the SVM flattens (+0.014 from 75% to 100%), so it
needs better features, not more artists; the CNN still gains +0.060 at the
end, so it would benefit from more data or stronger augmentation.

**Q4. Why subsample artists, not clips?**
Valid artists are unseen. Removing clips but keeping all artists would test
"more songs by known artists", which is not what our split measures, and
would overstate small-data performance on new artists.

**Q5. Explain ECE and compute it for a tiny example.**
Bin predictions by confidence; ECE = sum over bins of (bin share) x |bin
accuracy - bin confidence|. Two predictions at 0.95 with one right and two
at 0.65 both right: 0.5 x 0.45 + 0.5 x 0.35 = 0.40.

**Q6. Why does temperature scaling not change accuracy? Did it change your
clip F1?**
Dividing all logits by the same $T > 0$ keeps their order, so each window's
arg-max is unchanged (0 changes measured). But our clip prediction is the
arg-max of the *average* of window probabilities, which can change; seed 2's
clip macro-F1 moved from 0.635 to 0.623.

**Q7. What if you applied temperature scaling to the final CNN?**
Window probabilities would be much better calibrated (ECE 0.03 instead of
0.10 to 0.20), but clip probabilities would become under-confident for two of
three seeds, and clip predictions could shift slightly. Since we report clip
macro-F1 and do not use probabilities as decision thresholds, we do not
apply it.

**Q8. Your feature ablation and your permutation importance disagree about
MFCCs (0.042 vs 0.208). Which is right?**
Both, for different questions. Permutation importance: how much this trained
SVM relies on MFCCs (a lot). Ablation: how much a model retrained without
them loses (less), because other features are partly redundant with MFCCs
and the retrained model leans on them instead.

**Q9. What if you re-tuned the SVM for each window length?**
Each length would get its best C and gamma, which might change the ranking.
It would answer "best achievable per length" at the cost of 4 x 16 more
selections on valid and more selection bias; with differences already within
noise, it would not justify switching.

---

## 8. Further reading

- Guo, Pleiss, Sun & Weinberger (2017), *On calibration of modern neural
  networks*, ICML (temperature scaling, ECE). https://arxiv.org/abs/1706.04599
- Naeini, Cooper & Hauskrecht (2015), *Obtaining well calibrated
  probabilities using Bayesian binning*, AAAI (ECE).
  https://ojs.aaai.org/index.php/AAAI/article/view/9602
- Niculescu-Mizil & Caruana (2005), *Predicting good probabilities with
  supervised learning*, ICML (reliability diagrams, Platt scaling for SVMs
  and boosted trees). https://doi.org/10.1145/1102351.1102430
- Nixon et al. (2019), *Measuring calibration in deep learning*, CVPR
  workshops (limits of ECE, binning). https://arxiv.org/abs/1904.01685
- Perlich, Provost & Simonoff (2003), *Tree induction vs. logistic
  regression: a learning-curve analysis*, JMLR 4.
  https://www.jmlr.org/papers/v4/perlich03a.html
- Hastie, Tibshirani & Friedman (2009), *The Elements of Statistical
  Learning*, ch. 7 (bias-variance, model assessment and selection).
  https://hastie.su.domains/ElemStatLearn/
- Meyes, Lu, de Puiseau & Meisen (2019), *Ablation studies in artificial
  neural networks*. https://arxiv.org/abs/1901.08644
- scikit-learn user guide: *Probability calibration*
  https://scikit-learn.org/stable/modules/calibration.html ; *Permutation
  feature importance* https://scikit-learn.org/stable/modules/permutation_importance.html ;
  *Validation curves and learning curves*
  https://scikit-learn.org/stable/modules/learning_curve.html
