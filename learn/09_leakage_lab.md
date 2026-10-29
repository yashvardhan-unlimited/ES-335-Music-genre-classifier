# Lesson 09: Leakage lab: three deliberate mistakes and what they cost

Lessons 04 and 07 explained leakage in words. This lesson **measures** it.
We take the RBF SVM from lesson 08 and tune it four times on split C:
once correctly, and three times with a classic mistake built in. Each
mistaken version changes **exactly one thing** from the correct setup, so
any difference in the result is caused by that one mistake.

For every version we record three things:

- the **chosen hyperparameters** (C, gamma);
- the **reported score**: the number the procedure itself would print and a
  careless report would quote;
- the **true score**: the honest valid clip macro-F1 of those chosen
  hyperparameters (scaler fit on train only, model fit on train, soft vote
  over each valid clip's 10 windows).

Only split C train (442 clips, 4,420 windows) and valid (197 clips, 1,970
windows) are used. The test part was never loaded.

Headline: **5-fold cross-validation over windows reports 0.973 for a setting
whose real valid score is 0.433.** The other two mistakes are quieter but
still wrong, for reasons this lesson takes apart.

---

## 1. What we built

| File | What it does |
|---|---|
| `scripts/leakage_lab.py` | Runs the correct SVM grid search and the three mistaken ones, looks up the true valid clip macro-F1 of every setting, computes diagnostics, writes the JSON and the figure, prints a table. |
| `results/val/leakage_lab.json` | Per variant: description, selection metric, chosen params, reported score, true score, Spearman rank correlation between selection score and true score over the 16 settings, and the full grid (selection score and true score per setting). Plus `diagnostics` (how far the leaky scaler moved, whether an inner scaler cancels the leak, share of held-out windows with a sibling in training). |
| `results/figures/leakage_lab.png` | Reported (blue) vs true (orange) score per variant, chosen params under each pair, dashed line at the correct setup's true score. |

```bash
.venv/Scripts/python -W ignore scripts/leakage_lab.py    # about 2.5 min
```

It reuses `load_windows` and `param_grid` from `src/train_classical.py`,
`svm_pipeline` from `src/models.py`, and `ClipMacroF1Scorer` /
`make_predefined_split` from `src/evaluate.py`, so the correct variant is
literally the same code path as lesson 08. A built-in assert checks that
its 16 scores equal the grid in `results/val/C_s0_svm.json`. Two complete
runs gave identical JSON.

---

## 2. Concepts in depth

### 2.1 What "reported" and "true" mean here

A tuning procedure does two jobs: it **chooses** hyperparameters, and it
**produces a number** that people quote as "how good the model is". Leakage
can damage either job:

1. The number can be **wrong** (inflated, or just not measuring what it
   claims), even if the chosen setting is fine.
2. The choice can be **worse**, because the selection score rewards the
   wrong thing.

To see both, we need a ruler that is the same for every variant. Our ruler is
the correct procedure's valid score for each of the 16 settings: train-only
scaler, fit on train, soft vote over clips, clip macro-F1 on valid. Every
variant's chosen setting is looked up on this ruler. So:

```latex
\text{true}(\theta) = \text{macro-F1}_\text{clip}\bigl(\text{valid};\ \text{SVM}_\theta \text{ fit on train with train-only scaling}\bigr)
```

and the correct variant's choice is simply

```latex
\theta^\star = \arg\max_{\theta \in \text{grid}} \text{true}(\theta) .
```

**An important consequence.** Because the correct variant picks the arg-max
of exactly this ruler, its true score (0.458) is *by construction* at least
as high as any other variant's true score. The lab therefore cannot prove
that a mistaken procedure picks worse settings *on new data*; that would
need the test part. What it shows reliably is (a) how far each **reported
number** is from the truth, and (b) how well each selection score **ranks**
the 16 settings compared with the truth. Point (b) is measured by Spearman's
rank correlation (2.6).

### 2.2 Mistake 1: scaler fit on train+valid before the pipeline

**What the code does.** One `StandardScaler` is fit on all 6,390 train and
valid windows. Both parts are transformed with it. Then an SVM *without its
own scaler* (the Pipeline's `scale` step set to `"passthrough"`) is tuned
with the correct train -> valid split and clip macro-F1.

```latex
\mu_\text{leaky} = \frac{1}{n_\text{tr} + n_\text{va}} \sum_{i \in \text{tr} \cup \text{va}} x_i ,
\qquad
z_i = \frac{x_i - \mu_\text{leaky}}{\sigma_\text{leaky}} .
```

**The mechanism.** The training windows are now expressed in coordinates
that depend on where the valid windows are. If valid clips are, say, louder
on average, the "zero" of the RMS feature moves toward them, and the RBF
distances the SVM computes on train are measured in a frame partly set by
valid. It is unlabelled information (no genre labels leak), but it is
information about the evaluation data that a real system would not have:
when a new clip arrives, its features were not part of the scaler.

**How big is it here?** Small. Valid is 1,970 of 6,390 windows (31%), and
train and valid come from the same kind of audio, so the statistics barely
move:

- largest shift of a feature mean: 0.060 training standard deviations;
  median shift 0.024;
- ratio of leaky to train-only standard deviation: 0.937 to 1.203.

Result: the same setting is chosen (C = 100, gamma = 0.001), the 16 settings
are ranked almost identically (Spearman 0.962), and the reported score is
**0.452** against a true **0.458**. Notice the reported score is *lower*
here. Leakage does not promise an optimistic number in every run; it
promises a number produced by a procedure you cannot actually run on new
data. Its sign and size are luck.

**Why it is still wrong.**

1. **The number is not what it claims.** "Valid score" is supposed to mean
   "what happens on unseen clips". With mistake 1, valid was seen, just
   without labels.
2. **The habit scales badly.** The same pattern ("fit preprocessing on all
   the data, then split") is a serious leak for learned steps that use
   labels or many parameters: feature selection by correlation with the
   genre, PCA on everything, target encoding, oversampling (SMOTE) before
   splitting. A scaler is the gentlest case of a dangerous habit.
3. **It can be large when train and valid differ.** Our split is
   artist-disjoint but genre-mixed. With a split by time, recording studio,
   or country, the valid statistics could be far from train's, and the leaky
   scaler would quietly adapt the model to them.

**A subtle point: an inner scaler cancels it.** If you pre-scale globally
but *keep* the scaler inside the Pipeline, the inner scaler re-standardises
the training rows with train-only statistics. Standardisation is an affine
map ($z = a \odot x + b$), and an affine map followed by standardisation on
the same rows gives exactly the train-only standardisation. We checked it:
maximum probability difference 0.00055, 0 changed window predictions (the
tiny difference is floating-point rounding passing through the solver).
That is why the lab removes the inner scaler: otherwise "mistake 1" would
not be a mistake at all. In other people's code, the dangerous version is
"scale everything, then fit a bare model".

### 2.3 Mistake 2: KFold(5, shuffle=True) over train windows

**What the code does.** `GridSearchCV` with `KFold(n_splits=5,
shuffle=True, random_state=42)` on the **4,420 train windows only**. Each
fold holds out 884 random windows, trains on the other 3,536, and scores
the held-out windows by clip macro-F1 (soft vote over whichever windows of
each clip are held out). The setting with the best mean over 5 folds wins.
Then we look up its true valid score. The scorer and the pipeline are the
correct ones; only the cross-validation splitter is wrong.

**The mechanism: windows of one song on both sides.** Every clip has 10
windows. Shuffled KFold sends each window to a fold independently, so a
held-out window almost always has siblings from the same recording in the
training folds. On average 7.2 of its 9 siblings are in training (each sibling
is in training with probability 4/5, and 9 x 4/5 = 7.2). A held-out window with *no* sibling
in training would need all 10 windows of its clip in the same fold:

```latex
P(\text{no sibling in training}) = \left(\tfrac15\right)^{9} \approx 5 \times 10^{-7} \text{ per clip.}
```

Measured on our folds: **100%** of held-out windows have a sibling in the
training part of their fold.

Siblings are three-second slices of *the same recording*: the same
instruments, singer, mix, mastering, loudness, key and tempo. Their 57
features are much closer to each other than to any other clip (lesson 07:
81% of windows have a sibling as nearest neighbour). So the held-out window
is not a new example; it is a near-duplicate of training rows. The CV score
measures **recognising a recording**, not **recognising a genre**.

**Why the inflation is so large.** An RBF SVM with large C and large gamma
places a narrow bump on every training window and fits them all (lesson 08:
C = 100, gamma = 0.1 gets 100% of training windows right). Under shuffled
KFold that memorisation is *rewarded*, because every held-out window sits
right next to memorised siblings. The selection score rises with C and
gamma:

| C \ gamma | 0.001 | 0.01 | scale (0.0175) | 0.1 |
|---|---|---|---|---|
| 100 | 0.925 | 0.968 | **0.973** | 0.951 |
| 10 | 0.861 | 0.966 | 0.971 | 0.951 |
| 1 | 0.764 | 0.903 | 0.939 | 0.945 |
| 0.1 | 0.594 | 0.747 | 0.775 | 0.713 |

(5-fold CV clip macro-F1 over train windows; compare lesson 08's true valid
heatmap, where every cell is between 0.340 and 0.458.)

Result: chosen C = 100, gamma = scale; reported **0.973**; true **0.433**.
The reported number overstates the truth by 0.540, more than doubling it.
The ranking of settings is almost unrelated to the truth (Spearman 0.252).
The most striking cell: C = 0.1, gamma = 0.001 is **last** under KFold
(0.594) but truly one of the two best (0.457). A procedure that rewards
memorisation punishes exactly the smooth models that generalise.

**What would fix it.** Split so that a whole group is on one side: at least
`GroupKFold` with `groups=clip_id`, and for this project, groups by
**artist**, because clips of one artist also share production and style
(lesson 04). We use the simplest correct version: one predefined train ->
valid split that is already artist-disjoint. We did not run a GroupKFold
variant here.

### 2.4 Mistake 3: selecting by window accuracy instead of clip macro-F1

**What the code does.** The correct train -> valid split and the correct
pipeline, but `scoring="accuracy"`: the share of the 1,970 valid **windows**
predicted correctly. The setting with the highest window accuracy wins.

This is not leakage in the strict sense (no evaluation information reaches
training). It is a **selection-metric mismatch**: the procedure optimises one
quantity and the project reports another. It belongs in this lab because the
effect on the choice is the same kind: a selection score that does not
measure the target.

**Three differences between the two scores:**

1. **Unit.** Our prediction is per clip, made by averaging 10 window
   probabilities. A model whose windows are individually unsure but
   consistently lean the right way gets clips right through averaging;
   window accuracy does not see that.
2. **Decision rule.** For `SVC`, `scoring="accuracy"` calls `predict`,
   which uses the one-vs-one votes, not Platt's probabilities (lesson 08,
   2.7). So it scores a different classifier from the one we deploy.
3. **Class weighting.** Accuracy counts windows, so genres with more valid
   clips count more (valid has 24 rock clips but 13 pop clips). Macro-F1
   weights all 10 genres equally.

Result: chosen C = 10, gamma = 0.001 (window accuracy **0.452**), true clip
macro-F1 **0.440**, against 0.458 for the correct choice. Spearman between
the two scores over 16 settings is 0.620. The biggest disagreement: C = 0.1,
gamma = 0.1 has window accuracy 0.203 but clip macro-F1 0.340; voting
rescues many clips whose windows are individually wrong.

The reported number here (0.452) is not even on the same scale as the
others: it is a window accuracy, and quoting it as "the model's score"
would mix units. The figure's title says so.

### 2.5 How a leaky selection score can pick worse hyperparameters

Hyperparameter selection is an arg-max:

```latex
\hat\theta = \arg\max_{\theta} \hat S(\theta),
```

where $\hat S$ is the selection score. We want $\hat S$ to rank settings in
the same order as the true score $S$. Leakage adds a bias term that is
**not the same for every setting**:

```latex
\hat S(\theta) = S(\theta) + \text{bias}(\theta) + \text{noise}(\theta) .
```

If the bias were a constant, the arg-max would not change and only the
reported number would be wrong. Shuffled KFold adds a bias that **grows with
the model's capacity to memorise** (large C, large gamma), so the arg-max
moves toward memorising settings. In our numbers, bias(C = 0.1, gamma =
0.001) = 0.594 - 0.457 = 0.137, while bias(C = 100, gamma = scale) =
0.973 - 0.433 = 0.540. A bias that depends on $\theta$ is what turns a wrong
number into a wrong choice.

On valid, the cost of the wrong choice is 0.458 - 0.433 = 0.025 (mistake 2)
and 0.458 - 0.440 = 0.018 (mistake 3). Two honest caveats:

- Both are measured on the same valid set the correct variant selected on,
  so the correct variant wins by construction (2.1).
- On 197 clips, one clip changes macro-F1 by roughly 0.003 to 0.005, so
  differences of 0.02 are a few clips. The large, reliable damage of
  mistake 2 is to the **reported number** (0.973 vs 0.433), not to the
  choice.

With a stronger memorisation reward (e.g. a kNN with k = 1, or a larger
gamma grid) the wrong choice could be far worse; lesson 07 predicted that
window-level CV would pick k = 1 for kNN.

### 2.6 Spearman rank correlation

To compare rankings rather than values, replace each score by its rank
among the 16 settings and compute the ordinary (Pearson) correlation of the
ranks:

```latex
\rho = \operatorname{corr}\bigl(\operatorname{rank}(\hat S),\ \operatorname{rank}(S)\bigr) ,
\qquad
\rho = 1 - \frac{6 \sum_i d_i^2}{n(n^2 - 1)} \ \text{(no ties)},
```

with $d_i$ the rank difference of setting $i$ and $n = 16$. $\rho = 1$
means the same order, 0 means unrelated, -1 reversed. Ours: correct 1.000,
mistake 1 0.962, mistake 3 0.620, mistake 2 0.252. It does not care whether
scores are inflated, only whether they *order* the settings correctly, which
is all a selection score needs to do.

### 2.7 How to spot leakage in other people's notebooks

Read the code in the order the data flows, and ask at every step: "could
this line see anything about the rows it will be scored on?"

1. **Where is the split, and what is a row?** If rows are windows, frames,
   segments, augmentations, sentences of one document, or visits of one
   patient, a row-level `train_test_split` or `KFold` puts the same source on
   both sides. Look for a `groups=` argument (`GroupKFold`,
   `GroupShuffleSplit`, `StratifiedGroupKFold`). No groups + multiple rows
   per source = suspect.
2. **`fit` / `fit_transform` before the split.** `StandardScaler().fit_transform(X)`,
   `PCA().fit(X)`, `SelectKBest().fit(X, y)`, `SMOTE().fit_resample(X, y)` or
   target encoding on the full data, then `train_test_split`. The fix is a
   `Pipeline` passed whole into the CV.
3. **The score is too good.** Our own procedure produced 0.973 on GTZAN.
   Very high GTZAN accuracies from 3-second features with a random row split
   (a widely shared setup, using the Kaggle `features_3_sec.csv`) should
   trigger exactly this check. Compare with a group-split result before
   believing it.
4. **Selection and reporting on the same data.** `GridSearchCV(...).best_score_`
   reported as the final result; early stopping on the test set;
   "we tried several models and report the best on test".
5. **The metric used for selection differs from the one reported**
   (`scoring="accuracy"` but macro-F1 in the paper; window accuracy but
   clip-level claims).
6. **Duplicates and near-duplicates across splits.** GTZAN has exact
   repeats and same-artist clips (lesson 03); a random clip split puts them
   on both sides.
7. **Time.** For data with an order (charts by year, sensor logs), a random
   split lets the model see the future.
8. **Quick experiment.** Re-run their pipeline with a grouped split. If the
   score collapses (ours: 0.973 -> 0.433), the original number was leakage.

---

## 3. Why this choice, and the alternatives we rejected

| Choice | Alternative | Why |
|---|---|---|
| One change per variant | Combine mistakes in a "realistic bad notebook" | Combining would show a big number but not which mistake caused it; a viva asks "why", so each effect must be isolated. |
| True score = lookup in the correct grid | Refit each chosen setting separately | Identical by definition (the correct grid *is* "fit on train, score valid clip macro-F1" for every setting; lesson 08 asserts refit = grid score) and saves fits. |
| Mistake 1 without an inner scaler | Keep the Pipeline scaler | The inner scaler undoes the global scaling (affine map, verified: 0 changed predictions), so the mistake would vanish and the lab would teach the wrong lesson. |
| Mistake 2 on train windows only | KFold over train+valid | The specification says train windows; it is also how the mistake appears in practice (CV on "the training data"), and it keeps valid untouched for the true score. |
| Mistake 2 keeps clip macro-F1 as its scorer | Window accuracy in the KFold | Changing the scorer too would mix mistakes 2 and 3. |
| `random_state=42` for KFold | Unseeded | The JSON must be reproducible; two runs gave identical files. |
| Spearman correlation | Pearson on raw scores | Selection only needs the order; raw-score correlation would be affected by the large constant part of the KFold inflation. |
| SVM as the test model | kNN or XGBoost | The SVM has two capacity knobs whose memorisation corner (large C, gamma) is easy to see, and its correct heatmap is already in lesson 08. kNN with k = 1 would show an even stronger effect (lesson 07). |

---

## 4. Code walkthrough (`scripts/leakage_lab.py`)

- **L22–25**: imports from `src/`. The lab reuses the project's own loader,
  model factory, scorer and predefined split; nothing is re-implemented.
- **L31–33 `search`**: `GridSearchCV` over the config SVM grid with a
  given estimator, data, splitter and scorer; `refit=False` because we only
  need the per-setting scores; `n_jobs=-1`.
- **L36–38 `key`**: turns `{"clf__C": 100, "clf__gamma": 0.001}` into
  `"C=100, gamma=0.001"`, so settings from different searches can be
  matched against the truth table.
- **L41–56 `summarise`**: L44 takes the first best setting (`argmin` of
  rank, ties -> first, as in lesson 07); L50 looks up its true score; L52–53
  Spearman correlation between the variant's 16 selection scores and their
  16 true scores; L54–56 keep the whole grid for later inspection.
- **L59–62 `sibling_share`**: for every fold, which held-out windows have a
  clip id that also appears in that fold's training rows.
- **L65–97 `plot`**: paired bars (blue reported, orange true), value
  labels with a white background so the dashed reference line does not
  cross them, chosen params in the tick labels, y axis from 0.
- **L100 `main`**:
  - L102–106: train and valid windows of split C (test is locked by
    `load_split`), the train -> valid predefined split, clip macro-F1 scorer.
  - L111–112: **correct variant**; its 16 scores become `truth`.
  - L113–115: assert that `truth` equals the grid saved in lesson 08's
    `C_s0_svm.json`, so the lab and the model results cannot drift apart.
  - L120–123: **mistake 1**: scaler fit on `X` (train+valid stacked),
    applied to both, Pipeline scaler replaced by `"passthrough"`, same
    split and scorer.
  - L129–130: **mistake 2**: `KFold(5, shuffle=True, random_state=42)` on
    `X_tr, y_tr` only, correct pipeline and scorer.
  - L136: **mistake 3**: correct split and pipeline, `scoring="accuracy"`.
  - L141–142: how far the leaky scaler's means and SDs are from the
    train-only ones, in train SD units.
  - L146–147: the inner-scaler check: correct pipeline (with its scaler)
    fit on leakily pre-scaled train, compared with the same pipeline on raw
    train.
  - L160 onwards: write the JSON, draw the figure, print the table.

---

## 5. Pitfalls

1. **Reading the lab as "mistake 1 is harmless".** It moved the reported
   score by 0.006 *here*, in the gentle case of a scaler and similar
   train/valid distributions. The same pattern with feature selection or
   SMOTE, or with shifted distributions, is not gentle.
2. **Reading the lab as "the correct method finds better models".** On
   valid it wins by construction. The lab's evidence is about reported
   numbers and rankings (2.1).
3. **Fixing mistake 2 with GroupKFold by clip and stopping there.** Clips
   of one artist share a lot (lesson 04). Within split C train, 351 of the
   442 clips belong to one of 55 artists with more than one train clip (up
   to 26 clips for one artist), so clip-grouped CV would still leak artist
   identity. Group by artist.
4. **Comparing the reported numbers across variants.** Mistake 3 reports a
   window accuracy; the others report clip macro-F1. Always state the metric
   with the number.
5. **Pre-scaling and keeping the inner scaler, then "detecting" no leak.**
   The inner scaler cancels the affine leak exactly (up to rounding); a
   non-affine leaky step (PCA with fewer components, feature selection)
   would not be cancelled.
6. **Running this lab on test.** It would turn the test part into a second
   validation set. The lab only loads train and valid; `load_split` raises
   for test before the freeze tag.
7. **Shuffle=False does not fix KFold here.** Windows are stored clip by
   clip, so unshuffled KFold would split mostly between clips but cut some
   clips at fold boundaries and depend on file order: correct by accident.
   Use explicit groups.

---

## 6. Results from this step

From `results/val/leakage_lab.json` and `results/figures/leakage_lab.png`
(split C, train 4,420 windows, valid 1,970 windows / 197 clips):

| variant | chosen C, gamma | selection metric | reported | true valid clip macro-F1 | Spearman vs true |
|---|---|---|---|---|---|
| correct | 100, 0.001 | valid clip macro-F1 | 0.458 | **0.458** | 1.000 |
| (1) scaler on train+valid | 100, 0.001 | valid clip macro-F1 (leaky scaling) | 0.452 | 0.458 | 0.962 |
| (2) KFold over train windows | 100, scale | 5-fold CV clip macro-F1 | **0.973** | 0.433 | 0.252 |
| (3) select by window accuracy | 10, 0.001 | valid window accuracy | 0.452 | 0.440 | 0.620 |

Diagnostics:

- Mistake 1: feature means moved by at most 0.060 train SDs (median 0.024);
  SD ratio leaky / train-only 0.937 to 1.203. With the inner scaler kept,
  maximum probability difference 0.00055 and 0 of 1,970 window predictions
  changed.
- Mistake 2: 100% of held-out windows had a sibling window from the same
  clip in the training part of their fold.

Full grids (selection score -> true score) for each variant are in the
JSON; the KFold grid is reproduced in 2.3.

---

## 7. Viva questions with model answers

**Q1. Your KFold CV says 0.973 and your valid says 0.433. Which do you
report, and why?**
The valid number, 0.433. Every held-out window in the KFold had siblings
from the same recording in training (100% measured), so the CV score
measures recognising recordings. Valid is artist-disjoint from train, so it
measures recognising genres on new music, which is the question.

**Q2. Mistake 1 changed nothing about the choice and lowered the score.
Why call it a mistake?**
Because the number comes from a procedure that used valid's feature
distribution, which a deployed model would not have for new clips. That it
happened to cost 0.006 here is luck: valid is similar to train and a scaler
has few parameters. The same habit with label-using steps (feature
selection, SMOTE, target encoding) gives large optimistic errors.

**Q3. What if you had kept the StandardScaler inside the Pipeline and also
scaled globally first?**
The inner scaler would re-standardise with train-only statistics, and since
both steps are affine, the composition equals train-only scaling. We
measured a maximum probability difference of 0.00055 and no changed
predictions. The leak only acts when the model sees the globally scaled
data directly.

**Q4. Why does leakage change which hyperparameters are chosen, not just the
score?**
Because its bias depends on the setting. Memorising settings (large C,
large gamma) gain most when a held-out window's siblings are in training:
bias 0.540 for C = 100, gamma = scale, but only 0.137 for C = 0.1, gamma =
0.001. A setting-dependent bias moves the arg-max. A constant bias would
only shift the number.

**Q5. Is choosing by window accuracy leakage?**
Not strictly: no evaluation information enters training. It is a metric
mismatch: we select by one quantity (window accuracy, via OvO `predict`,
weighting genres by their window count) and report another (soft-voted clip
macro-F1, genres weighted equally). It picked C = 10 (true 0.440) instead of
C = 100 (0.458), and ranks settings with Spearman 0.620 against the truth.

**Q6. How would you correct mistake 2 if you had to use cross-validation
(e.g. for split B with 5 seeds)?**
Use a group splitter with groups = artist (`GroupKFold` or
`StratifiedGroupKFold`), and pass a Pipeline (scaler inside) into
`GridSearchCV` so that every fold refits its own scaler. Score per clip.

**Q7. The correct variant's true score is the highest. Does that prove the
correct method picks better models?**
No. The "true" score is the valid score the correct method maximised, so it
wins by construction. The lab proves that the leaky *reported numbers* are
wrong and that the leaky *rankings* disagree with the truth (Spearman 0.252
for KFold). Whether a choice is better on new data needs the test part or
a bootstrap.

**Q8. What if each clip had only one window (30 s features, no
windowing)?**
Then shuffled KFold over clips would no longer put siblings on both sides;
the within-clip leak would disappear. Artist leakage would remain, because
a random clip split still puts clips by one artist on both sides (lesson
04: about 71% of test clips in split B' have an artist in train).

**Q9. A notebook reports 92% on GTZAN with 3 s features. What do you check
first?**
How the split was made. If it is `train_test_split` or `KFold` on rows of
the 3 s feature table, windows of one song are on both sides, exactly our
mistake 2, and we have just seen that this produces scores near 0.97 for a
model that really scores 0.43. Then I check where the scaler was fit, and
whether duplicates and same-artist clips are separated.

**Q10. Why does the KFold heatmap rank C = 0.1, gamma = 0.001 last?**
That is the smoothest, most regularised setting: it cannot memorise
individual windows, so it gains least from siblings in training (0.594).
On genuinely new artists it is nearly the best (0.457). Leaky CV punishes
exactly the property we want.

---

## 8. Further reading

- Kaufman, Rosset, Perlich & Stitelman (2012), *Leakage in data mining:
  formulation, detection, and avoidance*, ACM TKDD 6(4).
  https://doi.org/10.1145/2382577.2382579
- Kapoor & Narayanan (2023), *Leakage and the reproducibility crisis in
  machine-learning-based science*, Patterns 4(9) (taxonomy of leakage types,
  checklist). https://doi.org/10.1016/j.patter.2023.100804
- Cawley & Talbot (2010), *On over-fitting in model selection and
  subsequent selection bias in performance evaluation*, JMLR 11.
  https://www.jmlr.org/papers/v11/cawley10a.html
- Varma & Simon (2006), *Bias in error estimation when using
  cross-validation for model selection*, BMC Bioinformatics 7:91.
  https://doi.org/10.1186/1471-2105-7-91
- Sturm (2013), *The GTZAN dataset: its contents, its faults, their effects
  on evaluation, and the future use of GTZAN*. https://arxiv.org/abs/1306.1461
- Flexer (2007), *A closer look on artist filters for musical genre
  classification*, Proc. ISMIR 2007 (artist effect on genre accuracy).
- scikit-learn user guide: *Common pitfalls and recommended practices*
  (data leakage section) https://scikit-learn.org/stable/common_pitfalls.html ;
  *Cross-validation iterators for grouped data*
  https://scikit-learn.org/stable/modules/cross_validation.html#group-k-fold
- Spearman (1904), *The proof and measurement of association between two
  things*, American Journal of Psychology 15(1).
  https://doi.org/10.2307/1412159
