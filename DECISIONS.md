# Design decisions

Each entry: date, decision, reason, alternative rejected.

## 2026-10-29: Split C (fault-filtered, artist-disjoint) is the main result

**Decision.** The headline numbers come from split C: the published
fault-filtered, artist-disjoint GTZAN split (443 train / 197 valid / 290 test
clips; Kereliuk, Sturm & Larsen 2015). Splits A, B and B' are reported only to
measure how much accuracy leakage adds.

**Reason.** GTZAN has many clips by the same artist, plus exact and near
duplicates, mislabels and distorted clips (Sturm 2013). With a random split,
the same artist (sometimes the same recording) lands in both train and test, so
a model can score well by recognising the artist, the album's production or the
recording itself rather than the genre. Our question asks about *new* music,
and a new clip in the real world normally comes from music the model has not
heard. Split C is the only one of our four splits where no test artist appears
in training, so it is the only one whose score answers that question. Using a
published split also makes our numbers comparable with prior work.

**Alternatives rejected.**
- *Random clip split (B) as main result:* it is the common choice in tutorials,
  but artist leakage inflates its score, so it overstates real-world accuracy.
- *Window-random split (A):* windows of one clip end up in both train and test,
  which is the worst leakage; kept only as a demonstration of that effect.
- *Building our own artist-disjoint split:* GTZAN has no official artist labels;
  the published split relies on Sturm's manual artist annotations. Re-deriving
  them would be slow and error-prone, and the result would not be comparable.
- *Cross-validation over C:* the fixed split is small (290 test clips), so we
  report an artist-cluster bootstrap CI instead to show the uncertainty.

All four splits use the same 443/197/290 clip counts, so differences between
them come from leakage, not from training-set size.

## 2026-10-29: Corrupt file jazz.00054 is recorded, then excluded

**Decision.** Every file gets an audit row in `splits/metadata.csv` (1000 rows).
Files that fail to load get `load_ok = False` and a row in
`results/load_errors.csv`, and every later step uses only `load_ok == True`
clips. In the Kaggle copy exactly one file fails: `jazz.00054.wav` has no
RIFF/WAVE header (its first bytes are already sample data), so neither
libsndfile nor audioread can open it. That leaves 999 clips, jazz has 99.

**Reason.** Keeping the row makes the loss visible and auditable instead of
silently shrinking the dataset. Loading happens inside try/except, so one bad
file cannot stop the pipeline.

**Alternatives rejected.**
- *Repairing the file* (reading the bytes as headerless 16-bit PCM): the
  sample rate, channel count and start offset are guesses; a wrong guess makes
  a plausible-looking but wrong clip.
- *Replacing it* with the clip from another GTZAN mirror: mixing sources breaks
  the "one download, one md5 per file" record, for a gain of one clip in 999.
- *Dropping it silently* in `list_clips`: hides a data fault that a grader
  would want to see.

## 2026-10-29: Kaggle's precomputed CSVs and images are ignored

**Decision.** Only `Data/genres_original/*.wav` is used. `features_30_sec.csv`,
`features_3_sec.csv` and `images_original/` are never read.

**Reason.** They were made by unknown code with unknown settings (window
boundaries, FFT size, scaling, image rendering), so they cannot be reproduced
or checked, and they would not match our config (`mel`, `mfcc`, 3 s windows).
The PNG spectrograms are rendered plots (axes cropped, colour-mapped,
compressed), not the spectrogram values. The 3 s CSV also invites the
window-random split (rows from one clip in train and test), the leak our
split A exists to measure. Computing features ourselves from the audio keeps
every number traceable to `config.yaml`.

**Alternative rejected.** *Using the CSVs as a quick baseline:* faster, but a
baseline on different features cannot be compared fairly with our models.

## 2026-10-29: Audit definitions

**Decision.** The audit loads audio with `librosa.load(sr=22050, mono=True)`,
the same call the feature step will use. A sample counts as clipped when
`|x| >= 0.999`; `clip_share` is the fraction of such samples. Loudness is
`rms_db = 20 log10(RMS)` in dB relative to full scale (dBFS). Byte-identical
files are found by md5 of the file bytes.

**Reason.** "Loads in the audit" then means exactly "loads in the pipeline".
0.999 rather than 1.0 because 16-bit PCM decoded to float peaks at 32767/32768
≈ 0.99997, never at exactly 1.0.

**Alternative rejected.** *Clipping as runs of consecutive full-scale samples:*
more precise for detecting distortion, but more code for an audit whose job is
only to flag clips worth listening to.

## 2026-10-29: Features: 3 s windows, clip-level tempo and HPSS, ref=1.0, float16 mels

**Decision 1: 3 s non-overlapping windows, clips padded/trimmed to exactly 30 s.**
Each clip is zero-padded at the end or trimmed to 661,500 samples, then cut
into 10 windows of 66,150 samples. The 9 clips shorter than 30 s (max 1,500
samples short) are padded; this is recorded as `clip_padded` in
`features/index.csv`.
*Reason:* 10 training rows per clip, a fixed input size for the CNN, and 3 s is
long enough to contain several beats and notes.
*Rejected:* overlapping windows (more rows but near-duplicate rows; adds no new
audio); dropping short clips (loses 9 clips for 0.07 s of audio); a 9-window
scheme (wastes ~3 s of every clip).

**Decision 2: tempo computed once per clip, on the full 30 s.**
`librosa.beat.beat_track` on the whole clip; the same value is the 57th
feature of all 10 windows.
*Reason:* a 3 s window contains only ~5-7 beats, too few for a stable tempo
estimate; tempo is a property of the song.
*Rejected:* per-window tempo (noisy, frequent octave errors).

**Decision 3: HPSS on the full clip, then sliced into windows.**
`librosa.effects.hpss` once per clip, cut into the same 10 windows.
*Reason:* HPSS median-filters the spectrogram along time and frequency; on the
full clip it is computed once instead of 10 times, and windows are not hurt by
filter edge effects at their boundaries. It is still a fixed per-clip
transform: no information from other clips.
*Rejected:* HPSS per window (about 10x slower, edge effects every 3 s).

**Decision 4: `power_to_db(ref=1.0)` instead of `ref=np.max`.**
*Reason:* with `ref=1.0` all clips share one fixed dB scale, so absolute
loudness differences (pop ≈ −13 dBFS vs classical ≈ −28 dBFS median RMS, see
lesson 03) stay in the features. `ref=np.max` would rescale each clip to its own
peak and erase them. Whether a model should use loudness is a modelling
question; it is answered later by normalisation fitted on training data, not
by throwing the information away here. librosa's default `top_db=80` is kept:
it floors each clip at 80 dB below its own maximum, a per-clip fixed transform.
*Rejected:* `ref=np.max` (per-clip normalisation hidden inside the features).

**Decision 5: log-mels cached as float16, one file per clip.**
*Reason:* 999 clips × 128 × 1292 values: float16 ≈ 330 MB vs ≈ 660 MB in
float32. float16 has ~3 significant digits; at 40 dB the step is ~0.03 dB,
far below anything audible or meaningful to a CNN. One file per clip lets the
CNN load clips lazily; full clips (not windows) are stored so windowing stays
a choice of the training code.
*Rejected:* float32 (2x disk for no benefit); storing pre-cut windows (fixes
the window scheme into the cache).

## 2026-10-29: Splits A, B, B', C: what each can still leak

**Decision.** Four splits, all with identical sizes (see next entry), built by
`scripts/make_splits.py` into `splits/`. Each file lists every window of the
999 usable clips with `part` in {train, valid, test, unused}.

| Split | Built how | Removes | Can still leak (measured on test, `results/split_summary.json`) |
|---|---|---|---|
| A | windows drawn at random from the 9,990 windows (seed 0) | nothing | the same clip: 970 of 974 test clips also have windows in train; same artist 98.5% |
| B | clips drawn from the 999 usable clips, stratified by genre (seeds 0-4) | within-clip leakage | exact duplicates (2-5 test clips have a byte-identical file in train) and artists (69.0-75.5% of test clips share an artist with train) |
| B' | same as B but drawn from the 929 usable fault-filtered clips (seeds 0-4) | + duplicates and faulty clips | artists (68.3-72.8%) |
| C | published fault-filtered, artist-filtered split | + artist leakage, as far as the published labels go | 22 of 290 test clips (7.6%) share a gtzan-bind artist name with train (next entry but one) |

Each step of the ladder A → B → B' → C removes one kind of leakage while
keeping sizes fixed, so the drop in score from one step to the next estimates
how much that kind of leakage was inflating the score.

**Why identical sizes.** If C had fewer training clips than B, a lower score on
C could come from less data, not from removing leakage. Fixing the train,
valid and test sizes for all four splits (and stratifying B and B' by genre)
makes leakage the only systematic difference. B and B' use 5 seeds to show how
much the score moves from the random draw alone.

**Rejected:** each split at its natural size (A and B from all 999 clips):
confounds leakage with training-set size.

## 2026-10-29: All splits use 442/197/290 clips (jazz.00054 is in published C train)

**Decision.** The published split C has 443/197/290 clips, but jazz.00054 is
in its train part and cannot be loaded. C therefore has 442 usable train clips,
and `config.yaml` `split_sizes` is set to 442/197/290 for all splits (A uses
4,420/1,970/2,900 windows). jazz.00054 keeps its `split_C = train` label in
`splits/metadata.csv` with `load_ok = False`, and has no windows in any split file.

**Reason.** B' is drawn from the same pool as C (929 usable clips), which cannot
supply 443 + 197 + 290 = 930. Using C's usable sizes everywhere keeps all
four splits identical in size.

**Rejected:** keeping 443 for A and B only (sizes would differ from B'/C by
one clip; small, but it breaks the "identical sizes" rule for no benefit);
replacing jazz.00054 with another clip in C (would no longer be the published split).

## 2026-10-29: Published split C kept as the main split despite residual artist overlap

**Decision.** Use the published split C unchanged (option 1 of three), and
report its artist overlap rather than hide or remove it.

**Finding.** With artist names from gtzan-bind (`seungheondoh/gtzan-bind`,
`artist_name`), 13 artists have clips in more than one part of C: Anthrax,
Bob Marley, Boney M., Burning Spear, Danzig, Dark Tranquillity, Earth, Wind &
Fire, Freddie McGregor, the Juilliard String Quartet credit, KC and the
Sunshine Band, Metallica, Peter Tosh, Slayer (74 clips). 22 of 290 test clips
(7.6%) share an artist with train; 14 clips in C have no artist name. The
published split was built from Sturm's own artist annotations; gtzan-bind's
names appear to come from automatic (Shazam) matching, so the two may disagree.
We have not verified which is right.

**Reason.** The published split is what prior work reports on, so our numbers
stay comparable; its sizes match the design; and the overlap is small compared
with B and B' (≈70%). Reporting it lets readers judge.

**Rejected.**
- *Dropping the 74 conflicting clips* ("C-strict"): changes sizes and no longer
  matches the published split. Possible later as a sensitivity check.
- *Rebuilding an artist-disjoint split from gtzan-bind names*: exact disjointness
  by those names, but not the published split, and it inherits any errors in
  the automatic artist matching.

`tests/test_splits.py` pins the exact set of 13 crossing artists, so any change
to C or the metadata is caught.

## 2026-10-29: Evaluation design (fixed before any model is compared)

**Decision.**
- Primary metric: clip-level macro-F1 after **soft voting** (mean of the 10
  window probabilities). Majority vote (ties broken by mean probability) is
  reported only as a comparison.
- Hyperparameters are scored by `ClipMacroF1Scorer` on the valid part via a
  `PredefinedSplit` (train = −1, valid = 0), never by k-fold over windows.
- Uncertainty: 95% percentile bootstrap, 2,000 resamples, resampling
  **artists** (clusters); a clip with no artist name is its own cluster.
- Model comparisons: paired bootstrap on the same resampled clusters,
  reporting the difference, its 95% CI and the share of resamples where A > B.

**Reason.** Clips are the unit of prediction; macro-F1 weighs every genre
equally; soft voting keeps window confidence; clips by one artist are
correlated, so resampling clips would make intervals too narrow (in a
synthetic check on C-valid's artist structure, 79 clusters for 197 clips, the
artist CI was 2.7× wider than the clip CI); pairing removes the shared
difficulty of the test set from a comparison.

**Rejected.** Window-level accuracy (counts each clip 10 times, not what a user
sees); accuracy as primary (hides a failing genre); normal-approximation CIs
(macro-F1 is bounded and its sampling distribution is not guaranteed normal);
treating missing-artist clips as one shared "unknown" artist (would glue
unrelated clips together); comparing two models by whether their separate CIs
overlap (ignores that they are scored on the same clips).

## 2026-10-30: Baselines and the classical tuning protocol

**Decision.**
- Two floors: `DummyClassifier` most-frequent and stratified (random draws
  with the train genre frequencies). Then kNN and multinomial logistic
  regression as simple baselines; every scaled model has its
  `StandardScaler` inside a `Pipeline`.
- Grids live in `config.yaml`: kNN k ∈ {1, 3, 5, 9, 15, 25} × weights ∈
  {uniform, distance}; logistic regression C ∈ {0.01, 0.1, 1, 10}, L2,
  `max_iter` 2000.
- `GridSearchCV(cv=PredefinedSplit, scoring=ClipMacroF1Scorer, refit=False)`.
  The best setting (ties → first in grid order) is refit on **train only**, and
  that model's valid predictions give every number in `results/val/*.json`.
  An assert checks that the refit score equals the grid-search score.
- Valid JSONs also record majority-vote macro-F1 and window accuracy, as
  diagnostics only.

**Reason.** A floor shows what "no signal" scores under our metric (with 10
genres macro-F1 can be far below accuracy for a constant predictor); a simple
baseline shows what any later model has to beat. `refit=True` would retrain
on train+valid, so per-genre F1 and the confusion matrix "on valid" would
come from a model that had seen valid. Grids in config follow the
all-constants-in-config rule.

**Rejected.** k-fold CV over windows (windows of one clip in both folds, see
lesson 04); `refit=True` (valid would no longer be held out for the reported
diagnostics); unscaled kNN/logreg (feature standard deviations run from 0.002 for `harm_rms_var` to 1.5e6 for `rolloff_var`, so
distance and the L2 penalty would be ruled by the largest-unit features);
picking the "simplest" model among near-ties (a rule we would have to
invent; ties are reported in the grid instead). Whether the final test model
is refit on train+valid is decided before the test freeze, not here.

## 2026-10-30: Classical models fit with one BLAS thread

**Decision.** The train-only refit in `tune_and_score` runs inside
`threadpool_limits(1)`, matching the grid-search workers (joblib gives each
worker one BLAS thread when `n_jobs=-1`).

**Reason.** Measured: logistic regression (C = 1) fit with all threads vs one
thread gave coefficients differing by up to 0.043 and 8 different valid
window predictions (valid clip macro-F1 0.457 vs 0.452). lbfgs stops at a
tolerance, and the thread count changes the floating-point summation order,
so it stops at a slightly different point. One thread makes the numbers the
same on every machine and makes the refit agree with the grid score (asserted).

**Rejected.** `n_jobs=1` everywhere (slower, still thread-count dependent in
the main process); converting features to float64 (reduces but does not
remove the dependence); ignoring it (the asserted equality would hold or fail
depending on the machine's core count).

## 2026-10-30: RBF SVM tuned on C x gamma, probabilities by Platt scaling

**Decision.** `Pipeline(StandardScaler, SVC(kernel="rbf", probability=True,
random_state=42))`, grid C {0.1, 1, 10, 100} x gamma {"scale", 0.001, 0.01,
0.1}, tuned like kNN/logreg (`GridSearchCV` on the predefined train->valid
fold, clip macro-F1, refit on train only). The JSON stores the numeric value
of `gamma="scale"` on our training data (0.0175) so the heatmap can place it
on the gamma axis.

**Reason.** C and gamma together set how flexible the boundary is; a log grid
over both is the standard search (Hsu, Chang & Lin). Soft voting needs window
probabilities, and an SVM only gives margins, so sklearn fits Platt's sigmoid
on top. gamma values are written as plain decimals in config because YAML
reads `1e-3` as a string.

**Rejected.** A linear SVM (logistic regression already covers the linear
case); `CalibratedClassifierCV(SVC(), ensemble=False)` (the replacement for
the deprecated `probability=True`, needed only from sklearn 1.11; we pin
1.9.1); majority vote on `predict` (would avoid Platt but throws away
confidence, and lesson 06 chose soft voting for all models). Known caveat:
Platt's internal 5-fold CV splits windows, not clips, so sibling windows can
sit on both sides of it; this affects only the probability mapping, never
which data the valid score is computed on.

**Note.** The best setting (C = 100, gamma = 0.001) lies on the corner of the
grid. The grid was fixed by the step specification and was not widened after
seeing valid scores.

## 2026-10-30: XGBoost tuned by a manual loop with early stopping on valid

**Decision.** `XGBClassifier` on unscaled window features, integer labels in
config genre order. Loop over max_depth {3, 5, 7} x learning_rate {0.05, 0.1};
fixed n_estimators 2000, subsample 0.8, colsample_bytree 0.8. Each fit
early-stops when valid **window** mlogloss has not improved for 50 rounds;
predictions use the trees up to the best round. The setting with the highest
valid **clip** macro-F1 wins (ties: first in grid order). The best setting's
train/valid mlogloss per round is stored in the JSON for the curve. The
summary table now reads every result JSON for the split, so it lists all
models even when only some were re-run.

**Reason.** The number of trees is the most important XGBoost capacity knob
and early stopping finds it in one fit instead of a grid over it. mlogloss is
smooth and changes every round, so it is a good stopping signal; clip
macro-F1 is our metric, so it chooses between settings, as for every other
model. Trees split on thresholds, which do not change under any monotone
rescaling of a feature, so no scaler is needed.

**Rejected.** `GridSearchCV` (cannot pass a per-fit `eval_set` cleanly, and
n_estimators would become another grid axis); early stopping on clip
macro-F1 (jumpy: one clip changes it by ~0.005, so the stopping point would be
noise); an inner train-only hold-out for early stopping (would shrink the
already small, artist-disjoint train set and break the equal-size design).
Known caveat: valid now picks both the tree count and the setting, so the
valid score is slightly optimistic for XGBoost. The test part answers that.

## 2026-10-30: Leakage lab design (scripts/leakage_lab.py)

**Decision.** Four SVM tunings on split C train/valid: the correct one and
three mistakes, each changing exactly one thing: (1) StandardScaler fit on
train+valid and the Pipeline's own scaler set to `passthrough`; (2)
`KFold(5, shuffle=True, random_state=42)` over train windows with the
correct pipeline and clip macro-F1 scorer; (3) the correct split with
`scoring="accuracy"` (window accuracy). The "true" score of any chosen
setting is its valid clip macro-F1 from the correct grid, which the script
asserts equals `results/val/C_s0_svm.json`. Spearman correlation between
each variant's selection scores and the true scores is recorded.

**Reason.** Isolating one mistake per variant makes each effect
attributable. The inner scaler is removed in (1) because pre-scaling
followed by a train-only scaler is an affine composition that cancels the
leak (measured: 0 changed predictions), which would hide the mistake.

**Rejected.** A combined "bad notebook" variant (effects not attributable);
running the lab on test (would make test a second validation set); a
GroupKFold-by-clip variant (not requested; would still leak artists, since
351 of 442 C-train clips share an artist with another train clip). Caveat
recorded in lesson 09: the correct variant wins on valid by construction,
so the lab's evidence is about reported numbers and rankings, not about
which choice is better on new data.

## 2026-10-30: Log-mel CNN design and training protocol

**Decision.** `GenreCNN`: 4 blocks Conv 3x3 (32, 64, 128, 128) -> BatchNorm
-> ReLU -> MaxPool 2x2, global average pooling, dropout 0.3, Linear(10);
conv layers without bias (BatchNorm's beta replaces it); 241,898 parameters.
Input: 130-frame (3 s) crops of each clip's full log-mel, normalised per
mel bin with mean/std from the **training clips only** (saved to
`results/mel_norm_{split}_s{seed}.json`). Training: an epoch = 10 random
crops per training clip, random gain +/-6 dB before normalisation,
SpecAugment (2 frequency masks <= 15 bins, 2 time masks <= 20 frames, set to
0 = the training mean) after it; Adam lr 1e-3, weight decay 1e-4, batch 64,
at most 60 epochs; after every epoch, valid clip macro-F1 (soft vote over the
10 fixed windows) decides early stopping (patience 8) and the best weights
are kept. Torch seed = config seed (42) + run seed; three run seeds (0, 1, 2)
from config. `valid_report` now takes (probs, classes, labels indexed by
clip) so the CNN is scored by the same code as the classical models.
`--refit_epochs N` trains on train+valid for N epochs (stats from
train+valid) for later use. (Split A was first refused by the CLI; see
"CNN on window-level split A" below for how it is handled now.)

**Reason.** A small CNN is the standard learned-feature model for log-mels
and is trainable on a CPU (about 49 s per epoch here). Random crops
multiply the effective training set (1,163 possible offsets per clip) and
augmentation counters overfitting with only 442 training clips. Early
stopping on the project metric keeps model selection consistent with every
other model. Per-bin normalisation from train only follows the
learned-preprocessing rule.

**Rejected.** Fixed 10 windows per clip for training (no crop diversity);
per-clip or per-window normalisation (removes absolute loudness/brightness
information and differs between train and eval if done on crops); AdamW
(decoupled weight decay; the step specification asks for Adam); a
pretrained audio network (e.g. PANNs, VGGish: trained on far more data,
possibly overlapping GTZAN sources, and it would answer a different
question); early stopping on valid loss (smooth, but not the project
metric; logged per epoch for the curves). Known caveat: valid picks the
epoch, so the CNN's valid score is slightly optimistic, like XGBoost's.

## 2026-10-30: One tuning protocol for every split; CNN with one run per split seed

**Decision.** `scripts/tune_all.sh` (also `make tune`) runs the same
protocol on A (seed 0), B (seeds 0-4), B' (seeds 0-4) and C: dummies, kNN,
logistic regression, SVM and XGBoost with the config grids, selected on
**that split's own valid part** by clip macro-F1 (soft vote), refit on train
only; the CNN with the config architecture and training settings,
early-stopped on the same metric. The CNN runs **once per split seed** on
A, B and B' (run seed = split seed; 11 runs) and three times on C (run seeds
0-2). Chosen settings are collected in `results/chosen_params.json`, scores
in `results/val/summary.csv`.

**Reason.** The ladder A -> B -> B' -> C is meant to measure leakage, so
nothing else may differ between the rungs: same models, grids, selection
rule, metric and training sizes. One CNN run per split seed: a CNN run takes
14 to 35 min on this CPU (the three C seeds: 14.4, 35.4, 26.2 min), so
three runs per split seed would add 22 runs (11 split seeds x 2), about 5-13 h; the 5 split seeds of B and B' already give a
spread that includes seed-to-seed CNN variation.

**Rejected.** Reusing C's chosen hyperparameters on the other splits (each
split must select on its own valid part, or A/B scores would not show what a
leaky protocol would really pick); 3 CNN seeds everywhere (time, see above).

## 2026-10-30: CNN on window-level split A uses fixed train windows

**Decision.** On split A (windows, not clips, are assigned to parts) the CNN
trains on exactly the train windows at their fixed positions (still with
gain and SpecAugment, no random offsets), computes the per-bin
normalisation from those windows' frames only, and scores each clip by soft
vote over its valid windows, the same rows the classical models use. On
clip-level splits nothing changes (verified: identical items and stats).

**Reason.** Random crops of the full clip would read audio of the clip's
valid and test windows. Known residue: each 130-frame window overlaps its
neighbour by about one frame, and the centred STFT frames at a window edge
include ~46 ms of the neighbouring window. This is negligible next to A's
designed leak (sibling windows in train) and A is the leaky reference.

**Rejected.** Refusing split A for the CNN (the ladder would lack the CNN on
its first rung); random crops within runs of consecutive train windows
(more code for a split that is a leakage reference, not a result).

## 2026-10-30: Grid-vs-refit check becomes a warning (exact duplicates)

**Decision.** `tune_and_score` keeps comparing the refit model's valid
macro-F1 with the grid-search score, but a mismatch is logged as a warning
and both are stored (`grid_valid_macro_f1`), instead of failing.

**Reason.** On split A, kNN with k = 3 gave 0.9132 when refit and 0.912 in
the grid worker. Split A/B pools contain GTZAN's exact-duplicate files (114
duplicated window rows in A train+valid), so kNN meets equal-distance ties,
and the joblib worker's data copy breaks them differently. Fitting directly
on either slicing gives 0.9132. The reported numbers are the refit model's.
Split C (no duplicates) still matches exactly.

**Rejected.** Dropping the check (it caught nothing wrong here, but it is
the guard that reported diagnostics belong to the selected model).

## 2026-10-30: Tuning figures only for split C

**Decision.** `train_classical.py` draws `val_knn_k.png`,
`val_svm_heatmap.png` and `val_xgb_curve.png` only when run on split C.

**Reason.** The lessons cite these as split C figures; running the protocol
on A/B/B' would overwrite them. Other splits' grids are in their JSONs.

## 2026-10-30: Keep 3 s windows and soft voting (after the validation analyses)

**Decision.** The window length stays **3 s** (10 windows per clip) and the
clip prediction stays the **soft vote** (mean of window probabilities), for
every model.

**Reason.** Window length (SVM with its 3 s-tuned params, split C valid,
`results/val/analysis_window_length.json`): 1 s 0.435, 3 s 0.458, 5 s 0.477,
10 s 0.449. The spread (0.042) is a handful of the 197 valid clips, the
params were tuned at 3 s (favouring it), and 5 s would be chosen *after*
seeing valid scores for a 0.019 gain, which is selection on noise; changing
it would also change the CNN input and every tuned model. Voting
(`analysis_voting.json`): soft minus majority is between -0.008 (CNN seed 2)
and +0.013 (CNN seed 0) for every model, soft >= majority for 8 of 9
results; soft voting uses each window's confidence and has no ties, and it
was the rule fixed in advance (lesson 06).

**Rejected.** 5 s windows (post-hoc, within noise, would invalidate all
tuning); majority vote (never clearly better, discards confidence); per-model
choice of window length or vote (more freedom to overfit valid, and models
would no longer be compared on equal terms).

## 2026-10-31: Local Streamlit web app for classification and results

**Decision.** `app.py` (Streamlit UI) on top of `src/webapp.py` (all logic, no
Streamlit, tested). It classifies uploads (WAV, MP3, FLAC, OGG) and sample clips
and shows every validation result. It runs on localhost only. It serves the
single CNN run with the best valid clip macro-F1 on split C (read from
`C_cnn_summary.json`: run seed 2, 0.635); the dashboard headlines the mean over
seeds (0.610 +/- 0.028), not that best run. Uploads use 10 evenly spaced 3 s
windows over the whole file (first 10 minutes), soft-voted, the same steps as
evaluation. Samples are 2 random valid clips per genre, drawn with the config
seed before seeing any score. Scores are shown raw and labelled as uncalibrated.

**Reason.** GitHub Pages cannot run the model. Keeping the app local needs no
exception to the rules on audio and models: audio and weights are read from
git-ignored folders. Samples come from valid because predicting test clips before
the freeze would break the test lock. Random selection keeps wrong predictions in
the samples, so the demo does not flatter the model. Temperature scaling is not
applied: on seed 2 cross-fitting worsened clip ECE (0.077 to 0.114, lesson 11).
`tests/test_webapp.py` checks that the app's path reproduces the stored valid
score on all 197 clips, so serving cannot drift from what was evaluated.

**Rejected.** A public host (Hugging Face Spaces): needs an exception for model
files and copyright-safe samples, since GTZAN must not be republished; the user
chose local first. Client-side inference in `docs/` (ONNX): puts weights in the
public site and needs mel extraction rewritten in JavaScript, a second pipeline
that could drift. Gradio: good for one input-output demo, weaker for a multi-part
dashboard. Flask or FastAPI plus a hand-written page: much more code for the same
result. Averaging the three CNN seeds: a new model with no measured score, and
the final model is decided at the freeze. Hand-picking samples the model gets
right: misleading.

## 2026-10-31: AAC and M4A uploads through PyAV

**Decision.** Accept `.aac` and `.m4a` in the web app. `load_audio` still tries
librosa first and falls back to PyAV (`av`) only for files librosa cannot read.

**Reason.** A user's AAC file was refused, and libsndfile has no AAC decoder.
PyAV ships its own FFmpeg libraries, so nothing else needs installing. librosa
stays first so WAV, MP3, FLAC and OGG keep the exact path the stored scores were
measured on. PyAV resamples with its own filter, so AAC audio is not bit-identical
to what librosa would give. Measured on 30 valid clips re-encoded as AAC and as
M4A: same top genre on 29 of 30 in each, largest score change 0.063. The stored
scores still describe WAV input only.

**Rejected.** Installing ffmpeg system-wide for audioread: an extra manual step on
every machine, and audioread cannot read in-memory uploads. `imageio-ffmpeg`: a
separate binary and temp files for each upload. Decoding every format with PyAV:
would move the evaluated formats off their measured path.

## 2026-10-31: "Not sure" under a top score of 0.7

**Decision.** When the soft-vote top score is below `app.min_confidence` (0.7), the
web app heads the result "Not sure: best guess <genre>" with a short explanation,
and still draws the chart. The predicted genre itself does not change.

**Reason.** A real upload scored country at 27%, and stating that flatly overclaims.
Chosen on C valid only, 197 clips, served run: accuracy is about 0.43 in every band
from 0.4 to 0.7, then about 0.85 above it. Under 0.7: 103 clips, accuracy 0.427. At
0.7 or more: 94 clips, accuracy 0.883. Seeds 0 and 1 agree (0.466 and 0.360 below,
0.875 and 0.815 above). This is display only, so macro-F1 and the protocol are
unaffected. `tests/test_webapp.py` guards the claim in the app's wording. Cost: 52%
of valid clips show "Not sure". Scores above 0.7 can still be wrong (`blues.00009`,
87%), and 0.7 is unmeasured on music outside GTZAN.

**Rejected.** A cut-off of 0.5: states a genre on clips between 0.5 and 0.7, where
accuracy is under 0.5. Window agreement: accuracy does not rise steadily with it.
Quoting the measured accuracies in the app: they would be hand-copied numbers that
go stale, so the wording says "less than half" and a test checks it. Temperature
scaling first: it worsened calibration on this run (lesson 11).

## 2026-10-31: Whole-code audit: AAC decoding follows librosa, tuning resumes

**Decision.** `decode_with_av` now only decodes: it averages channels and resamples
with `librosa.resample`, as `librosa.load` does, and it skips damaged packets. This
replaces the FFmpeg mono downmix in the entry "AAC and M4A uploads through PyAV".
`tune_all.sh` skips runs that have a result file. The test lock stays shut when git is
missing, and `load_split` rejects unknown parts.

**Reason.** FFmpeg's float downmix made stereo AAC 2.98 dB louder and 5.1 AAC 10.64 dB
louder than librosa's average, and the CNN uses a fixed dB reference, so this was a
training-serving skew. The earlier check missed it because its clips were mono. Now,
on 30 valid clips, stereo AAC and M4A each keep the same top genre on 30 of 30, with a
largest score change of 0.030. Stored scores never used this path and are unchanged.

**Rejected.** Per-clip loudness normalisation: a change to the model's input, which
needs retraining. Replacing `SVC(probability=True)` now: it would change stored SVM
probabilities, and it is safe on the pinned sklearn 1.9.1. A hash of settings in each
result file, so that resuming notices changed grids: more machinery than one project
needs, so the script's header says to delete old files instead.

## 2026-11-01: The artist bootstrap clusters on shared credits, not exact names

**Decision.** `splits.credit_clusters` puts two artist credits in one cluster if they share a
performer, directly or through a third credit, after splitting credits on `, ` and ` & `. The
test-time bootstrap must use it: `cluster_bootstrap_ci` and `paired_bootstrap_diff` now read a
`cluster` column by default, so a caller who forgets gets a KeyError, not a silent exact-name
run. `artist_groups` keeps exact names, so the stored learning curves and cross-fitted
temperatures stay valid.

**Reason.** gtzan-bind writes one performer in several ways: "Bob Marley & The Wailers" (32
clips), "Bob Marley" (3). Exact names give 321 artists; shared credits give 294. Resampling a
performer's clips as two independent clusters makes the interval too narrow, which is the
error the artist bootstrap exists to prevent. Test clips linked to train in C rise from 22
(7.6%) to 25 (`results/split_summary.json`, `test_clips_credit_cluster_in_train`). All 17
merged groups were read by eye and are correct. Found before the freeze, so no test score
informed it.

**Rejected.** A hand-written alias table: it found only lead-name variants and missed links
such as "Los Angeles Philharmonic & Leonard Bernstein". Changing `artist_groups` itself: every
learning curve subset would change, needing about 12 CNN fits and new numbers in lesson 11,
for a change that touches few artists in analyses that do not report a final score. Rebuilding
split C from these clusters: it would stop being the published split.

## 2026-11-01: Second audit: resume under WSL, near-silent uploads, atomic result files

**Decision.** `tune_all.sh` strips CR from the seed list. Uploads below `app.min_rms_db`
(-60 dBFS) are refused as silent. Result JSONs that mark a run as finished go through
`utils.write_json` (temporary file, then rename). `bootstrap.n`, `bootstrap.ci`, `audit.clip_threshold`
and `mfcc.n_mfcc` are now read from config. `torchaudio` is dropped from requirements: nothing
imports it. `scripts/check_lesson_refs.py` checks named line references in the lessons.

**Reason.** Run from WSL, `.venv/Scripts/python.exe` prints CRLF, so the last CNN seed became
`2\r`, its result file never matched, and every resume retrained seed 2, the served model.
One bit of dither came back as "classical" at 99.7% with no "Not sure". -60 dBFS is 17.6 dB
below the quietest GTZAN clip (-42.4 dBFS), so no real clip is near it. A half-written result
crashed the dashboard and would have counted as finished on resume.

**Rejected.** Using `.venv/bin/python` in WSL: the venv is a Windows one. A pytest for the CR
fix: Git Bash strips CR itself, so it could not fail there; the fix was checked with a stub
under WSL before and after. Making the reference check a test: moving code is not a bug, and
it would fail on every edit above a referenced line. Moving `SVC(probability=True)` to the new
API: unchanged reason (see the entry above).


## 2026-11-14: Split D, random artist-grouped splits, added between B' and C

**Decision.** `splits.make_split_D(seed)`, seeds 0-4: the 929 usable fault-filtered clips (the B'
and C pool), 442/197/290 clips, with every credit cluster (`credit_clusters`; a clip without an
artist is its own group) inside one part. Groups are placed largest first, each in a random part
(weighted by free room) among the parts where it keeps every genre within 2 clips of that part's
quota; if none allows that, in the part where it adds least to the squared distance from the
quotas. Then single-clip groups of different genres are swapped between parts while that lowers
the same distance. Every model runs on D with the same protocol as on the other splits.

**Reason.** The step B' -> C changed two things at once: it removed artist leakage, and it
swapped five random draws for one fixed published partition with its own valid and test clips
(C valid has 13 pop clips and 24 rock). A drop from B' to C could therefore come from either.
D keeps B''s pool, sizes and randomness and removes only artist leakage, so B' -> D measures
artist leakage alone and D -> C measures how the published partition differs from a random
artist-disjoint one. Balance: the worst genre is 3.6 clips off its quota on D (B': 0.8,
C: 6.6), and any two D seeds put 54-64% of clips in different parts (B': 56-62%).
A first version placed groups by cost alone; it put the biggest clusters in the same part
on every seed (any two seeds differed on only 25-30% of clips), so D's seed spread would have
understated split-to-split variation. The random placement fixed that. Known limit: Bob Marley
(30 reggae clips) fits only in train's reggae quota, so it is in train on every seed.

**Rejected.** `StratifiedGroupKFold` (folds of roughly equal size, not exactly 442/197/290);
`GroupShuffleSplit` (no genre stratification); replacing C as the main result with D (C is
the published split that prior work reports, and the project question fixes it as the main
result; D is context for reading C).

## 2026-11-14: SVM grid widened one step past the old corner (before the test freeze)

**Decision.** `grids.svm` is now C {0.1, 1, 10, 100, 1000} x gamma {scale, 0.0001, 0.001,
0.01, 0.1} on every split, and the leakage lab uses the same 25 settings.

**Reason.** The old best on C (C = 100, gamma = 0.001) was on the grid corner, so the SVM
might have been under-tuned, which would overstate the CNN's lead. Widening a grid on the
valid part is ordinary tuning: test has never been loaded. Result on C: the same setting
wins (0.458), now with untried neighbours on both sides, so the old corner was a real optimum
and not a cut-off. This replaces the "not widened" note in the SVM entry of 2026-10-30.

**Rejected.** Leaving the grid fixed (an examiner can fairly call a corner optimum
under-tuned); widening only on C (the splits must share one protocol).

## 2026-11-14: Test protocol, fixed before the pre-test-freeze tag

**Decision.** After the tag, `python scripts/evaluate_part.py --part test` runs once and scores two
sets of models, both already fixed:

1. **Final models (the headline).** On split C, every model with its chosen settings refit on
   train+valid (639 clips). Classical models: the same hyperparameters, scaler refit on
   train+valid. XGBoost: the same hyperparameters and the same early-stopped tree count. CNN:
   each run seed retrained (`src.train_cnn --refit_epochs`) for its own best epoch count (see the
   2026-11-14 entry on `min_epochs`; before it 8, 26 and 13 for seeds 0, 1, 2), normalisation statistics from train+valid, torch seed = seed + run
   seed as before. **Headline: the final CNN's clip macro-F1 on C test, mean over its three run
   seeds, with a 95% artist-cluster bootstrap CI** (2,000 resamples of credit clusters, the three
   runs averaged inside each resample). Beside it: the final SVM, and the paired difference
   CNN - SVM with its CI and the share of resamples in which the CNN is better.
2. **Ladder models.** The selected models exactly as trained (train only) on every split: A,
   B (5 seeds), B' (5), D (5), C; the CNN with one run per split seed, three on C. These answer
   the leakage question with one identical protocol on every rung, and their test scores can be
   compared with their valid scores. Reported: mean over seeds with a CI, sd over seeds, and the
   drops A - B (within-clip leakage), B - B' (duplicates and faulty clips), B' - D (artist
   leakage), D - C (published partition against random artist-disjoint draws), B' - C.

Also reported for both sets on C: per-genre F1 with clip counts, confusion matrices, and
**C-strict**: C test without the 25 clips whose credit cluster has a clip in C train.

**Rules after the tag.** Nothing chosen on valid changes: models, settings, window length,
vote rule, the web app's 0.7 "Not sure" threshold and its served model. If a bug turns up after
the test run, it is fixed in a commit marked post-freeze, and both the original and the
corrected numbers are reported. The tag never moves.

**Reason for refitting the headline models.** Chosen by the project owner: the final models
should use all 639 non-test clips, 45% more than train alone, which matters on a dataset this
small. The settings, tree count and epoch count are all fixed from validation before the
test part is read, so the refit adds training data but no new choice made after seeing test.

**Reason for keeping the ladder train-only.** Refitting the CNN on every split would take
19 extra trainings (about 7 hours on this CPU) for comparisons that need only to be equal
across splits, not as strong as possible. Train-only ladder models are what was validated.

**Rejected.** Refitting everything (time, see above); the best single CNN seed as the
headline (choosing the best of three on valid is optimistic); scaling the epoch or tree count
up with the larger training set (one more rule invented without a way to check it); scoring
test more than once.

## 2026-11-14: Macro-F1 averages over the genres with a true clip (bootstrap fix, before the test freeze)

**Decision.** `evaluate.macro_f1` averages per-genre F1 over the genres present in `y_true`
(sklearn `labels=np.unique(y_true)`), not over genres present in `y_true` or `y_pred`.

**Reason.** In an artist-cluster bootstrap, a resample can miss every artist of a genre. C valid
hip-hop is a single credit cluster, and blues and jazz have three each, so 45% of resamples lack
at least one genre. sklearn's default then gives the missing genre F1 = 0 whenever the model
predicts it for some clip, which pulls the resampled scores down: on C valid the SVM's resamples
averaged 0.434 against a point estimate of 0.458, and CNN seed 2's 0.581 against 0.635, so the
intervals sat almost entirely below the point estimate. With the fix the SVM's resample mean is
0.457. The false positives still count, as lost recall of the true genres. Every full valid
part contains all 10 genres, so no stored score changes: 108 of the 109 rebuilt valid prediction
files give the identical score, and the one exception was a draft file written before its SVM
was retuned. Also: C's per-genre F1 for hip-hop measures one artist, and the CI widths reflect
genres that rest on 1-3 artists. That is a limitation of the data, now stated on the site.

**Rejected.** Bootstrapping within genre strata (with one hip-hop cluster that stratum would
never vary, which understates uncertainty); keeping sklearn's default (a downward bias in every
artist-grouped interval, worst on C).

## 2026-11-14: Last nine ladder CNN runs trained on the laptop GPU

**Decision.** The CNN runs on B' seeds 1-4 and D seeds 0-4 are trained on the laptop's RTX 4050
(torch 2.15.0+cu132, `--device cuda`). All other CNN runs, including every C run and the three
final C models refit on train+valid, were trained on CPU and are not retrained. Prediction
(`src/predict.py`) always runs on CPU.

**Reason.** Windows stopped the two CPU training jobs for low memory after about 2.5 hours,
with 9 ladder runs left (roughly 4-6 more hours on CPU). Each run's device is in its result file,
so the change is traceable. A GPU changes floating-point summation order, so a GPU run differs
from what the CPU would have produced for the same seed by about as much as a different seed
does; these runs are single draws of a split and seed either way. After the CUDA build was
installed, the stored C seed 2 CNN rebuilt on CPU to exactly its stored valid macro-F1 (0.634843).

**Rejected.** Continuing on CPU (hours more, with the same memory risk); Colab (the same
non-reproducibility across devices, plus copying files back by hand); retraining the earlier
CPU runs on GPU for uniformity (it would change the C runs the test protocol already fixed).

## 2026-11-14: CNN early stopping never before epoch 20; every CNN run retrained on the GPU

**Decision.** `cnn.min_epochs: 20`: training continues while `epoch < 20` even if valid clip
macro-F1 has not improved for `patience` (8) epochs; after that the rule is unchanged, and the
best-valid epoch is still kept. Every CNN run (A; B, B' and D seeds 0-4; C run seeds 0-2) and the
three final C models (refit on train+valid for each run's new best epoch count) were retrained
under it on the laptop GPU (`--device cuda`). Each result file records `patience`, `min_epochs`
and `device`. This replaces the previous entry's plan to keep the CPU-trained C runs.

**Reason.** Valid clip macro-F1 on 197 clips swings by up to 0.25 between consecutive epochs
(B' seed 4: 0.33, 0.59, 0.34, 0.50, 0.34, ...). With patience 8 alone, one early lucky peak
could end a run while the training loss was still falling: B' seed 4 kept epoch 2 (0.587) and
stopped at epoch 10, D seed 3 kept epoch 6, C seed 0 kept epoch 8. Those runs measure an
accident of the stopping rule, not the model. The test part is still locked, and the change
applies to every split and seed alike, so the ladder keeps one protocol. The GPU makes the
full retrain take about an hour instead of a day.

**Rejected.** Keeping the runs and reporting the problem (it biases the ladder by chance, and
the final C models' epoch counts come from these runs); patience 15 (an epoch-2 peak can still
win if nothing beats it, and every run gets slower); selecting the epoch by valid loss (every
other choice in the project is made on valid clip macro-F1, the reported metric).

## 2026-11-17: Retrain complete; B seed 3 retrained again; final C models refit on the GPU

**Decision.** B' seeds 1-4 and D seeds 0-4 were retrained under `min_epochs: 20` on the laptop GPU,
so all 19 ladder CNN runs now use one rule. The three final C models were refit on train+valid on
the GPU for each run's best epoch (14, 14, 25), replacing the CPU refits (8, 26, 13 epochs).
B seed 3 was retrained once more: its stored score (0.8146, GPU) differed from the CPU rebuild
(0.8196) because one valid clip has a top-two margin of 0.0005 that flips between devices. The
new run (best epoch 26, 0.7903) rebuilds to the same score. B CNN mean: 0.787 -> 0.782. The other
18 runs rebuild exactly. The validation tables, summaries and analyses were rebuilt; the test
part was not touched.

**Reason.** `scripts/evaluate_part.py --part valid` refuses any run whose rebuilt score differs
from its stored score, so the headline numbers always come from the saved weights.

**Rejected.** Editing the stored score by hand or loosening the check (it would hide real
mismatches); keeping the old B seed 3 run (it would stop the validation rebuild).
