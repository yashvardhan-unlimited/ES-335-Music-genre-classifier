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
