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
