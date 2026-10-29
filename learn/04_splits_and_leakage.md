# Lesson 04: Splits and leakage

How the data are divided into train, validation and test decides what a score
*means*. This lesson builds four splits that are identical in size and differ
only in what can leak from training into testing, so their scores can be
compared to measure each kind of leakage.

---

## 1. What we built

| File | What it does |
|---|---|
| `src/fetch_meta.py` | Downloads `seungheondoh/gtzan-bind` (Hugging Face), keeps `track_id, path, title, artist_name, album, fault_filtered_split`, checks the mapping to our clip ids, writes `splits/gtzan_bind.csv`. |
| `src/splits.py` | `add_split_C_to_metadata`, `make_split_A/B/Bprime/C`, `load_split` (with the test lock). |
| `scripts/make_splits.py` | Builds everything and `results/split_summary.json`. |
| `splits/metadata.csv` | Now also has `artist` and `split_C`. |
| `splits/A.csv`, `B_seed{0-4}.csv`, `Bp_seed{0-4}.csv`, `C.csv` | One row per window of the 999 usable clips: `clip_id, window_idx, genre, part`. |
| `tests/test_splits.py` | Exact counts, whole clips per part, fault-filtered-only for B'/C, C equals the published split, the documented artist overlap, the test lock. |

```bash
.venv/Scripts/python -W ignore -m src.fetch_meta        # once; needs internet
.venv/Scripts/python -W ignore scripts/make_splits.py   # or: make splits
.venv/Scripts/python -m pytest -q tests/test_splits.py
```

**Mapping gtzan-bind to our files.** gtzan-bind's `path` is
`reggae/reggae.00052.wav`; its stem `reggae.00052` is exactly our `clip_id` and
equals gtzan-bind's own `track_id`. `fetch_meta.py` asserts all 1000 rows map
one-to-one onto our 1000 files.

---

## 2. Concepts in depth

### 2.1 The i.i.d. assumption

Almost every guarantee in machine learning assumes training and test examples
are **independent and identically distributed** (i.i.d.): each drawn
separately from the same distribution. Then the test score is an unbiased
estimate of the score on new data from that distribution:

```latex
\mathbb{E}\big[\widehat{\text{err}}_{\text{test}}\big] = \text{err}_{\mathcal{D}}(f)
\quad \text{when test examples are i.i.d. from } \mathcal{D} \text{ and independent of training.}
```

GTZAN violates independence in three ways:
- the 10 windows of one clip are cut from the same recording;
- some clips are copies (14 byte-identical pairs) or excerpts of the same song;
- many clips share an artist (Bob Marley & The Wailers has 26 reggae clips in
  C's train part alone).

When test examples are *dependent* on training examples, the estimate is
biased upwards: the model is partly tested on things it has effectively seen.

### 2.2 What data leakage is

**Leakage**: information reaches the model during training (or model
selection) that will not be available when it is really used, so the measured
score exceeds the real one. Two families:
- *Leakage through the split*: test examples that are near-copies of training
  examples (this lesson).
- *Leakage through preprocessing or selection*: scalers fitted on all data,
  hyperparameters chosen on test (handled by Pipelines and the test lock).

### 2.3 Group splitting: split by the unit you want to generalise across

Ask: *what will be new when the model is used?* For a genre classifier: new
songs by new artists. So whole songs, and ideally whole artists, must sit on
one side of the split. Formally, give each example a group label $g_i$
(clip, song, artist) and require

```latex
g_i \ne g_j \quad \text{for every } i \in \text{train},\ j \in \text{test}.
```

The generalisation you can claim is exactly across the unit you split by.
Split by window → you only claim generalisation to new windows of known clips.
Split by artist → new artists.

### 2.4 Three kinds of leakage in GTZAN

1. **Within-song (window) leakage.** Adjacent 3 s windows of one clip share
   instruments, mix, key and tempo; our tempo feature is even *identical*
   across a clip's 10 windows. A test window whose siblings are in training is
   almost a training example.
2. **Repeat leakage.** Byte-identical files and different excerpts of the same
   recording. A test clip whose copy is in training is a training example.
3. **Artist leakage.** An artist's voice, band line-up, studio and mastering
   are shared across their songs. A model can learn "this sounds like Bob
   Marley's records → reggae", which fails for an unknown artist.

### 2.5 The ladder A → B → B' → C

All four splits use **442 train / 197 valid / 290 test clips** (A: ×10
windows). Measured on each test part (`results/split_summary.json`):

| Split | Built how | Test clips also in train | Test clips with a byte-identical file in train | Test clips sharing an artist with train |
|---|---|---|---|---|
| A | random windows of the 999 usable clips | **970 of 974** | 1 | 959 (98.5%) |
| B (seeds 0-4) | random clips of the 999 usable, genre-stratified | 0 | **5, 3, 2, 4, 4** | 69.0-75.5% (mean 72.1%) |
| B' (seeds 0-4) | random clips of the 929 usable fault-filtered | 0 | **0** | 68.3-72.8% (mean 70.8%) |
| C | published fault-filtered, artist-filtered split | 0 | 0 | **22 of 290 (7.6%)** |

What each step removes:
- **A → B** removes within-clip leakage: no test clip has windows in training
  (970 → 0).
- **B → B'** removes exact repeats and Sturm's faulty clips (2-5 → 0
  identical-file test clips) while keeping artist leakage (~71%).
- **B' → C** removes most artist leakage (~71% → 7.6%) with the same pool of
  clips and the same sizes. B' and C contain *exactly the same 929 clips*;
  only the grouping differs. So the score difference B' − C isolates the
  effect of artist leakage.

*Worked check of the repeat numbers.* In lesson 03 we predicted that a random
split would put about $14 \times 2 \cdot \frac{443}{999} \cdot \frac{290}{998} \approx 3.6$
duplicate pairs across train and test. With 442 train clips the prediction is
still 3.6. The five B seeds give 5, 3, 2, 4, 4: mean **3.6**.

**C is not perfectly artist-disjoint by gtzan-bind's names.** 13 artists
cross parts (Anthrax, Bob Marley, Boney M., Burning Spear, Danzig, Dark
Tranquillity, Earth, Wind & Fire, Freddie McGregor, a Juilliard String Quartet
credit, KC and the Sunshine Band, Metallica, Peter Tosh, Slayer; 74 clips).
The published split was made with Sturm's own artist labels; gtzan-bind's names
appear to come from automatic matching, and the two can disagree. We kept the
published split for comparability and report the 7.6% honestly (DECISIONS.md).
Also, 6 of C's 290 test clips have no artist name, so "7.6%" counts only
known artists.

### 2.6 Why identical sizes isolate leakage from training size

Test accuracy depends on both leakage and the amount of training data (more
data → better models, the learning curve). If C had 300 training clips and B
had 700, a lower score on C would mix both effects. With identical sizes:

```latex
\text{score}(B') - \text{score}(C) \approx \text{effect of artist leakage}
```

because nothing else systematically differs. The 5 seeds of B and B' show how
much the score moves just from which clips happen to be drawn, so we know how
big a difference must be before it means anything.

**The jazz.00054 adjustment.** Published C is 443/197/290, but jazz.00054 (in
C's train part) cannot be loaded. B' is drawn from the same 929 usable clips
and cannot supply 930, so all splits use 442/197/290.

### 2.7 Stratification

Stratified sampling draws each genre in proportion to its share of the pool, so
every part has (almost) the same genre mix:

```latex
n_{g,\text{part}} \approx n_{\text{part}} \cdot \frac{N_g}{N}
```

Without it, a random test part of 290 could easily have 20 clips of one genre
and 38 of another, adding noise to per-genre F1 and to macro-F1. Real numbers:
- B seed 0 test: **29 of every genre**; train 44 or 45.
- C test (published, chosen by artist, not stratified): 26 (reggae) to 32
  (rock); train 41 (pop) to 48 (classical).
- B' seed 0 test: 26 to 31, because the fault-filtered pool itself is
  unbalanced (jazz and pop lost more clips to filtering).

We stratify with two calls of `sklearn.model_selection.train_test_split(...,
stratify=genre)`: pool → train + rest, then rest → valid + test.

### 2.8 The test lock, and why the test set is used once

Every time you look at a test score and then change something (a feature, a
hyperparameter, the model), the test set becomes part of model selection. With
enough looks you fit the test noise: if you try $k$ models and keep the best on
test, the expected best score rises with $k$ even if all models are equally
good. The honest procedure:

1. Choose everything on **valid** (clip-level macro-F1).
2. Freeze code and choices; create the git tag `pre-test-freeze`.
3. Evaluate on test **once** and report that number.

`load_split(name, "test")` raises `PermissionError("test set locked until
pre-test-freeze tag")` unless `allow_test=True` **and** `git tag -l
pre-test-freeze` returns the tag. Two keys: a code flag (so tests can't be
loaded by accident) and a git tag (a public, timestamped record that choices
were frozen before the test was opened).

---

## 3. Why this choice, and the alternatives we rejected

| Choice | Alternative rejected | Why |
|---|---|---|
| Published C as main split | Rebuild an artist-disjoint split from gtzan-bind names | Comparability with prior work; gtzan-bind names may be wrong. Residual 7.6% is reported. |
| Same sizes 442/197/290 everywhere | Natural sizes per split | Isolates leakage from training size. |
| B' from exactly C's clip pool | B' from a different subset | Then B' − C differs only in grouping. |
| 5 seeds for B and B' | 1 seed | Shows split-to-split variance. |
| Stratify B and B' by genre | Plain random | Stable per-genre counts; less noise. |
| One window-level file per split | Clip-level files | Same format everywhere; tests can check that windows of a clip share a part. |
| Lock with flag + git tag | Honour system | Mistakes are blocked in code; the tag is evidence. |
| Keep jazz.00054 row with `split_C = train`, `load_ok = False` | Delete it | The published label stays visible and auditable. |

---

## 4. Code walkthrough

`src/fetch_meta.py`
- `map_to_clip_id(path)`: `Path(path).stem` turns `reggae/reggae.00052.wav` into
  `reggae.00052`.
- `main`: three asserts guard the mapping: clip ids unique and 1000 rows; the
  set equals our audio files; the stem equals gtzan-bind's own `track_id`.

`src/splits.py`
- Line 24, `add_split_C_to_metadata`: merges `artist` and `split_C` into
  `metadata.csv`. `validate="1:1"` (line 29) makes pandas raise if any clip
  matched twice. Line 31 asserts the published counts 443/197/290.
- Line 41, `_expand`: `index.repeat(10)` and `np.tile(arange(10))` turn one row
  per clip into ten rows per window. A clip's part is copied to all its
  windows, so B, B', C cannot split a clip by construction.
- Lines 48-60, `_stratified_parts`: if the pool is bigger than 929 (B: 999),
  line 54 first picks 929 clips stratified by genre; lines 55-58 split those
  into train (442), then valid (197) and test (290), each stratified. Leftover
  clips are `unused`.
- Lines 63-72, `make_split_A`: shuffles all 9,990 window rows with
  `np.random.default_rng(seed).permutation` (line 67) and cuts the permutation
  at 4,420 / 6,390 / 9,290 (line 69); the remaining 700 windows are `unused`.
  Windows of one clip end up in different parts: that is the point of A.
- Lines 84-92, `make_split_Bprime`: same as B but only rows with
  `split_C` non-missing enter the draw.
- Lines 95-101, `make_split_C`: `part = split_C`, with `unused` for the 70 clips
  outside C. jazz.00054 is not among the usable clips, so it has no rows.
- Lines 104-107, `is_test_unlocked`: runs `git tag -l pre-test-freeze`; any git
  error counts as locked. (Named `is_...` so pytest does not mistake it for a test.)
- Lines 114-119, `load_split`: the lock is checked *before* the file is read.

`scripts/make_splits.py` → `summarise`: for each split, the set of clips per
part (a clip is "in" a part if any of its windows is), genre counts, and three
test-part leakage measures: clip also in train, byte-identical file (md5) in
train, artist in train (missing artists never count as shared).

---

## 5. Pitfalls

1. **Splitting windows (A) and reporting it as the result.** 970 of 974 test
   clips have siblings in training; the score measures memory.
2. **Random clip split on GTZAN (B).** Duplicates and ~72% shared artists make
   the score optimistic for new music.
3. **Comparing splits of different sizes.** Leakage and data size get mixed up.
4. **Assuming the published split is perfect.** Our check found 13 crossing
   artist names; always measure, don't assume.
5. **Treating a missing artist as a unique artist.** 6 test clips in C have no
   name; they could belong to a training artist. We count only known names and
   say so.
6. **Peeking at test "just once" during development.** Each look turns test
   into validation. The lock makes this an explicit act.
7. **Using `cross_val_score` with default k-fold over windows.** That silently
   builds split A inside every fold.

---

## 6. Results from this step

- `splits/metadata.csv`: `split_C` counts **443 / 197 / 290** (published), 70
  clips outside C. **jazz.00054 is in C (train)**; it stays with
  `load_ok = False`, so usable C is **442 / 197 / 290**.
- Split files: `A.csv` (4,420 / 1,970 / 2,900 windows, 700 unused),
  `B_seed0-4.csv`, `Bp_seed0-4.csv`, `C.csv` (442 / 197 / 290 clips).
- Leakage on the test part (`results/split_summary.json`): see the table in
  2.5. Shared-artist share: **A 98.5%, B 69.0-75.5%, B' 68.3-72.8%, C 7.6%**.
  C is **not 0** with gtzan-bind's artist names (13 crossing artists, 22 test
  clips); see DECISIONS.md for why we kept it.
- `pytest -q`: **31 passed** (including the lock: `load_split("C", "test")`
  raises `PermissionError`).

---

## 7. Viva questions with model answers

**Q1. Why not split the 9,990 windows randomly?**
Windows of one clip are near-copies; with a window split 970 of 974 test clips
have windows in training. The score would measure recognition of seen
recordings, not genre.

**Q2. What does B' add over B?**
It removes exact duplicates and faulty clips while keeping artist overlap. B
puts 2-5 test clips with an identical file in train; B' puts 0. B − B'
estimates the effect of duplicates/faults.

**Q3. Why must B' and C have the same clip pool?**
So that their only difference is how clips are grouped (random vs by artist).
Then B' − C estimates artist leakage alone.

**Q4. Your main split still has 7.6% shared artists. Is that a problem?**
It is a limitation we report. It is an order of magnitude below B/B' (~71%),
it is the published split used in prior work, and the overlap may partly come
from gtzan-bind's automatic artist names rather than real overlap. A strict
variant dropping the 74 clips is a possible sensitivity check.

**Q5. What if you didn't keep sizes identical?**
A lower score on C could be blamed on less training data; the comparison would
no longer isolate leakage.

**Q6. Why 5 seeds for B and B' but one C?**
C is fixed (published). For random splits, the seeds show how much scores
change from the draw alone, which tells us whether B' − C is bigger than noise.

**Q7. Why is the test set locked with a git tag?**
The tag is a timestamped public record that all choices were frozen before
test evaluation; the code flag prevents accidental loading. Together they make
"test used once" checkable.

**Q8. What if you stratified C?**
C is defined by artists; re-balancing genres would move clips between parts
and break artist-disjointness and comparability. We accept its 26-32 test
clips per genre; macro-F1 weights genres equally anyway.

**Q9. What would happen with k-fold cross-validation on windows for tuning?**
Each fold would be split A internally; hyperparameters would be chosen for
memorisation. We tune on the valid part of the chosen split instead.

---

## 8. Further reading

- Kaufman, Rosset, Perlich & Stitelman (2012), *Leakage in data mining:
  formulation, detection, and avoidance*, ACM TKDD. https://doi.org/10.1145/2382577.2382579
- Flexer (2007), *A closer look on artist filters for musical genre
  classification*, Proc. ISMIR 2007 (the original argument for artist filters).
- Sturm (2013), *The GTZAN dataset: its contents, its faults, their effects on
  evaluation, and its future use*. https://arxiv.org/abs/1306.1461
- Kereliuk, Sturm & Larsen (2015), *Deep learning and music adversaries*
  (the fault-filtered split). https://arxiv.org/abs/1507.04761
- gtzan-bind dataset card. https://huggingface.co/datasets/seungheondoh/gtzan-bind
- scikit-learn, *Cross-validation: grouped data* (`GroupKFold`,
  `StratifiedGroupKFold`). https://scikit-learn.org/stable/modules/cross_validation.html#group-k-fold
