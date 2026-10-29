# Lesson 03: The data, and GTZAN's faults

We download GTZAN, check every file, and look at it before any modelling. The
audit is what lets us say "999 usable clips" with evidence, and the faults we
find here explain why the main result uses the artist-disjoint split C.

---

## 1. What we built

| File | What it does |
|---|---|
| `scripts/download_data.sh`, `scripts/download_data.ps1` | `kaggle datasets download -d andradaolteanu/gtzan-dataset-music-genre-classification -p data/raw --unzip`; skip if the data is already there. |
| `config.yaml` → `paths.audio_dir` | `data/raw/Data/genres_original`, where the unzip put the 1000 wav files. |
| `src/data.py` | `list_clips`, `audit_clip`, `build_metadata`. |
| `splits/metadata.csv` | One row per file (1000): `clip_id, genre, path, load_ok, n_samples, duration_s, md5, peak, clip_share, rms_db`. |
| `results/load_errors.csv` | Files that failed to load, with the error. |
| `scripts/eda.py` | Six figures in `results/figures/`, `results/data_summary.json`, and `results/eda_descriptors.csv` (per-clip centroid and tempo). |
| `tests/test_data.py` | Tests on real files: `blues.00000`, `classical.00000` (good) and `jazz.00054` (corrupt). Skipped if the data is not downloaded. |

Commands (Windows paths; `make data` runs all three):

```bash
bash scripts/download_data.sh                  # or: powershell -File scripts/download_data.ps1
.venv/Scripts/python -W ignore -m src.data     # ~30 s for 1000 files
.venv/Scripts/python -W ignore scripts/eda.py  # ~1 min
.venv/Scripts/python -m pytest -q
```

The download worked without a `kaggle.json`. If Kaggle asks for credentials,
the token goes in `%USERPROFILE%\.kaggle\kaggle.json` (Linux/macOS:
`~/.kaggle/kaggle.json`), and it is git-ignored.

---

## 2. Concepts in depth

### 2.1 What GTZAN is and how it was built

GTZAN is the dataset of Tzanetakis & Cook (2002), the paper that made
automatic genre classification a standard task. George Tzanetakis collected
1000 excerpts of 30 s, 100 per genre for 10 genres, in 2000-2001 from his
personal CDs, radio and microphone recordings. Files are 22,050 Hz, mono, 16-bit
(originally `.au`; the Kaggle copy is `.wav`). We confirmed the format of
`blues.00000.wav`: `samplerate: 22050 Hz, channels: 1, subtype: Signed 16 bit PCM`.

Important consequences of *how* it was built:
- One person chose the excerpts and assigned the labels, with no documented
  protocol. Labels reflect one listener's judgment in 2001.
- Excerpts were taken from whatever recordings were at hand, so some albums
  and artists contribute many clips, and some recordings appear more than once.
- No artist, album or track metadata was shipped. Sturm later identified most
  of it by listening (Sturm 2013).

### 2.2 Licence: why audio is never republished

GTZAN has no open licence (Kaggle shows `License(s): other`). The excerpts are
from commercial, copyrighted recordings; the dataset is circulated for
research use. So we download it for our own analysis but never put audio in
git or on the website (`.gitignore` blocks `*.wav`, `*.au`, `*.mp3` and
`/data/`). The site shows figures and numbers only.

### 2.3 Digital audio basics

**Sampling.** A microphone signal is continuous. Digital audio stores its value
$sr$ times per second. GTZAN uses $sr = 22{,}050$ Hz. By the Nyquist theorem,
the highest frequency that can be represented is

```latex
f_{\max} = \frac{sr}{2} = 11{,}025\ \text{Hz}
```

so anything above ~11 kHz (cymbal shimmer, "air") is lost. That is why the
mel plots stop near 11 kHz.

**Mono.** One channel. Stereo width cues, which differ between genres and eras,
are not available to the model.

**Amplitude.** Each sample is a 16-bit integer in $[-32768, 32767]$. Loaders
divide by 32768, so a sample $x$ lies in $[-1, 1)$. "Full scale" means
$|x| = 1$; a signal that would go beyond it is cut off (**clipped**), which
flattens wave peaks and adds distortion.

**Decibels.** Hearing is roughly logarithmic, so levels are given in dB
relative to full scale (dBFS):

```latex
L_{\text{dBFS}} = 20 \log_{10} a, \qquad a \in (0, 1]
```

0 dBFS is the loudest possible; −6 dB is half the amplitude; −20 dB is a tenth.

**RMS.** The root-mean-square of the $N$ samples measures average level
(loudness), unlike the peak, which is a single extreme sample:

```latex
\mathrm{RMS} = \sqrt{\frac{1}{N}\sum_{n=1}^{N} x_n^2}, \qquad
\texttt{rms\_db} = 20 \log_{10}\mathrm{RMS}
```

*Worked example, `blues.00000`* (from `splits/metadata.csv`):
`n_samples = 661,794`, so duration $= 661794 / 22050 = 30.013$ s;
`peak = 0.885` ⇒ $20\log_{10} 0.885 = -1.06$ dBFS; `rms_db = -17.03` ⇒
$\mathrm{RMS} = 10^{-17.03/20} = 0.141$. Peak over RMS (the crest factor) is
$-1.06 - (-17.03) \approx 16$ dB: a fairly dynamic recording.

**Spectral centroid** (used in one EDA figure). For a frame with magnitude
spectrum $|X_k|$ at frequencies $f_k$, the centroid is the "centre of mass" of
the spectrum, perceived as brightness:

```latex
c = \frac{\sum_k f_k\, |X_k|}{\sum_k |X_k|}
```

We average it over all frames of the clip.

**Log-mel spectrogram** (one EDA figure; the CNN's input later). Short-time
Fourier transform with `n_fft = 2048`, `hop_length = 512`, mapped onto 128 mel
bands (a frequency scale spaced like human pitch perception), then converted
to dB with `power_to_db(ref=1.0)`. A 30 s clip of 661,794 samples gives
$1 + \lfloor 661794 / 512 \rfloor = 1293$ frames.

### 2.4 What each audit column means

| Column | Meaning | Why we record it |
|---|---|---|
| `clip_id` | File stem, e.g. `blues.00000` | Join key for every later table. |
| `genre` | Parent folder name | The label. |
| `path` | Relative path to the wav | Portable across machines. |
| `load_ok` | `librosa.load` succeeded and returned samples | Only these clips are used downstream. |
| `n_samples` | Number of samples after loading at 22,050 Hz | Exact length; needed for windowing. |
| `duration_s` | `n_samples / 22050` | Check the "30 s" assumption. |
| `md5` | Hash of the file's bytes | Byte-identical files have the same md5 → exact duplicates; also a fingerprint of our copy. |
| `peak` | $\max_n |x_n|$ | Reaching ~1 suggests limiting or clipping. |
| `clip_share` | Fraction of samples with $|x| \ge 0.999$ | Amount of clipping. |
| `rms_db` | Loudness in dBFS | Level differences between genres (production). |

### 2.5 What our numbers show

All numbers from `results/data_summary.json`, `splits/metadata.csv` and
`results/eda_descriptors.csv`.

**Loading.** 1000 files, **999 load**. The single failure is **jazz.00054**
(`NoBackendError` in `results/load_errors.csv`). Looking at its bytes: a valid
wav starts with `RIFF....WAVEfmt`; jazz.00054 starts directly with sample data
(`cb15 1e16 ...`). The header is missing, so no decoder knows the format. Jazz
therefore has 99 clips, every other genre 100.

**Durations.** Not all exactly 30 s. Three lengths dominate: 661,794 samples
(30.013 s, 640 clips), 661,504 (30.000 s, 295 clips), 675,808 (30.649 s, 10
clips). Range: **29.932 s** (hiphop.00032) to **30.649 s**. **9 clips are
shorter than 30 s**: classical.00049, classical.00051, country.00003,
country.00004, country.00007, disco.00014, hiphop.00031, hiphop.00032,
rock.00038. Ten 3 s windows need $10 \times 66150 = 661{,}500$ samples, so these
9 clips are short by at most 1,500 samples (0.07 s). The windowing step must
handle this (pad or shorten the last window), not crash or silently drop them.

**Clipping.** No clip has more than 1% of its samples clipped (the
`clip_share_over_1pct` list is empty). The maximum is 0.95% (hiphop.00019).
180 of 999 clips have at least one sample at $|x| \ge 0.999$; 18 have more than
0.1% (pop 14, hiphop 2, reggae 2). Share of clips with any clipping per genre:
pop 69%, metal 31%, hiphop 23%, rock 16%, blues 12%, disco 10%, reggae 10%,
country 9%, classical 0%, jazz 0%.

**Loudness.** Median `rms_db`: pop −13.3, hiphop −14.1, metal −16.9, rock
−17.3, blues −17.5, disco −17.6, reggae −18.3, country −18.4, jazz −21.2,
classical −28.0 dBFS. Quietest clip classical.00080 (−42.4), loudest pop.00062
(−7.8).

**Exact duplicates.** 14 pairs of files are byte-identical (same md5):
13 within one genre (metal 7 pairs, pop 2, hiphop 2, disco 1, reggae 1) and
**one across genres: metal.00058 = rock.00016**. The same audio carries two
different labels.

**Centroid and tempo.** Median spectral centroid: classical 1345 Hz, blues
1631, jazz 1698, country 1767, reggae 2070, rock 2279, hiphop 2500, disco 2546,
metal 2683, pop 3144. Median estimated tempo ranges from 103 BPM (hiphop) to
136 BPM (reggae), with most genres between 112 and 123 BPM.

### 2.6 GTZAN's known faults (Sturm 2013) and why each breaks evaluation

Sturm listened to all 1000 excerpts and documented four kinds of fault.

1. **Repetitions.** Exact repetitions (the same excerpt twice) and recording
   repetitions (different excerpts of the same song). *Why it breaks
   evaluation:* if one copy is in training and the other in test, the model is
   tested on something it has already seen. Our md5 check found 14 exact pairs;
   md5 only catches byte-identical files, so different excerpts of one song are
   not detected here.

   *Worked example.* With a random clip split (B: 443 train / 290 test drawn
   from 999 clips), the chance a given pair is split with one copy in train and
   one in test is about $2 \cdot \frac{443}{999} \cdot \frac{290}{998} \approx
   0.258$. Over our 14 exact pairs that is $14 \times 0.258 \approx 3.6$ pairs
   expected to leak straight from train to test, and that is only the
   byte-identical ones.

2. **Mislabelings.** Excerpts whose label is doubtful or wrong (e.g. a track in
   one genre folder that most listeners would put elsewhere). metal.00058 =
   rock.00016 is a case we found ourselves: one audio file, two labels. *Why
   it breaks evaluation:* a mislabelled test clip counts a correct prediction
   as an error, and a mislabelled training clip teaches the wrong mapping.
   Accuracy near 100% on GTZAN is therefore a warning sign, not an achievement.

3. **Distortions.** Clips with heavy clipping, dropouts or other recording
   damage. *Why it breaks evaluation:* a model can learn "distorted sound =
   genre X" if the damage concentrates in one genre, which is a property of
   the recordings, not of the music.

4. **Artist repetition.** Many clips per artist (some genres are dominated by a
   few artists). *Why it breaks evaluation:* under a random split the same
   artist is in train and test, so the model can identify the artist's voice,
   band or studio sound instead of the genre. Sturm and later Kereliuk et al.
   showed scores drop markedly once artists are kept disjoint. This is the
   leakage split C removes, and the gap between B and C is our second research
   question.

Our audit adds a fifth, smaller issue: **the corrupt file** (jazz.00054),
specific to this copy.

### 2.7 Reading every EDA figure

**`class_counts.png`** (bar chart, y-axis from 0). Grey = files, blue =
loadable. Every genre has 100 files; jazz has 99 loadable. The dataset is
balanced, so chance accuracy is ≈10%, and macro-F1 and accuracy will be close.

**`duration_hist.png`.** Almost all clips sit in two bins at 30.000 and 30.013
s; a small group at 30.649 s; a handful just under the red 30 s line. Read it
with the list of 9 short clips above.

**`clipping_by_genre.png`** (box plot, symmetric-log y-axis so that 0 and
small values are both visible). For most genres the box is flat at 0: most clips
have no clipped samples. Pop is the exception: its median clip has some
clipping and many outliers reach 0.1-0.9%; metal and hiphop have upper
whiskers or outliers above 0. Classical and jazz have none. Clipping is a
**mastering** artefact (the "loudness war" in pop production), so it
correlates with genre in GTZAN, but for production reasons, not musical ones.

**`rms_by_genre.png`** (box plot of loudness). Pop and hiphop are loudest with
narrow boxes (consistently compressed masters); classical is quietest and
widest (wide dynamics, varied recordings); jazz is second quietest. A model
could get some way just from loudness, which is why features should be
examined with this in mind (and why per-clip normalisation is a choice we will
have to justify later).

**`mel_per_genre.png`** (first loadable clip per genre, 0-30 s, mel axis up to
~11 kHz, colour −80 to +20 dB). What to look for:
- *classical.00000*: much less energy above ~2 kHz (blue region at the top),
  clear horizontal lines (sustained notes).
- *metal.00000*: dense, bright energy across the whole spectrum all the time
  (distorted guitars), no gaps.
- *disco.00000*, *hiphop.00000*, *pop.00000*: regular vertical stripes (drum
  hits on a steady beat) and strong low-frequency energy (bass, kick drum).
- *jazz.00000*: large changes over time, with quiet passages (blue columns).
- *rock.00000*: a clear change near 10 s, a reminder that one 3 s window may
  not represent the whole clip, which is why we average ten windows.
These are single examples, not genre averages; they show what kind of patterns
a CNN can see.

**`centroid_tempo_by_genre.png`** (two box plots).
*Centroid:* clear ordering, classical darkest, pop brightest; classical and
pop barely overlap. But blues, jazz and country overlap heavily, and so do
disco, hiphop and metal. One brightness number separates some genres, not all.
*Tempo:* the boxes overlap almost entirely; tempo alone is a weak genre cue.
Two reasons: tempo estimators make octave errors (reporting half or double the
true tempo), and the estimate is quantised, because librosa picks a lag in
whole frames. With 512-sample hops at 22,050 Hz, a lag of 21 frames gives
$60 \cdot 22050 / (512 \cdot 21) = 123.05$ BPM, which is why many medians sit at
exactly 123.

---

## 3. Why this choice, and the alternatives we rejected

| Choice | Alternative rejected | Why |
|---|---|---|
| Kaggle mirror via the `kaggle` CLI | Original marsyas.info tarball / Hugging Face copy | Scriptable one-liner; same files Sturm's analysis refers to (plus the known corrupt one). |
| Record jazz.00054 with `load_ok=False`, then exclude | Repair it, replace it, or drop it silently | See DECISIONS.md: repair is guesswork, replacing mixes sources, silent drop hides a fault. |
| Ignore Kaggle's CSVs and images | Use them as features | Unknown settings, unreproducible, rendered images are not spectrogram values; 3 s CSV encourages window leakage. |
| Audit with `librosa.load` | `soundfile.read` | Same loader as the feature step, so "loads in the audit" = "loads in the pipeline". |
| md5 of file bytes | Audio fingerprinting | Stdlib, exact, instant. Fingerprinting would also find recording repeats, but split C already handles those. |
| One file at a time | Load all audio into memory | 999 × 661,794 float32 samples ≈ 2.6 GB; per-file keeps memory flat. |
| `clip_share` threshold 0.999 | Exactly 1.0 | 16-bit full scale decodes to 32767/32768 ≈ 0.99997 on the positive side. |

---

## 4. Code walkthrough

`src/data.py`

- Line 13, `CLIP_THRESHOLD = 0.999`: one named constant rather than a magic number.
- Lines 16-22, `list_clips`: `Path(audio_dir).glob("*/*.wav")` finds files
  exactly one folder deep (genre folders). `sorted` makes the order
  deterministic (filesystem order is not guaranteed). `p.stem` gives
  `blues.00000`; `p.parent.name` gives the genre; `as_posix()` stores forward
  slashes so the CSV works on Linux too.
- Lines 25-54, `audit_clip`:
  - Line 33: md5 is computed from the raw bytes *before* trying to decode, so
    even the corrupt file gets a hash. `hashlib.file_digest` streams the file
    in chunks.
  - Lines 36-40: `librosa.load(path, sr=sr, mono=True)` inside `try`. librosa
    first tries soundfile (libsndfile), then falls back to audioread; for
    jazz.00054 both fail and we get `NoBackendError`. Catching `Exception`
    broadly is deliberate: different backends raise different types, and the
    error text is kept, not swallowed.
  - Line 45: RMS computed in float64 so squaring 661,794 values does not lose
    precision.
  - Line 51: `np.mean(a >= CLIP_THRESHOLD)`: the mean of a boolean array is the
    fraction of `True` values.
  - Line 52: `max(rms, 1e-10)` avoids `log10(0) = -inf` for a silent file.
- Lines 57-72, `build_metadata`: loops over files one by one, logs every 100,
  joins the file list and audits column-wise, writes the error table (line 69)
  and the metadata without the `error` column (line 70).

`scripts/eda.py`

- Lines 27-32, `log_mel`: every mel parameter comes from `config.yaml`.
- Lines 35-40, `centroid_tempo`: loads one clip, returns mean centroid and
  `librosa.feature.tempo`. Run with `joblib.Parallel(n_jobs=-1)`: each worker
  loads one file at a time, so memory stays small while using all cores.
- Lines 49 onward, `main`: reads only `metadata.csv` (no re-auditing), filters
  `load_ok`, draws the six figures, and writes `data_summary.json`. Duplicates
  are found with `meta.groupby("md5")` and groups of size > 1.

`tests/test_data.py`

- Module-level `skipif`: tests need the real data; on a fresh clone without
  data they skip instead of failing.
- `test_list_clips_ids_and_genres`: 1000 files, sorted, genre equals the
  clip-id prefix for every row.
- `test_audit_good_clip` (blues.00000, classical.00000): loads, ~30 s,
  `n_samples` matches duration, md5 matches an independent `hashlib.md5`,
  values in valid ranges.
- `test_audit_corrupt_clip_does_not_raise` (jazz.00054): returns
  `load_ok=False` with an error message and still has a 32-character md5.

---

## 5. Pitfalls

1. **Letting one corrupt file crash the run**, or worse, catching the error and
   dropping the file without a record. We keep a row and an error log.
2. **Assuming every clip is exactly 30 s.** 9 are shorter; code that slices
   ten 66,150-sample windows without checking would fail or produce a short
   last window silently.
3. **Using Kaggle's `features_3_sec.csv` with a random row split.** Rows from
   the same clip land in train and test: window leakage (split A), which
   inflates accuracy.
4. **Ignoring duplicates.** 14 byte-identical pairs, one with conflicting
   labels. Random splits leak them; a test set containing both labels of
   metal.00058/rock.00016 penalises any prediction.
5. **Learning production instead of genre.** Loudness and clipping differ by
   genre in GTZAN because of mastering. A model may use them; artist-disjoint
   evaluation limits how far this carries, because production is shared within
   an artist's albums.
6. **Committing audio or the Kaggle token.** Blocked by `.gitignore`; check
   `git status` before committing.
7. **Reading the mel figure as typical.** It shows one clip per genre. Use the
   box plots for genre-level statements.

---

## 6. Results from this step

- `splits/metadata.csv`: **1000 rows, 999 `load_ok`**. Failure: **jazz.00054**
  (missing RIFF header), as expected.
- Clips per genre (loadable): 100 each, jazz 99.
- Duration min **29.932 s**, max **30.649 s**; **9 clips < 30 s**.
- Clips with `clip_share > 1%`: **none** (max 0.95%, hiphop.00019).
- **14 byte-identical pairs**, including the cross-genre pair
  **metal.00058 = rock.00016**.
- Figures: `results/figures/class_counts.png`, `duration_hist.png`,
  `clipping_by_genre.png`, `rms_by_genre.png`, `mel_per_genre.png`,
  `centroid_tempo_by_genre.png`. Summary: `results/data_summary.json`.
- `pytest -q`: `6 passed` (2 utils tests, 4 data tests on real files).

---

## 7. Viva questions with model answers

**Q1. How many clips do you use and why not 1000?**
999. jazz.00054 in the Kaggle copy has no WAV header and cannot be decoded. We
keep it in metadata.csv with `load_ok=False` and exclude it downstream, rather
than guessing how to repair it.

**Q2. What is the highest frequency the models can use?**
11,025 Hz, half the 22,050 Hz sampling rate (Nyquist).

**Q3. Why record md5 hashes?**
They detect byte-identical files. We found 14 identical pairs; one pair has two
different genre labels. They also fingerprint our copy of the data so a grader
can confirm they have the same files.

**Q4. What if two identical clips land in train and test?**
The model is evaluated on audio it trained on, so the score overstates
performance. With a random split about a quarter of each pair would be split
that way (~3.6 of our 14 pairs expected). The fault-filtered split C removes
repetitions.

**Q5. Pop clips are louder and more clipped. Is that a problem?**
It is a real property of pop mastering, but it is a production cue rather than
a musical one, and it is shared by songs from the same artist or album. A model
that leans on it may do well on random splits and worse on artist-disjoint
splits, which is part of what we measure.

**Q6. Why does the tempo plot show so little difference between genres?**
Tempo varies more within genres than between them, tempo estimators make
octave errors, and librosa's estimate is quantised to frame lags (many values
at exactly 123.05 BPM). Tempo alone is a weak feature.

**Q7. Why not use Kaggle's precomputed feature CSVs?**
Unknown code and settings, so they are not reproducible or consistent with our
config, and the 3 s CSV invites window-level random splits, i.e. leakage.

**Q8. What if you loaded all audio into memory at once?**
About 2.6 GB of float32 for 999 clips; feasible on some machines, but one-at-a-
time processing keeps memory flat and lets a failure on one file be isolated.

**Q9. What if a clip is shorter than 30 s?**
Nine are, by up to 0.07 s. The windowing step must decide explicitly (pad the
last window or accept a slightly shorter one) and record the choice in
DECISIONS.md.

**Q10. Why is there no licence issue with your website?**
It shows only figures and numbers derived from the audio, never the audio
itself.

---

## 8. Further reading

- Tzanetakis & Cook (2002), *Musical genre classification of audio signals*,
  IEEE TSAP 10(5). https://doi.org/10.1109/TSA.2002.800560
- Sturm (2013), *The GTZAN dataset: Its contents, its faults, their effects on
  evaluation, and its future use*. https://arxiv.org/abs/1306.1461
- Kereliuk, Sturm & Larsen (2015), *Deep learning and music adversaries*.
  https://arxiv.org/abs/1507.04761
- Kaggle mirror used here:
  https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
- librosa documentation: `load`, `feature.melspectrogram`,
  `feature.spectral_centroid`, `feature.tempo`. https://librosa.org/doc/latest/
- Müller (2015), *Fundamentals of Music Processing*, Springer (chapters on
  audio representation and the STFT). https://www.audiolabs-erlangen.de/fau/professor/mueller/bookFMP
