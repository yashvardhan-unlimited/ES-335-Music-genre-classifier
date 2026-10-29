# Lesson 05: Audio features

Models do not hear audio; they get numbers. This step turns each clip into
(a) 57 handcrafted numbers per 3 s window for the classical models and (b) a
log-mel spectrogram per clip for the CNN. Every transform here is **fixed**: it
looks at one clip only, so it cannot leak information between train and test.

(Lesson 04, splits, is pending a decision about split C; its number is
reserved.)

---

## 1. What we built

| File | What it does |
|---|---|
| `src/features.py` | `load_clip`, `windows`, `clip_tempo`, `handcrafted_features`, `feature_names`, `clip_mel`, `clip_features`. |
| `scripts/extract_features.py` | Runs `clip_features` on all usable clips in parallel (joblib + tqdm), writes the outputs below, draws the pipeline figure. |
| `features/handcrafted.npy` | float32 array, one row per window, 57 columns. |
| `features/index.csv` | Row labels for that array: `clip_id, window_idx, genre, clip_padded`. |
| `features/feature_names.json` | The 57 column names in order. |
| `features/mel/<clip_id>.npy` | float16 log-mel of the full clip, shape 128 × 1292. |
| `results/figures/feature_pipeline.png` | Waveform → spectrogram → log-mel → MFCCs for one window. |
| `tests/test_features.py` | Window shape/order, 57 unique names, padding of a short clip, real-clip features finite, mel shape. |

```bash
.venv/Scripts/python -W ignore scripts/extract_features.py   # or: make features
.venv/Scripts/python -m pytest -q tests/test_features.py
```

`features/` is git-ignored (regenerable, ~350 MB); results and figures are committed.

---

## 2. Concepts in depth

### 2.1 Sampling and Nyquist (recap from lesson 03)

A clip is a sequence $x[n]$, $n = 0,\dots,N-1$, sampled at $sr = 22050$ Hz.
Frequencies above $sr/2 = 11025$ Hz cannot be represented. Every frequency axis
in this lesson therefore ends at 11,025 Hz.

We fix every clip to $N = 30 \times 22050 = 661{,}500$ samples (pad with zeros
or trim), then reshape into 10 windows of $66{,}150$ samples. Window $k$ is
samples $[66150k, 66150(k+1))$.

### 2.2 Framing and windowing (Hann)

A 3 s window is still too long to describe with one spectrum: notes change many
times per second. So we cut it into short overlapping **frames** of
`n_fft = 2048` samples (93 ms), moving by `hop_length = 512` samples (23 ms).
A 66,150-sample window gives $1 + \lfloor 66150/512 \rfloor = 130$ frames
(librosa pads the signal by `n_fft/2` at both ends, "centred" frames).

Cutting a frame out with hard edges creates artificial jumps at its ends, which
smear energy across all frequencies (**spectral leakage**, a different
"leakage" from data leakage). We multiply each frame by a **Hann window**,
which tapers smoothly to zero at both ends:

```latex
w[n] = 0.5 - 0.5\cos\!\left(\frac{2\pi n}{N_{\text{fft}}}\right), \qquad n = 0,\dots,N_{\text{fft}}-1
```

With hop = n_fft/4, Hann frames overlap by 75%, so every sample is covered
evenly.

### 2.3 The STFT and the time-frequency trade-off

The short-time Fourier transform takes the discrete Fourier transform of each
windowed frame $m$:

```latex
X[m, k] = \sum_{n=0}^{N_{\text{fft}}-1} x[n + mH]\, w[n]\, e^{-2\pi i k n / N_{\text{fft}}},
\qquad k = 0, \dots, N_{\text{fft}}/2
```

$H$ = hop length. Bin $k$ corresponds to frequency $f_k = k \cdot sr / N_{\text{fft}}$.
We keep the magnitude $|X[m,k]|$ (the phase is discarded).

*Numbers:* $N_{\text{fft}}/2 + 1 = 1025$ bins, spaced
$22050/2048 = 10.77$ Hz apart, each frame covering $2048/22050 = 92.9$ ms.

**Trade-off.** Frequency resolution is $sr/N_{\text{fft}}$, time resolution is
about $N_{\text{fft}}/sr$. Their product is fixed: a longer frame separates
close pitches better but blurs fast drum hits in time, and vice versa. 2048 at
22,050 Hz (≈93 ms, ≈11 Hz) is the standard compromise for music, fine enough
to separate semitones above ~200 Hz.

### 2.4 The mel scale and the filterbank

Human pitch perception is roughly linear below ~1 kHz and logarithmic above:
the step from 100 to 200 Hz sounds as big as 1000 to 2000 Hz. The mel scale
models this. The common (HTK) formula is

```latex
m(f) = 2595 \log_{10}\!\left(1 + \frac{f}{700}\right)
```

(librosa's default "Slaney" mel is exactly linear below 1 kHz and logarithmic
above; same idea.)

The **mel filterbank** is 128 overlapping triangular filters, equally spaced on
the mel scale from 0 to 11,025 Hz. Each mel band is a weighted sum of STFT
power bins:

```latex
M[m, b] = \sum_k F[b, k]\, |X[m, k]|^2
```

*Numbers from our config:* the lowest filter spans 0–52 Hz and covers 4 STFT
bins; the highest spans ~10.45–11.03 kHz and covers 53 bins. The 1025 bins
shrink to 128 bands: detail is kept where hearing is sensitive (low
frequencies) and pooled where it is not.

### 2.5 Log compression and decibels

Mel energies span many orders of magnitude, and loudness perception is
logarithmic. We convert power to decibels:

```latex
M_{\text{dB}}[m,b] = 10 \log_{10}\frac{\max(M[m,b],\ a_{\min})}{\text{ref}}
```

with `ref = 1.0` (a fixed reference for every clip) and $a_{\min} = 10^{-10}$
(so silence gives −100 dB, not −∞). librosa also applies `top_db = 80`:
values more than 80 dB below the clip's maximum are raised to that floor.

*Real example:* `features/mel/blues.00000.npy` ranges from −45.8 to +34.2 dB,
exactly 80 dB apart, so the floor is active on this clip's quietest bins.

`ref=1.0` vs `ref=np.max`: with `np.max`, each clip's loudest bin becomes 0 dB,
so a quiet classical recording and a loud pop master look equally loud. With
`ref=1.0` the absolute level stays in the numbers (DECISIONS.md).

### 2.6 DCT and MFCCs: why low coefficients describe timbre

A log-mel frame is a 128-value curve over frequency. Its broad shape (where
the energy sits, how fast it falls off) is the **spectral envelope**, which
our ears hear as **timbre** (what makes a guitar sound different from a
trumpet). Fine ripples on top come from individual harmonics (pitch).

The discrete cosine transform (DCT-II) expresses the curve as a sum of cosines
of increasing frequency *across the mel axis*:

```latex
c_j = \sum_{b=0}^{B-1} M_{\text{dB}}[b] \cos\!\left(\frac{\pi j (b + 0.5)}{B}\right), \qquad j = 0, \dots, 19
```

- $c_0$ (our `mfcc1`) is proportional to the average log energy: overall
  loudness.
- $c_1$ (`mfcc2`) measures the overall tilt: more energy low than high (large
  positive) or the reverse.
- Higher $c_j$ describe progressively finer wiggles of the envelope.

Keeping only the first 20 coefficients is a smoothing: it keeps the envelope
(timbre) and drops the harmonic fine structure (pitch), which is why MFCCs
describe "what instrument/production" more than "which notes". The DCT also
decorrelates the bands, which suits simple models.

*Real example (blues.00000, window 0):* `mfcc1_mean = −119.0` (overall level),
`mfcc2_mean = 125.1` (strong tilt: much more energy in low frequencies, as
the mel plot shows), `mfcc1_var = 2494` (loudness varies a lot frame to frame:
note attacks and pauses, visible in the waveform).

### 2.7 Chroma

Chroma folds all frequencies onto the 12 pitch classes (C, C♯, …, B),
ignoring octave: energy at 220 Hz and 440 Hz both go to A. Per frame it is a
12-value vector, normalised so its maximum is 1. It describes harmony and key.
As specified, we keep only the mean and variance **over all 12 bins and all
frames** (2 numbers), which captures "how concentrated vs spread out" the
harmonic content is rather than which notes. blues.00000 window 0:
`chroma_mean = 0.336`.

### 2.8 Spectral centroid, bandwidth, roll-off, ZCR, RMS

Per frame, with magnitude spectrum $S_k$ at frequencies $f_k$:

```latex
\text{centroid} = \frac{\sum_k f_k S_k}{\sum_k S_k}
\qquad
\text{bandwidth} = \left(\frac{\sum_k S_k (f_k - \text{centroid})^2}{\sum_k S_k}\right)^{1/2}
```

- **Centroid**: brightness (centre of mass of the spectrum).
- **Bandwidth**: spread around the centroid (narrow tone vs broadband noise).
- **Roll-off**: the frequency below which 85% of the energy lies (librosa
  default); another brightness measure, less sensitive to a few loud peaks.
- **Zero-crossing rate** (ZCR): fraction of consecutive samples that change
  sign; high for noisy, bright sounds (cymbals, distortion), low for smooth
  tones.
- **RMS**: frame loudness, $\sqrt{\frac{1}{N}\sum x_n^2}$ over the frame.

blues.00000 window 0: centroid mean 1773 Hz, bandwidth 1972 Hz, roll-off
3714 Hz, ZCR 0.082, RMS 0.130.

For each feature we keep the **mean** (typical value over the 130 frames) and
the **variance** (how much it changes within the window).

### 2.9 HPSS: harmonic-percussive source separation

On a spectrogram, sustained harmonic sounds (voice, chords) are **horizontal
lines** (stable in frequency over time); percussive hits are **vertical
lines** (broadband, brief). HPSS (Fitzgerald 2010) median-filters the
spectrogram horizontally (keeps horizontal structures → harmonic) and
vertically (keeps vertical → percussive), builds soft masks from the two, and
inverts each masked spectrogram back to a waveform. We then take the RMS
of each part. blues.00000 window 0: harmonic RMS 0.084, percussive RMS 0.058.
The ratio says how drum-driven a piece is (high percussive share: hiphop,
disco; low: classical).

We run HPSS once on the full 30 s and slice into windows: 10× less work, and
the median filters never see an artificial window edge.

### 2.10 Tempo estimation

`librosa.beat.beat_track` computes an **onset strength envelope** (how much
the spectrum increases from frame to frame, i.e. how likely a note or drum
hit starts there), finds the periodicity of that envelope via
autocorrelation, weighted towards ~120 BPM, and tracks beats with dynamic
programming. Tempo in BPM from a lag of $L$ frames:

```latex
\text{BPM} = \frac{60 \cdot sr}{H \cdot L}
```

Because $L$ is a whole number of frames, estimates are quantised: $L=21$ gives
$60 \cdot 22050 / (512 \cdot 21) = 123.05$ BPM and $L=20$ gives 129.2 BPM. Our
blues.00000 clip-level tempo is **123.05**; per-window estimates on the same
clip were 129.2 or 123.0. On this clip they happen to be stable; on clips with
fewer clear beats a 3 s window (≈6 beats at 120 BPM) gives much less to work
with, and octave errors (half or double tempo) become common. Hence one tempo
per clip, copied to all 10 windows.

### 2.11 Fixed transforms vs learned statistics: why only the latter can leak

A **fixed transform** computes a clip's features from that clip alone, with
constants chosen in advance (n_fft, 128 mels, ref=1.0). Whether a clip is in
train or test makes no difference to its features, so no information moves
between parts.

A **learned statistic** is computed from many clips: a mean and standard
deviation for standardisation, PCA directions, a feature-selection ranking,
`ref=np.max` across the dataset, a global dB normalisation for the CNN. If it
is computed on all 999 clips, the test clips influence the transform that is
applied to train, and the model is tuned to the test distribution. This is
leakage, even though no label is used.

Rule: this step contains **only** fixed transforms. Every learned statistic is
fitted later, on the training part only, inside an sklearn `Pipeline` (or
saved from train for the CNN).

### 2.12 Walking through `feature_pipeline.png`

The figure shows window 0 (0–3 s) of blues.00000.

1. **Waveform** (66,150 samples). Amplitude over time. Bursts at ~0.3, 0.7,
   1.0, 1.25 and 2.65 s are notes/phrases; the envelope already shows rhythm,
   but frequency content is invisible.
2. **STFT magnitude** (1025 bins × 130 frames, linear Hz axis). Energy is
   concentrated below ~2 kHz; vertical streaks at note onsets are broadband
   attacks; on a linear axis most of the image is the top half, where little
   happens.
3. **Log-mel** (128 bands, mel axis). Same data, but the low frequencies are
   stretched out: horizontal stripes between 200 Hz and 2 kHz are the
   harmonics of the guitar/voice, and their slight curves are pitch bends
   typical of blues. This is the CNN's input.
4. **MFCCs** (20 coefficients). Coefficient 1 (bottom row) is strongly
   negative and tracks loudness (lighter at the loud bursts), coefficient 2 is
   large and positive throughout (the low-heavy tilt), and higher coefficients
   are small with little structure: the envelope is summarised by the first
   few. Means and variances of these 20 rows are 40 of our 57 features.

---

## 3. Why this choice, and the alternatives we rejected

| Choice | Alternative rejected | Why |
|---|---|---|
| Pad/trim to exactly 661,500 samples | Drop short clips; variable lengths | 9 clips short by ≤ 0.07 s; fixed shape for all models. |
| 10 non-overlapping 3 s windows | Overlapping windows | Overlap duplicates audio across rows; no new information. |
| Tempo once per clip | Per window | 3 s holds ~6 beats; per-window tempo is noisy. |
| HPSS on the full clip | Per window | 10× faster, no window-edge artefacts. |
| `ref=1.0` | `ref=np.max` | Keeps absolute loudness; normalisation is a training-time choice. |
| float16 mels per clip | float32; pre-cut windows | Half the disk; windowing stays in training code. |
| Means and variances | Full frame-level sequences | Classical models need fixed-length vectors; the CNN gets the sequences. |
| Our own extraction | Kaggle CSVs | Reproducible from `config.yaml` (DECISIONS.md). |

---

## 4. Code walkthrough (`src/features.py`)

- Lines 12-16: constants from config; line 16 asserts that 10 × 66,150 =
  661,500, so an edited config that breaks windowing fails immediately.
- Lines 19-24, `load_clip`: `padded = y.size < N_CLIP`; line 23 zero-pads at the
  end with `np.pad` or slices to 661,500 samples. Returns float32 and the flag.
- Lines 27-30, `windows`: `reshape(10, 66150)` is a view (no copy). The assert
  guards against an unpadded input.
- Lines 33-36, `clip_tempo`: `beat_track` on the full clip; `np.atleast_1d`
  because recent librosa versions return tempo as a 1-element array.
- Lines 39-44, `feature_names`: the 57 names, built in the same order as the
  feature vector; a test checks length, uniqueness and the first/last names.
- Lines 51-71, `handcrafted_features`: one STFT magnitude `S` reused for
  chroma, centroid, bandwidth and roll-off (computed once). MFCC rows are
  appended as mean, var pairs (40 values); line 61 gives chroma the power
  spectrum `S**2` (librosa's convention); line 65 uses the same frame length
  and hop for ZCR and RMS as for the STFT; line 70 appends the clip tempo.
- Lines 74-77, `clip_mel`: mel power spectrogram of the full clip, dB with
  `ref` from config, cast to float16.
- Lines 80-89, `clip_features`: loads once, HPSS once (line 83), tempo once,
  then iterates over the windows of the signal and of its harmonic and
  percussive parts together with `zip`.

`scripts/extract_features.py`: `Parallel(n_jobs=-1, return_as="generator")`
returns results in input order while they finish, so `tqdm` can show progress;
each worker writes its mel straight to disk, so the parent never holds 999
spectrograms. `index.csv` is built with `np.repeat`/`np.tile` so row $i$ of the
array is clip $\lfloor i/10 \rfloor$, window $i \bmod 10$.

---

## 5. Pitfalls

1. **Standardising features here** (e.g. z-scoring over all clips before
   saving). That is a learned statistic over test clips: leakage. We save raw
   values only.
2. **`ref=np.max`** silently removes loudness, and **a dataset-wide max** would
   leak.
3. **Splitting windows randomly later.** All 10 windows of a clip share the
   clip's tempo exactly, and adjacent windows share HPSS context; a window-level
   split (split A) lets the model match test windows to their training
   siblings.
4. **Forgetting the padding.** Without padding, `reshape` fails on 9 clips (or,
   with a lenient slicer, their last window would be short).
5. **Mismatched frame settings** between features (e.g. ZCR with a different
   frame length) make features hard to interpret together.
6. **Treating float16 as exact.** Fine for dB values, but convert to float32
   before computing statistics to avoid overflow and rounding.
7. **Holding all audio in memory** (≈2.6 GB for 999 clips) instead
   of one clip per worker.

---

## 6. Results from this step

- `features/handcrafted.npy`: **9,990 × 57**, float32, **0 NaN/inf** values
  (999 usable clips × 10 windows).
- `features/mel/`: **999 files**, every one **128 × 1292** float16; the set of
  files equals the set of `load_ok` clips.
- **9 clips padded** (`clip_padded = True`): classical.00049, classical.00051,
  country.00003, country.00004, country.00007, disco.00014, hiphop.00031,
  hiphop.00032, rock.00038. All other clips trimmed to 30.000 s.
- Clip tempo: min 58.7, median 117.5, max 234.9 BPM (the max is likely a
  double-tempo octave error).
- Runtime: **284.5 s** for 999 clips with joblib on all cores. `features/` is
  319 MB on disk.
- Median per genre (all windows) of a few features:

| genre | mfcc1_mean | centroid_mean (Hz) | zcr_mean | harm_rms_mean | perc_rms_mean |
|---|---|---|---|---|---|
| blues | −159.7 | 1722 | 0.072 | 0.094 | 0.042 |
| classical | −321.9 | 1323 | 0.073 | 0.026 | 0.004 |
| country | −135.9 | 1813 | 0.079 | 0.090 | 0.038 |
| disco | −88.5 | 2542 | 0.127 | 0.078 | 0.065 |
| hiphop | −98.8 | 2515 | 0.104 | 0.110 | 0.082 |
| jazz | −213.1 | 1662 | 0.066 | 0.062 | 0.019 |
| metal | −66.7 | 2629 | 0.148 | 0.088 | 0.060 |
| pop | −56.9 | 3130 | 0.130 | 0.131 | 0.094 |
| reggae | −162.0 | 2059 | 0.080 | 0.080 | 0.046 |
| rock | −99.5 | 2258 | 0.109 | 0.088 | 0.048 |

  Classical has almost no percussive energy (0.004) and the lowest `mfcc1`
  (quietest); pop and metal are loudest and brightest. These are medians:
  the distributions overlap, which is why we need models rather than rules.
- Figure: `results/figures/feature_pipeline.png`.
- `pytest -q`: `10 passed` (4 of them in `tests/test_features.py`).

---

## 7. Viva questions with model answers

**Q1. Why log-mel instead of the raw STFT for the CNN?**
128 bands instead of 1025 bins, spaced like human pitch perception, and log
compression that matches loudness perception and tames the dynamic range. The
CNN gets a compact image where musically relevant detail is not dwarfed.

**Q2. Why do low MFCCs describe timbre and not pitch?**
The DCT of the log-mel spectrum separates slow variation across frequency (the
envelope, i.e. timbre) into low coefficients and fast ripples (individual
harmonics, i.e. pitch) into high ones. Keeping 20 keeps the envelope.

**Q3. What if you used n_fft = 512?**
23 ms frames, 43 Hz bins: sharper onsets, but low notes a semitone apart fall
into the same bin and the lowest mel filters would contain zero or one bin.

**Q4. Is computing HPSS on the full clip leakage?**
No. It uses only that clip's audio. Leakage needs information from other
clips (or labels) to influence a clip's features or the model.

**Q5. What if you had standardised the 57 features over all 9,990 windows
before saving?**
The means and variances would include test windows, so the scaling applied to
training data would depend on test data. Small in effect but a real leak, and
avoidable by fitting the scaler in a Pipeline on train only.

**Q6. Why is tempo the same for all 10 windows of a clip, and does that cause
problems?**
It is a song property and too noisy on 3 s. It does make windows of one clip
even more similar, which is one more reason never to split windows of a clip
across train and test.

**Q7. Why store the full-clip mel instead of ten window mels?**
The windowing choice (3 s, overlap, random crops for augmentation) then stays
in the training code and can change without re-extracting.

**Q8. What does the 80 dB floor (`top_db`) do and is it a learned statistic?**
It floors each clip at 80 dB below that clip's own maximum. It depends only
on the clip itself, so it is a fixed per-clip transform, not leakage. It does
mean very quiet bins are compared on a per-clip floor.

**Q9. What if a clip were stereo?**
`librosa.load(mono=True)` averages the channels; GTZAN is mono already, so
nothing changes here.

---

## 8. Further reading

- Müller (2015), *Fundamentals of Music Processing*, ch. 2 (Fourier analysis,
  STFT) and ch. 3 (chroma). https://www.audiolabs-erlangen.de/fau/professor/mueller/bookFMP
- Davis & Mermelstein (1980), *Comparison of parametric representations for
  monosyllabic word recognition* (MFCCs). https://doi.org/10.1109/TASSP.1980.1163420
- Fitzgerald (2010), *Harmonic/percussive separation using median filtering*,
  DAFx. https://arrow.tudublin.ie/argcon/67/
- Ellis (2007), *Beat tracking by dynamic programming*, J. New Music Research.
  https://doi.org/10.1080/09298210701653344
- McFee et al. (2015), *librosa: Audio and music signal analysis in Python*,
  SciPy. https://doi.org/10.25080/Majora-7b98e3ed-003
- Tzanetakis & Cook (2002), the original timbral/rhythm/pitch feature set
  for GTZAN. https://doi.org/10.1109/TSA.2002.800560
