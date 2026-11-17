# Lesson 21: A pretrained model, checked after the test run (validation only)

Lesson 18 diagnosed the ~0.55 plateau: each genre in split C is learned from a handful of artists,
and the test part has different ones. A diagnosis makes a prediction: a model that already knows a
lot of music should do much better. This lesson tests that prediction with a pretrained audio
network, **on the validation part only**, after the one-time test run. It also explains why the
result, although large, cannot replace the headline.

Nothing here changed a model, setting, prediction or score that the headline uses. The test part was
never loaded.

---

## 1. What we built

| file | what it does |
|---|---|
| `scripts/pretrained_embeddings.py` | embeds every 3 s window of C train and C valid with PANNs CNN14, tunes logistic regression on valid exactly as for the handcrafted features, and compares it with the existing models on the same resampled artists |
| `src/features.py` | `resample_windows`: 22,050 Hz windows → 32 kHz, CNN14's input rate |
| `tests/test_features.py` | `test_resample_windows_to_32k`: ten 3 s windows become (10, 96,000) float32 |
| `config.yaml` | `pretrained:` checkpoint path, download URL, sample rate |
| `requirements.txt` | `panns-inference==0.1.1`, `torchlibrosa==0.1.0` |
| `results/val/exploratory/` | `C_panns_logreg.json` (scores, grid, per-genre F1, paired comparisons) and the clip predictions |
| `docs/index.html` | new subsection "Checked after the test run" in "Why the scores stop near 0.55"; the first future-work item now points to it; figures dimmed in dark mode |

Commands (repo root; downloads are about 330 MB):

```bash
mkdir -p models/pretrained ~/panns_data
curl -L -o "models/pretrained/Cnn14_mAP=0.431.pth" "https://zenodo.org/records/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
curl -L -o ~/panns_data/class_labels_indices.csv http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv
python scripts/pretrained_embeddings.py     # ~35 s of embedding on a GPU, then tuning and bootstrap
```

The package's own downloader calls `wget`, which Windows lacks, so the files are fetched with `curl`
and the checkpoint path is passed explicitly.

## 2. Concepts in depth

### 2.1 Transfer learning and the linear probe

Intuition: our CNN learned everything from 442 clips. CNN14 learned from about two million
10-second YouTube clips (AudioSet) to recognise 527 sound classes, many of them musical (instruments,
genres, singing). Its second-to-last layer is a 2,048-number summary of a sound: an **embedding**.
If those numbers already separate genres, a very simple classifier on top should do well. Fitting
only a linear classifier on frozen embeddings is called a **linear probe**: it measures how much
genre information the pretrained features already contain, without retraining the network.

Math: CNN14 is a fixed function $\phi$ from a window to $\mathbb{R}^{2048}$. We fit multinomial
logistic regression on standardised embeddings:

```latex
p(g \mid x) = \frac{\exp(w_g^\top z + b_g)}{\sum_{h} \exp(w_h^\top z + b_h)}, \qquad
z = \frac{\phi(x) - \mu_{\text{train}}}{\sigma_{\text{train}}}
```

minimising cross-entropy plus $\frac{1}{2C}\lVert w \rVert^2$, with $C$ chosen on validation from the
same grid as before, {0.01, 0.1, 1, 10}. The scaler's $\mu$ and $\sigma$ come from training windows only
(inside the sklearn Pipeline). The clip prediction is the soft vote over its ten windows.

Because $\phi$ is fixed and never sees our labels, embedding a clip is a fixed transform, like
computing MFCCs. Only the logistic regression learns from our data.

### 2.2 Worked example with our numbers

`results/val/exploratory/C_panns_logreg.json`:

| model (C valid, trained on C train) | macro-F1 | 95% interval | pretrained minus this (paired) |
|---|---|---|---|
| CNN14 embeddings + logistic regression | 0.764 | 0.650 to 0.824 | – |
| CNN on log-mels, mean of 3 runs | 0.620 | 0.510 to 0.673 | +0.144 (+0.076 to +0.230) |
| logistic regression, handcrafted | 0.468 | 0.381 to 0.588 | +0.296 (+0.154 to +0.381) |
| RBF SVM, handcrafted | 0.458 | 0.366 to 0.568 | +0.306 (+0.170 to +0.387) |

The grid: C = 0.01 → 0.764, 0.1 → 0.750, 1 → 0.717, 10 → 0.716. Validation accuracy 0.777.

Per genre, compared with logistic regression on handcrafted features: blues 0.32 → 0.91, hip-hop
0.17 → 0.80, metal 0.52 → 0.95, disco 0.59 → 0.87. The weakest stay country (0.49), rock (0.57) and
pop (0.58).

### 2.3 Why a 0.30 gain is believable here, and what it does not show

Three checks make the number trustworthy *as a validation number*:

1. **Not a lucky setting.** All four grid values score 0.716 to 0.764, every one above the CNN's
   0.620. Selection over four settings can add a little optimism; it cannot create a 0.30 gap.
2. **The comparison is paired and controlled.** Logistic regression, scaler, grid, scorer, soft vote
   and clips are identical to the handcrafted model; only the input features differ. The paired
   interval (+0.15 to +0.38) is far from zero.
3. **Reproducible.** Two runs, one embedding from scratch and one from the cache, give identical
   results.

And two reasons it is **not** comparable with the headline:

1. **Order of events.** The model was added after the test part was scored. Scoring it on test now
   would be choosing a model after seeing test (lesson 17, winner's curse; lesson 18, the
   garden of forking paths).
2. **Pretraining can leak artists.** AudioSet is drawn from YouTube and labelled with genres such as
   "Rock music" and "Jazz". The network may have heard C's validation artists, even these recordings,
   during pretraining. Split C keeps artists apart between *our* train and test; it says nothing
   about the pretraining data. This is the same kind of leakage the project measures, entering
   through a different door, and we cannot check how large it is.

So the result supports the diagnosis (knowing more music helps a lot) but does not measure
performance on truly unheard artists.

## 3. Why this choice, and the alternatives we rejected

- **CNN14 (PANNs)** is the standard pretrained audio network, installable as one small package,
  and fast (639 clips in about 35 s on our GPU). The website's own future-work list named it.
- **Rejected: MERT or CLAP.** Music-specific or text-audio models may be stronger, but need larger
  libraries with remote code; the question was "does pretrained knowledge help", not "which is best".
- **Rejected: fine-tuning CNN14.** More settings to choose on 197 validation clips and much more
  compute; a linear probe answers the question with one hyperparameter.
- **Rejected: scoring on test.** See 2.3. A test number chosen after the headline would look like a
  result and be a selection.
- **Rejected: embedding all 999 clips.** Embedding is a fixed transform, so it would not leak, but the
  script embeds only C train and valid so the test part is never even read.
- **Same 3 s windows and soft vote** as every other model, so the comparison changes one thing.

## 4. Code walkthrough

- **L37-39 `resample_windows`** (`src/features.py`): `librosa.resample` along the last axis turns
  (10, 66,150) at 22,050 Hz into (10, 96,000) at 32 kHz, kept as float32 for the GPU.
- **L40-56 `embed_clips`** (`scripts/pretrained_embeddings.py`): returns the cache if its clip list
  matches; otherwise imports `panns_inference` (the import reads the AudioSet label file), loads
  CNN14 on the GPU if there is one, and for each clip runs `load_clip` → `windows` →
  `resample_windows` → `model.inference`, keeping the embedding (index 1 of the returned pair). Clip
  ids are saved as strings, because NumPy refuses to load object arrays without pickle. That was a
  real bug on the first run.
- **L59-63 `part_matrix`**: picks each window's embedding by (clip position, window index) with one
  NumPy fancy index, and returns X and y indexed by clip id, the format `tune_and_score` expects.
- **L66-83 `compare`**: for each model, the bootstrap distribution over the same artist-cluster
  resamples (same rows, clusters and seed), averaging the CNN's three runs as the headline does. The
  pretrained array minus each other array is a paired bootstrap; its percentiles give the interval.
- **L86-127 `main`**: loads only `load_split("C", "train")` and `("C", "valid")`, and asserts that
  every embedded clip is train or valid. `tune_and_score` runs the identical grid search. The refit
  under `threadpool_limits(1)` repeats its final fit to get clip predictions, and an assertion checks
  that their macro-F1 equals the reported one. The other models' predictions are merged by clip id
  with `validate="1:1"`.

## 5. Pitfalls

- **Treating 0.76 as the new headline.** It is validation, it is post-test, and pretraining may have
  heard these artists.
- **Comparing against the test headline (0.55).** Different part, different clips. Compare like with
  like: the CNN's *validation* score is 0.620.
- **Fitting the scaler on all windows.** The scaler sits inside the Pipeline, so it is fitted on
  training windows only, as for every other model.
- **Normalising differently from CNN14's training.** CNN14 computes its own log-mel internally from
  the raw waveform; we pass raw audio at its expected 32 kHz rate. Passing 22,050 Hz audio would
  silently shift every frequency.
- **The best C at the grid's edge.** C = 0.01 is the smallest value tried. A smaller C might score a
  little higher, but trying it now would be more selection on validation, and the whole grid is
  already above the CNN.

## 6. Results from this step

C validation macro-F1 0.764 (0.650 to 0.824) for CNN14 embeddings + logistic regression (C = 0.01),
against 0.620 (CNN, mean of 3 runs), 0.468 (logistic regression, handcrafted) and 0.458 (SVM). Paired
differences: +0.144, +0.296 and +0.306, all with intervals above zero. File:
`results/val/exploratory/C_panns_logreg.json`. Website: "Checked after the test run: a pretrained model
(validation only)".

## 7. Viva questions with model answers

1. **Why not report the pretrained model's test score?** It was added after the test run, so any test
   score would come from a model chosen after seeing test. The headline must come from choices fixed
   before the tag.
2. **The pretrained model scores 0.76. Is your 0.55 headline now wrong?** No. They answer different
   questions on different parts. 0.55 is a test score for models fixed in advance; 0.76 is a
   validation score for a model that may have heard these artists in pretraining.
3. **How could pretraining leak?** AudioSet comes from YouTube and includes popular music with genre
   labels; a GTZAN artist or recording may appear there. Our artist-disjoint split does not cover the
   pretraining data.
4. **What if you fine-tuned CNN14 instead of a linear probe?** Likely higher still, but with more
   settings to choose on 197 clips, so more selection on noise; the probe already answers the
   question.
5. **What if you passed 22,050 Hz audio straight to CNN14?** It assumes 32 kHz, so every frequency
   would be read about 1.45 times too low and timbre would be distorted; scores would likely drop.
6. **What would a fair test of the pretrained model need?** Pretraining data checked for GTZAN's
   artists (or a model trained without them), the model fixed before looking at test, and then one
   test run, as for the headline.
7. **Why compare against logistic regression on handcrafted features rather than the CNN?** It
   isolates the features: same classifier, grid, scaler and vote; only the input changes. The CNN
   comparison is shown too.

## 8. Further reading

- Q. Kong et al., "PANNs: Large-scale pretrained audio neural networks for audio pattern
  recognition," IEEE/ACM TASLP 28, 2020. https://arxiv.org/abs/1912.10211
- J. F. Gemmeke et al., "Audio Set: An ontology and human-labeled dataset for audio events," ICASSP
  2017. https://research.google.com/audioset/
- `panns_inference` code. https://github.com/qiuqiangkong/panns_inference
- G. Alain and Y. Bengio, "Understanding intermediate layers using linear classifier probes," 2016.
  https://arxiv.org/abs/1610.01644
- S. Kornblith, J. Shlens and Q. V. Le, "Do better ImageNet models transfer better?" CVPR 2019
  (linear probes as a measure of transfer). https://arxiv.org/abs/1805.08974
