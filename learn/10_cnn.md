# Lesson 10: A convolutional network on log-mel spectrograms

Every model so far looked at 57 numbers per 3 s window that **we** chose
(lesson 05: MFCC means, spectral centroid, tempo...). A convolutional neural
network (CNN) instead looks at the **log-mel spectrogram itself**, a
128 x 130 picture of each 3 s window, and **learns** which local patterns
matter. This lesson explains every piece of that network from first
principles, why we need heavy augmentation with only 442 training clips,
and how to read our training curves over three random seeds.

Same rules as before: split C (artist-disjoint), 442 train clips, 197 valid
clips, the unit of prediction is the clip (soft vote over 10 fixed windows),
the metric is clip macro-F1. The test part was never loaded.

Headline from this step: the CNN reaches **valid clip macro-F1 0.610 +/- 0.028**
(mean +/- sd over three seeds: 0.580, 0.615, 0.635). It is the first model
that clearly beats logistic regression (0.468) and the SVM (0.458) on split C.
The best epoch varies a lot between seeds (8, 26, 13; median **13**), and the
spread between seeds (0.055) is larger than the gap between our best two
classical models.

---

## 1. What we built

| File | What it does |
|---|---|
| `config.yaml` (`cnn`) | Every CNN constant: crop length 130 frames, 10 crops per clip per epoch, channels 32/64/128/128, dropout 0.3, batch 64, Adam lr 1e-3, weight decay 1e-4, 60 epochs max, patience 8, run seeds 0/1/2, augmentation sizes. |
| `src/cnn.py` | `load_mels`, `window_starts` (the 10 fixed eval windows), `compute_norm_stats` (per-mel-bin mean/std from the clips you pass, i.e. training clips), `random_gain`, `spec_augment`, `MelWindowDataset` (train mode: random augmented crops; eval mode: the 10 fixed windows; split A: only the part's own windows), `GenreCNN`, `n_params`. |
| `src/train_cnn.py` | CLI. `fit` trains with early stopping on valid clip macro-F1 (or for a fixed number of epochs in refit mode); `run` loads the split, trains, saves weights, normalisation stats and a result JSON with per-epoch curves; `summarise` combines the seeds and draws the curves figure. |
| `src/train_classical.py` | `valid_report(probs, classes, y)` now takes the per-window labels indexed by clip id, so the CNN is scored by exactly the same code as the classical models (classical JSONs re-checked: unchanged). |
| `scripts/extract_features.py` | New `--mels-only` mode (log-mels only, for Colab). |
| `scripts/download_data.sh` | Falls back to `kaggle` on PATH and environment-variable credentials (Colab). |
| `notebooks/colab_cnn.ipynb` | Clones the repo, installs requirements, takes `kaggle.json` by upload into **environment variables only** (the uploaded file is deleted immediately), downloads GTZAN, computes mels, trains the config seeds on a GPU, summarises, zips the results for download. |
| `tests/test_cnn.py` | Eval windows cover the clip without overrun; normalisation uses only the given clips; eval items are the exact normalised windows and are deterministic; train items are random crops; augmentation bounds; output shape and parameter count; a tiny training run early-stops, is reproducible with the same seed, and refit mode runs. |
| `scripts/lesson10_examples.py` | Parameter count per layer, feature-map shapes, receptive fields, multiply-adds, normalisation stats ranges (numbers quoted below). |
| `models/cnn_C_s{0,1,2}.pt` | Best weights per seed (not committed: `models/` is ignored). |
| `results/mel_norm_C_s{0,1,2}.json` | Per-bin mean and std (fit on the 442 training clips). |
| `results/val/C_s{0,1,2}_cnn.json` | Best epoch, valid scores at that epoch (same fields as the classical models), per-epoch history, timing. |
| `results/val/C_cnn_summary.json` | Per-seed results, **median best epoch**, mean/std/min/max of valid clip macro-F1. |
| `results/figures/cnn_curves_C.png` | Loss curves and valid clip macro-F1 per epoch for the three seeds. |

```bash
# one seed (CPU here: ~49 s per epoch; GPU: use the Colab notebook)
.venv/Scripts/python -W ignore -m src.train_cnn --split C --seed 0 --epochs 60 --device auto
# the three config seeds, then the summary JSON and figure
for s in 0 1 2; do .venv/Scripts/python -W ignore -m src.train_cnn --split C --seed $s; done
.venv/Scripts/python -W ignore -m src.train_cnn --split C --summary
# later (final model): train+valid for a fixed number of epochs, no valid monitoring
.venv/Scripts/python -W ignore -m src.train_cnn --split C --seed 0 --refit_epochs N
# numbers in this lesson, and tests
.venv/Scripts/python -W ignore scripts/lesson10_examples.py
.venv/Scripts/python -m pytest -q tests/test_cnn.py
```

**Time per epoch (CPU, no GPU on this laptop; PyTorch 2.15 CPU build, 8 threads).**
With nothing else running, one epoch took **48.8 s** (41.5 s training on
4,420 augmented crops, 7.3 s scoring 1,970 valid windows). During the three
seed runs other jobs (classical tuning, feature extraction) shared the CPU,
so the recorded medians are 54.8, 62.8 and 69.0 s per epoch; the fastest
epochs, 48.5 to 50.1 s, match the clean measurement. The seeds ran 16, 34
and 21 epochs: 14.4 + 35.4 + 26.2 = 75.9 minutes in total. On a Colab GPU
the same run should be much faster; the notebook logs the time per epoch so
it can be compared.

---

## 2. Concepts in depth

### 2.1 The input: a log-mel window as an image

Each clip's log-mel (lesson 05) is a matrix of 128 mel bins (rows, low
frequencies at the bottom) by 1,292 frames (columns, 23.2 ms apart: 512
samples at 22,050 Hz), values in dB. A 3 s window is 130 consecutive
columns: a 128 x 130 "image" with one channel.

Two facts about this image shape everything below:

- **The time axis is homogeneous.** A drum hit or a chord means the same
  thing at second 1 or second 2. Moving a pattern left or right should not
  change what it is.
- **The frequency axis is not.** Row 10 is around 200 Hz (bass), row 100 is
  several kHz (cymbals, sibilance). Moving a pattern up by 20 rows changes
  what it is: a bass line shifted up becomes a melody in a different
  register, and the mel scale is not even linear in pitch. We come back to
  this in 2.4.

### 2.2 Convolution as local pattern detection

**Intuition.** A 3 x 3 convolution filter is a tiny template, 3 mel bins
tall and 3 frames wide. It slides over the whole spectrogram and, at every
position, computes how strongly the local 3 x 3 patch matches the template.
The output is a new image, a **feature map**, that is bright wherever the
pattern occurs. A filter with weights like

```
-1  -1  -1
 2   2   2
-1  -1  -1
```

responds to a horizontal ridge: energy in one mel bin, sustained over time
(a held note). A filter with the transposed pattern responds to a vertical
line: energy across many bins at one instant (a drum onset). The network
does not get these templates from us; it **learns** the 9 numbers of each
filter by gradient descent.

**Math.** For an input with $C_\text{in}$ channels $x_c$ and a filter
$w \in \mathbb{R}^{C_\text{in} \times 3 \times 3}$, output channel $k$ at
position $(f, t)$ is

```latex
z_k(f, t) = \sum_{c=1}^{C_\text{in}} \sum_{i=-1}^{1} \sum_{j=-1}^{1} w_{k,c,i,j}\; x_c(f+i,\ t+j)\ \ (+\ b_k) .
```

(Deep-learning libraries compute this cross-correlation and call it
convolution; flipping the kernel would not change what can be learned.)
`padding=1` adds a border of zeros so the output has the same height and
width as the input.

Each layer has $C_\text{out}$ such filters, so it turns $C_\text{in}$ feature
maps into $C_\text{out}$ feature maps. Deeper layers combine the maps of
earlier layers: if layer 1 detects onsets and held notes, layer 2 can detect
"onset followed by a held note", and so on.

### 2.3 Weight sharing and the parameter count

**Weight sharing** means the same 3 x 3 filter is used at every position.
The number of weights therefore depends on the filter size and the channel
counts, **not on the image size**:

```latex
\text{params}(\text{conv}) = C_\text{out} \cdot C_\text{in} \cdot 3 \cdot 3 \ (+\ C_\text{out} \text{ biases}).
```

Our layers (from `scripts/lesson10_examples.py`):

| layer | output (channels x mel x frames) | parameters | multiply-adds per window |
|---|---|---|---|
| Conv 1 -> 32 | 32 x 128 x 130 | 288 | 4,792,320 |
| BatchNorm | 32 x 128 x 130 | 64 | |
| MaxPool | 32 x 64 x 65 | 0 | |
| Conv 32 -> 64 | 64 x 64 x 65 | 18,432 | 76,677,120 |
| BatchNorm | | 128 | |
| MaxPool | 64 x 32 x 32 | 0 | |
| Conv 64 -> 128 | 128 x 32 x 32 | 73,728 | 75,497,472 |
| BatchNorm | | 256 | |
| MaxPool | 128 x 16 x 16 | 0 | |
| Conv 128 -> 128 | 128 x 16 x 16 | 147,456 | 37,748,736 |
| BatchNorm | | 256 | |
| MaxPool | 128 x 8 x 8 | 0 | |
| GAP, dropout | 128 | 0 | |
| Linear 128 -> 10 | 10 | 1,290 | |
| **total** | | **241,898** | **194,715,648** |

A **BatchNorm** layer has 2 parameters per channel (scale $\gamma$, shift
$\beta$); the last linear layer has $128 \cdot 10 + 10 = 1{,}290$. Our
convolutions have **no bias** because BatchNorm's $\beta$ immediately adds a
per-channel constant anyway, so a conv bias would be redundant (it would be
subtracted out by BatchNorm's mean).

```latex
9\,(1\cdot32 + 32\cdot64 + 64\cdot128 + 128\cdot128) + 2\,(32+64+128+128) + 1290
= 239{,}904 + 704 + 1{,}290 = 241{,}898 .
```

For comparison: a fully connected layer from the 128 x 130 input to the
first block's 32 x 128 x 130 output would need
8,860,467,200 weights, about 37,000 times our whole network. Weight sharing is
what makes learning from 442 clips possible at all.

Notice where the work is: the first layer has the fewest weights (288) but
runs on the largest map, while the last conv has 61% of the weights but runs
on a 16 x 16 map. One forward pass of one window costs about 195 million
multiply-adds; training (forward + backward, roughly 3x) on 4,420 crops per
epoch is about 2.6 x 10^12 multiply-adds, which is why a CPU epoch takes
about 49 s.

### 2.4 Receptive fields, and why translation fits time better than frequency

**Receptive field.** A unit in a feature map depends on a patch of the
input: its receptive field. A 3 x 3 conv grows it by 2 x (current step
between neighbouring units, the "jump"); a 2 x 2 max-pool grows it by 1 x
jump and doubles the jump:

```latex
r_\ell = r_{\ell-1} + (k_\ell - 1)\, j_{\ell-1}, \qquad j_\ell = j_{\ell-1}\, s_\ell ,
```

with kernel size $k$ and stride $s$ (conv: $k=3, s=1$; pool: $k=2, s=2$).
Starting from $r=1, j=1$:

| after | receptive field (bins x frames) | jump | in seconds |
|---|---|---|---|
| block 1 conv / pool | 3 / 4 | 1 / 2 | 0.07 / 0.09 |
| block 2 conv / pool | 8 / 10 | 2 / 4 | 0.19 / 0.23 |
| block 3 conv / pool | 18 / 22 | 4 / 8 | 0.42 / 0.51 |
| block 4 conv / pool | 38 / 46 | 8 / 16 | 0.88 / 1.07 |

So each unit of the last feature map (8 x 8 positions) sees a 46-bin x
46-frame patch: about a third of the mel range and about one second. Global
average pooling (2.6) then combines all 64 positions, so the final decision
uses the whole 3 s window, but each detected pattern is at most ~1 s long and
spans about a third of the frequency range.

**Translation equivariance.** Because the filter is the same everywhere,
shifting the input shifts the feature map by the same amount:

```latex
(\text{conv}\ (T_\delta x)) = T_\delta\, (\text{conv}\ x),
```

where $T_\delta$ shifts by $\delta$ positions. Pooling and global averaging
then make the output approximately **invariant** to the shift: the network
says "there is a drum pattern" regardless of when it happened.

- **Along time this is exactly right**: genre does not depend on when, inside
  3 s, the snare hits.
- **Along frequency it is only partly right.** The network will treat a
  pattern at 200 Hz and the same shape at 2 kHz as the same detection. Some
  of that is useful (a vibrato looks similar in any register, transposing a
  song does not change its genre), but much is not: kick drums live low,
  hi-hats high, and the mel bins are not evenly spaced in Hz. Two things
  give our network back some frequency awareness: zero padding at the
  borders lets filters tell the top and bottom edge apart, and the receptive
  field of the last layer (46 of 128 bins) is not the full height, so
  "where" survives into the deeper layers until global averaging. Designs
  that respect frequency more explicitly use tall filters spanning all bins
  (Pons et al. 2016), add a frequency-position channel, or pool over time
  only. We keep the simple square design as the standard baseline CNN.

### 2.5 BatchNorm, ReLU, max-pooling

**Batch normalisation** (Ioffe & Szegedy 2015). For each channel, over the
current mini-batch $B$ and all positions:

```latex
\hat z = \frac{z - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \qquad y = \gamma \hat z + \beta .
```

It keeps every layer's inputs on a stable scale while the earlier layers
change, which allows a larger learning rate and faster, more reliable
training; the noise from batch statistics also regularises a little. At
evaluation time it uses running averages of $\mu$ and $\sigma^2$ collected
during training instead of batch statistics (`model.eval()` switches this,
which is why `predict` calls it). Forgetting `model.eval()` would make
valid predictions depend on which windows share a batch.

**ReLU** $\max(0, z)$. Without a non-linearity, stacked convolutions would
collapse into one linear filter. ReLU is the simplest non-linearity that
trains well: its gradient is 1 for active units (no vanishing as in
sigmoids) and it produces sparse activations ("this pattern is present /
absent").

**Max-pooling 2 x 2** keeps the largest value in each 2 x 2 block. It halves
height and width (128 x 130 -> 64 x 65 -> 32 x 32 -> 16 x 16 -> 8 x 8),
which cuts computation for the next layer by 4, doubles the receptive field
growth, and adds a little tolerance to small shifts ("the pattern occurred
somewhere in this 2 x 2 area"). (65 is odd, so the last frame of that map is
dropped by the next pooling; harmless.)

### 2.6 Global average pooling and dropout

After block 4 we have 128 feature maps of 8 x 8. **Global average pooling
(GAP)** replaces each map by its mean: 128 numbers saying "how much of
pattern k is present in this window, on average". Compared with flattening
(128 x 8 x 8 = 8,192 inputs to the linear layer, 81,930 weights), GAP has no
parameters, fixes the classifier's input size whatever the crop length, and
is strongly invariant to *where* patterns occur. It is the main reason the
network has only 241,898 parameters.

**Dropout 0.3** (Srivastava et al. 2014): during training, each of the 128
pooled values is set to 0 with probability 0.3 (and the rest scaled by
1/0.7). The classifier cannot rely on any single pattern detector, which
reduces co-adaptation and overfitting. At evaluation dropout is off.

### 2.7 Cross-entropy and Adam

The network outputs 10 **logits** $z_g$; the softmax turns them into
probabilities $p_g = e^{z_g}/\sum_h e^{z_h}$, exactly as in multinomial
logistic regression (lesson 07). Training minimises the mean cross-entropy
over a mini-batch of 64 crops:

```latex
L = -\frac{1}{64}\sum_{i=1}^{64} \log p_{y_i}(x_i) .
```

Its gradient with respect to the logits is again $p - \text{onehot}(y)$;
backpropagation carries it through the network by the chain rule.

**Adam** (Kingma & Ba 2015) keeps, per parameter, running averages of the
gradient and of its square, and steps by their ratio:

```latex
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t,\quad
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2,\quad
\theta_t = \theta_{t-1} - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon},
```

with $\hat m_t, \hat v_t$ bias-corrected, $\beta_1 = 0.9$, $\beta_2 =
0.999$, $\eta = 10^{-3}$. Dividing by $\sqrt{\hat v_t}$ gives every parameter
its own effective step size, so layers with very different gradient scales
train at similar speed without hand tuning. That robustness is why it is the
default for small CNNs.

### 2.8 Weight decay

`weight_decay=1e-4` in `torch.optim.Adam` adds $\lambda\theta$ to every
gradient, i.e. an L2 penalty $\tfrac{\lambda}{2}\lVert\theta\rVert^2$ on the
loss, the same idea as C in logistic regression and the SVM: prefer small
weights, simpler functions. A subtlety: inside Adam, the added
$\lambda\theta$ is divided by $\sqrt{\hat v_t}$ like the rest of the
gradient, so parameters with large gradients are decayed less. **AdamW**
(Loshchilov & Hutter 2019) instead subtracts $\eta\lambda\theta$ directly
("decoupled" decay), which behaves more like true L2 regularisation. We use
Adam as specified; with $\lambda = 10^{-4}$ and BatchNorm after every conv
(which makes the conv weights' scale irrelevant to the output) the
difference is small here.

### 2.9 Augmentation and SpecAugment: why we need them with 442 clips

**The problem.** 241,898 parameters and only 442 training clips from
artist-disjoint groups. A network this size can memorise 442 clips
(including their artist-specific sound) long before it learns genre. Every
regulariser above helps, but the strongest tool is to make the training
data **look larger and more varied** in ways that do not change the genre.

Our three augmentations, all applied **only to training crops**:

1. **Random crops.** Each epoch takes 10 crops per clip at random offsets
   (any of 1,292 - 130 + 1 = 1,163 start frames) instead of always the same
   10 windows. Over 60 epochs the network sees up to 600 different crops per
   clip, never exactly the same input twice in a row, and learns that genre
   does not depend on where the window starts.
2. **Random gain, uniform in [-6, +6] dB**, added to the whole log-mel
   window **before** normalisation. Turning the volume up by $g$ dB
   multiplies the power by $10^{g/10}$, which is exactly $+g$ dB on every
   cell of a log spectrogram. Loudness differs between recordings for
   reasons unrelated to genre (mastering, radio level, the recording chain
   GTZAN came from); the network should not rely on it. It must be added
   before normalisation, because that is where it corresponds to a real
   change in the audio; after normalisation the same shift would be scaled
   differently in every mel bin (each has its own std).
3. **SpecAugment** (Park et al. 2019), after normalisation: 2 frequency
   masks of width 0 to 15 mel bins and 2 time masks of width 0 to 20 frames,
   at random positions, set to 0 (= the training mean, since data are
   normalised). On average about 11.7% of the mel bins and 15.4% of the
   frames are masked (ignoring overlaps). The network must recognise the
   genre with a band of frequencies or a short stretch of time missing, so
   it cannot rely on one narrow cue (one instrument's band, one beat). We
   omit SpecAugment's time-warping step, which is the most expensive and
   gave the smallest gain in the original paper.

None of these are applied at evaluation: valid windows are the 10 fixed,
unmodified windows of each clip, so the score is deterministic and measures
the real audio.

### 2.10 Per-mel-bin normalisation from training clips only

Before the network, every mel bin $f$ is standardised:

```latex
\tilde x(f, t) = \frac{x(f, t) - \mu_f}{\sigma_f}, \qquad
\mu_f, \sigma_f \text{ over all frames of all 442 training clips.}
```

Low bins carry far more energy than high bins (lesson 05's mel plots), so
without this the first layer would see inputs whose scale changes by tens of
dB from bottom to top. These 256 numbers are **learned statistics** and so,
by the project rule, come from training clips only. They are saved to
`results/mel_norm_C_s{seed}.json` so the exact same transform can be applied
to test clips later. On our 442 training clips the bin means
range from -41.0 dB (top bins) to 8.2 dB (bin 2, deep bass) and the standard
deviations from 11.0 to 15.7 dB (`scripts/lesson10_examples.py`), so without
normalisation the bottom rows would be about 50 dB brighter than the top
rows for every clip.

### 2.11 Early stopping on clip macro-F1

After every epoch we predict all 1,970 valid windows (10 fixed windows x 197
clips), soft-vote per clip, and compute clip macro-F1 with the same
`valid_report` as every other model. If it beats the best so far, we keep a
copy of the weights. If 8 epochs pass without improvement, training stops,
and the kept weights are the result. The best epoch is recorded; its median
over seeds will fix the number of epochs for a later train+valid refit
(`--refit_epochs`), where no valid set is left to stop on.

Unlike XGBoost (lesson 08) we stop on the project metric itself, not on the
loss. Clip macro-F1 moves in jumps of a few thousandths, so "8 epochs
without a new best" can be triggered by noise; the patience of 8 and the
fact that each epoch changes the network a lot (unlike one boosting round)
make that acceptable. The valid loss is logged too, so we can see both.

---

## 3. Why this choice, and the alternatives we rejected

| Choice | Alternative | Why |
|---|---|---|
| Small 4-block CNN from scratch | A pretrained audio network (PANNs, VGGish, AST) | Pretrained models learned from millions of clips, possibly overlapping GTZAN's sources; they would answer "how good is a big pretrained model", not "what can be learned from our 442 clips". A possible extension, not the baseline. |
| 3 x 3 square filters | Tall frequency-spanning filters / 1-D convolution over time | Square 3 x 3 stacks are the standard image-style baseline for spectrograms (Choi et al. 2016) and need no music-specific design; the trade-offs are discussed in 2.4. |
| Global average pooling | Flatten + dense layer | 81,930 vs 1,290 weights in the head, less overfitting, independent of crop length. |
| BatchNorm, no conv bias | Conv bias + BatchNorm | The bias is cancelled by BatchNorm's mean subtraction; BatchNorm's beta does its job. |
| Random 130-frame crops, 10 per clip per epoch | The fixed 10 windows | Fixed windows give the same 4,420 inputs every epoch; random crops give 1,163 possible offsets per clip. 10 per clip keeps epochs comparable in size to the classical training set. |
| Per-mel-bin stats from train | Per-clip / per-window normalisation | Per-clip normalisation removes absolute spectral level (a real genre cue: metal is bright and loud) and behaves differently on crops than on whole clips. Global per-bin stats keep the cue and are a fixed, saved transform. |
| Gain before normalisation | Gain after | Only before normalisation is it a real volume change (2.9). |
| Adam, lr 1e-3, wd 1e-4 | SGD with momentum + schedule; AdamW | Adam's defaults train small CNNs reliably without a learning-rate schedule search; AdamW is discussed in 2.8. |
| Early stop on valid clip macro-F1 | On valid loss | Our metric decides, as for all models; loss is logged. |
| Three seeds | One seed | A CNN's result depends on its random initialisation, crop order and masks; one seed would mix that noise into the model comparison (2.12). |
| CPU here, Colab GPU notebook | Require a GPU | The CPU run is feasible (about 49 s per epoch, 76 min for three seeds); the notebook makes faster reruns possible. |

---

## 4. Code walkthrough

### `src/cnn.py`

- **L24–26 `load_mels`**: loads each clip's float16 log-mel and stacks them as
  float32 (n_clips, 128, 1292). The whole train set (442 clips) is about 290 MB
  in memory, so items are cut from RAM, not from disk.
- **L29–34 `window_starts`**: `np.linspace(0, n_frames - 130, 10).round()`.
  Even spacing from 0 to the last possible start guarantees the 10th window
  ends exactly at the last frame (tested).
- **L37–55 `compute_norm_stats`**: accumulates per-bin sums and sums of
  squares in float64 over every frame of the clips passed in, then
  $\sigma^2 = E[x^2] - \mu^2$. A floor of 1e-6 on $\sigma$ prevents division
  by zero. With `window_idx` (split A) only the frames of the listed windows
  count. The caller passes **training** clips only.
- **L58–60 `random_gain`**: one uniform draw in [-6, 6] dB from torch's RNG
  (so `set_seed` controls it), added to every cell.
- **L63–81 `spec_augment`**: for each mask, a width uniform in [0, max] and a
  start uniform over the valid positions; the band is set to 0 on a copy.
- **L84–131 `MelWindowDataset`**:
  - L104–108: clip-level splits. Train mode lists 10 items per clip and
    sets `random_crop`; eval mode lists the 10 fixed windows.
  - L109–112: split A. Items are exactly the (clip, window) pairs of the
    part, at fixed positions.
  - L113–114: `clip_ids` and `item_labels` per item, used later for the soft
    vote and the per-window labels.
  - L119–131 `__getitem__`: random start (train, clip-level) or fixed start;
    slice 130 frames; gain (train) **before** normalising; normalise with the
    passed stats; add the channel axis; SpecAugment (train) **after**
    normalising.
- **L134–153 `GenreCNN`**: L145 builds each block (`bias=False` because of
  BatchNorm, `padding=1` keeps the size before pooling); L149 the head: GAP,
  flatten, dropout, linear.
- **L156 `n_params`**: counts trainable parameters (241,898; tested against
  the formula).

### `src/train_cnn.py`

- **L39–41 `clip_labels`**: one genre per clip of a part (used by the analysis
  scripts).
- **L44–50 `rows_to_clips`**: turns the split's window rows into genre per
  clip plus, for window-level split A only, the list of windows per clip.
- **L53–57 `encode`**: genre names to int64 codes in config order (int64
  because PyTorch's loss needs `Long` targets; pandas gives int8).
- **L60–64 `pick_device`**: `auto` = CUDA if available, else CPU.
- **L67–72 `predict`**: `model.eval()` (BatchNorm running stats, dropout
  off), no gradients, softmax probabilities for every valid window in order.
- **L75–87 `train_epoch`**: the standard loop: zero gradients, forward,
  cross-entropy, backward, Adam step; returns the mean loss per crop.
- **L90–139 `fit`**:
  - L98–102: stats from the training clips, training dataset, shuffled
    loader (batch 64), model, Adam (lr 1e-3, weight decay 1e-4).
  - L103–107: valid dataset in eval mode **with the training stats**, and
    per-window labels for scoring.
  - L113–118: after each epoch: valid probabilities, valid loss, and
    `valid_report` (the same function as for the classical models), giving
    clip macro-F1 by soft vote.
  - L125–133: keep a copy of the weights when clip macro-F1 improves
    (strictly), otherwise count; stop after 8 epochs without improvement.
  - Refit mode (no valid): run all epochs, keep the last weights.
- **L142–175 `run`**: L144 seeds everything with 42 + run seed; L145–155
  loads the parts and trains (refit: train+valid together); L158 saves the
  weights to `models/`; L160 saves the normalisation stats (needed to
  preprocess test clips identically); then the result JSON with the
  best-epoch report and the full history.
- **L178–207 `plot_curves`**, **L210–228 `summarise`**: the figure and the
  summary JSON over the config seeds; L219 is the median best epoch.
- **L231 `main`**: the CLI (`--split`, `--seed`, `--epochs`, `--device`,
  `--refit_epochs`, `--summary`).

### `tests/test_cnn.py`

- `test_window_starts_cover_clip_without_overrun`: the exact starts
  0, 129, ..., 1162 and the last window ends at frame 1292. A naive
  `round(i * 129.2)` would start the last window at 1163 and overrun by one
  frame.
- `test_norm_stats_use_only_the_given_clips`: the stats equal the mean/std of
  exactly the clips passed in, and change when a different clip is passed,
  so they cannot silently include other clips.
- `test_eval_items_are_fixed_normalised_windows`: item 13 is clip 2's window
  3, equal to the normalised slice of the raw mel, and identical on a second
  call (no augmentation in eval mode).
- `test_train_items_are_random_augmented_crops`: dataset length = clips x
  crops, two calls for the same item differ.
- `test_augmentations`: SpecAugment never masks more than 2 x 15 bins and
  2 x 20 frames and does not modify its input; gain adds one constant within
  +/-6 dB.
- `test_model_shape_and_parameter_count`: output (batch, 10), and the
  parameter count equals the formula in 2.3.
- `test_fit_early_stops_and_is_reproducible`: on 8 synthetic clips, two runs
  with the same seed give identical losses; the best epoch lies within the
  run; the report covers 2 clips x 10 windows; refit mode runs without
  valid.
- `test_window_level_items_and_stats_use_only_listed_windows`: split A
  mode: only the listed windows become items (train and eval), at their
  fixed positions, and the normalisation uses only their frames.
- `test_rows_to_clips_marks_window_level_parts`: complete clips give `None`
  (clip-level split), partial clips give their window lists.

---

## 5. Pitfalls

1. **Normalisation stats over all clips.** Computing $\mu_f, \sigma_f$ on
   train+valid(+test) is the leakage-lab mistake 1 (lesson 09) in CNN form.
   Ours come from training clips only and are saved for test time.
2. **Augmenting the evaluation data.** Random crops or masks at valid time
   would make the valid score random and measure a distorted signal.
3. **Forgetting `model.eval()`.** BatchNorm would use the valid batch's own
   statistics and dropout would stay on; scores would depend on batch
   composition.
4. **Random crops on split A.** Split A puts different windows of one clip
   in train, valid and test. Cropping anywhere in the full clip during
   training would read audio of valid/test windows: within-clip leakage.
   The CLI refuses split A; a CNN on A would need crops restricted to the
   training windows.
5. **Gain after normalisation.** It would no longer correspond to a volume
   change (each bin has its own std), so the network would learn invariance
   to something that never happens in real audio.
6. **Early stopping on valid, then quoting the valid score as final.** Valid
   picked the epoch; the score is slightly optimistic. The test part
   answers that, once.
7. **Comparing models on one CNN seed.** Seed-to-seed spread can be as large
   as the differences between models (2.12).
8. **Refitting on train+valid without fixing the epoch count first.** There
   is no valid left to stop on; the count must come from the seeds' best
   epochs (the median, recorded in the summary), decided before the test
   freeze.
9. **Expecting bit-identical results on another machine.** The mels in
   `features/mel` were computed in parallel worker processes; recomputing
   them in one process differs by at most 0.008 dB (one float16 rounding
   step) in about 0.01% of cells. GPU convolutions also use different
   algorithms. Results on Colab will be close, not identical.
10. **Credentials in the repo.** The Colab notebook keeps the Kaggle key in
    environment variables of the runtime and deletes the uploaded file at
    once; `kaggle.json` is also in `.gitignore`.

---

## 6. Results from this step

From `results/val/C_s{0,1,2}_cnn.json`, `results/val/C_cnn_summary.json` and
`results/figures/cnn_curves_C.png`. Split C valid: 197 clips.

| run seed (torch seed) | best epoch / epochs run | valid clip macro-F1 | majority-vote F1 | clip accuracy | window accuracy |
|---|---|---|---|---|---|
| 0 (42) | 8 / 16 | 0.580 | 0.568 | 0.599 | 0.551 |
| 1 (43) | 26 / 34 | 0.615 | 0.603 | 0.609 | 0.578 |
| 2 (44) | 13 / 21 | 0.635 | 0.642 | 0.645 | 0.585 |
| **mean +/- sd** | **median 13** | **0.610 +/- 0.028** | | | |

**Against the other models on the same valid part:**

| model | valid clip macro-F1 |
|---|---|
| stratified dummy | 0.128 |
| kNN | 0.424 |
| logistic regression | 0.468 |
| RBF SVM | 0.458 |
| XGBoost | 0.419 |
| CNN (3 seeds) | 0.580 / 0.615 / 0.635 |

Even the worst CNN seed beats the best classical model by 0.112. On 197
clips this is a gap of many clips, not a few, though the test part (with the
artist bootstrap) is what will put an interval on it.

**Reading the curves (`results/figures/cnn_curves_C.png`).**

- *Train loss* (left, solid) falls smoothly from about 1.7 in epoch 1 to
  0.58 (seed 0, epoch 16) and 0.37 (seed 1, epoch 34). It is measured on
  augmented crops with dropout on, so it overstates the loss the network
  would have on clean training windows.
- *Valid loss* (left, dashed) does **not** fall with it: it jumps between
  about 1.3 and 3.6 from epoch to epoch. Its minimum coincides with the best
  F1 epoch for seeds 0 and 1 (1.287 at epoch 8; 1.432 at epoch 26) and is one
  epoch off for seed 2 (1.339 at epoch 12, best F1 at 13). Spikes in
  cross-entropy with a stable F1 mean the network is sometimes **very
  confident and wrong** on some valid windows: cross-entropy punishes those
  heavily, F1 only counts them once. The calibration analysis (lesson 11)
  measures this directly.
- *Valid clip macro-F1* (right) rises from 0.16-0.30 in epoch 1 to the
  0.5-0.6 range by epoch 8 or so and then fluctuates by about 0.1 between
  epochs. The kept epoch is the peak of a noisy curve, so the kept score is
  somewhat optimistic (the same "maximum of noisy values" effect as in
  lesson 09).
- *The gap*: train loss keeps falling while valid loss does not improve:
  the network keeps fitting training-specific detail (artists, recordings)
  after it stops learning genre. Early stopping cuts this.

**Seed variance.** Same data, same code, same hyperparameters; only the
torch seed differs (initial weights, crop positions, augmentation draws,
batch order). Valid F1 ranges from 0.580 to 0.635 (range 0.055, sd 0.028)
and the best epoch from 8 to 26. Two consequences: comparisons between
models must use several seeds (a single seed could have shown 0.580 or
0.635), and the number of epochs for a final refit is taken as the
**median** best epoch, 13, not any single seed's value.

**Per-genre F1** (valid; seeds 0 / 1 / 2): blues 0.54 / 0.56 / 0.65,
classical 0.88 / 0.83 / 0.98, country 0.32 / 0.32 / 0.42, disco
0.65 / 0.79 / 0.68, hiphop 0.75 / 0.91 / 0.89, jazz 0.60 / 0.54 / 0.58,
metal 0.83 / 0.81 / 0.86, pop 0.27 / 0.47 / 0.57, reggae 0.51 / 0.57 / 0.56,
rock 0.45 / 0.34 / 0.15. The biggest change from the classical models is
**hiphop**: 0.13 to 0.19 for logreg, SVM and XGBoost (lesson 08), 0.75 to
0.91 for the CNN. Rhythmic patterns over about 1 s (the receptive field) are
exactly what the 57 window averages cannot describe. Country and rock stay
hard; in seed 0's confusion matrix 9 of 23 country clips are predicted as
rock.

---

## 7. Viva questions with model answers

**Q1. How many parameters does your CNN have, and why so few?**
241,898: 239,904 in the four 3 x 3 convolutions, 704 in BatchNorm, 1,290 in
the final layer. Weight sharing makes the conv layers independent of the
image size, and global average pooling replaces a dense layer that would
need 81,930 parameters after block 4. A fully connected layer on the raw input
would need 8.9 billion weights.

**Q2. What is the receptive field of your last layer, and why does it
matter?**
46 mel bins x 46 frames, about 1.07 s and a third of the mel range. Each
detected pattern spans at most that much. Global average pooling then
combines the 64 positions, so the decision uses the whole 3 s window, but
no single feature looks at more than about 1 s.

**Q3. Is translation equivariance appropriate for spectrograms?**
Along time yes: a drum pattern means the same thing anywhere in the window.
Along frequency only partly: a kick drum and a hi-hat are different
instruments partly *because* of their frequency. Our square filters share
weights along frequency anyway; the incomplete receptive field and zero
padding keep some position information. Frequency-aware designs use tall
filters or pool only over time.

**Q4. Why is the gain augmentation added before normalisation?**
A volume change of g dB adds g to every cell of the dB spectrogram, which is
what we add. After per-bin normalisation the same shift would be divided by
a different standard deviation in each bin (11.0 to 15.7 dB), which does not
correspond to any real change in the audio.

**Q5. What if you trained without augmentation?**
The network would see the same windows without variation and, with 241,898
parameters and only 442 clips, would memorise them faster: training loss
would fall faster and valid loss would turn up earlier. We did not run this
ablation; the expectation comes from the growing train/valid gap we already
see with augmentation.

**Q6. Your valid loss jumps between 1.3 and 3.6. Is training broken?**
No. Valid clip macro-F1 stays in the 0.4-0.6 range while the loss jumps,
so the predictions are mostly the same; the spikes come from a few windows
predicted wrongly with very high confidence, which cross-entropy punishes
heavily. That is a calibration problem (lesson 11), not a training failure.
Training loss falls smoothly.

**Q7. Why early-stop on clip macro-F1, and what does that do to the
reported valid score?**
It is our metric, so the epoch is chosen the same way as hyperparameters
for every other model. Because the kept epoch is the maximum of a noisy
curve, the valid score is optimistic. The test part, used once, gives the
unbiased number.

**Q8. Why three seeds, and why the median best epoch?**
The seed changes the result by up to 0.055 in valid F1 and the best epoch
from 8 to 26. One seed could not distinguish model differences from luck.
For a refit on train+valid there is no valid set to stop on, so the epoch
count must be fixed in advance; the median (13) is robust to the outlier
seed (26).

**Q9. Why does the CNN beat the SVM by so much on hiphop?**
Hiphop's distinctive cues are rhythmic: beats and their placement over
fractions of a second. The 57 handcrafted features average every frame of a
3 s window, which erases timing. The CNN sees the time-frequency image and
its filters respond to local onset patterns. Hiphop F1 goes from 0.13-0.19
(classical models) to 0.75-0.91.

**Q10. What would go wrong if you computed the normalisation statistics on
all clips?**
Valid and test spectra would shape the input scaling of the training data:
the leakage-lab mistake 1 in a CNN. Our stats come from the 442 training
clips only and are saved, so test clips will later be scaled with exactly
the same numbers.

---

## 8. Further reading

- LeCun, Bottou, Bengio & Haffner (1998), *Gradient-based learning applied
  to document recognition*, Proc. IEEE 86(11) (convolutional networks).
  https://doi.org/10.1109/5.726791
- Choi, Fazekas & Sandler (2016), *Automatic tagging using deep
  convolutional neural networks*, ISMIR (3 x 3 CNNs on mel spectrograms).
  https://arxiv.org/abs/1606.00298
- Pons, Lidy & Serra (2016), *Experimenting with musically motivated
  convolutional neural networks*, CBMI (filter shapes for time vs
  frequency). https://doi.org/10.1109/CBMI.2016.7500246
- Ioffe & Szegedy (2015), *Batch normalization*, ICML.
  https://arxiv.org/abs/1502.03167
- Srivastava et al. (2014), *Dropout: a simple way to prevent neural
  networks from overfitting*, JMLR 15. https://jmlr.org/papers/v15/srivastava14a.html
- Lin, Chen & Yan (2014), *Network in network*, ICLR (global average
  pooling). https://arxiv.org/abs/1312.4400
- Kingma & Ba (2015), *Adam: a method for stochastic optimization*, ICLR.
  https://arxiv.org/abs/1412.6980
- Loshchilov & Hutter (2019), *Decoupled weight decay regularization*, ICLR
  (AdamW). https://arxiv.org/abs/1711.05101
- Park et al. (2019), *SpecAugment: a simple data augmentation method for
  automatic speech recognition*, Interspeech. https://arxiv.org/abs/1904.08779
- Luo, Li, Urtasun & Zemel (2016), *Understanding the effective receptive
  field in deep convolutional neural networks*, NeurIPS.
  https://arxiv.org/abs/1701.04128
- Bouthillier et al. (2021), *Accounting for variance in machine learning
  benchmarks*, MLSys (seed variance). https://arxiv.org/abs/2103.03098
- Goodfellow, Bengio & Courville (2016), *Deep Learning*, ch. 7
  (regularisation, early stopping, dropout), ch. 8 (optimisation), ch. 9
  (convolutional networks). https://www.deeplearningbook.org/
- PyTorch docs: `torch.nn.Conv2d`, `BatchNorm2d`, `torch.optim.Adam`.
  https://pytorch.org/docs/stable/nn.html
