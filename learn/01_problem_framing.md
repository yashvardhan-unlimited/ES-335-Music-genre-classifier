# Lesson 01: Problem framing

Before writing any model code we pin down *exactly* what we are predicting,
from what, when, and what a score would and would not prove. Everything later
(splits, features, metrics) follows from these choices.

---

## 1. What we built

No code in this lesson; the framing is recorded in documents that later code
must obey.

| File | What it holds |
|---|---|
| `README.md` → *Question* | The research question in two sentences. |
| `DECISIONS.md` | First decision: split C (artist-disjoint) is the main result, with reasons and rejected alternatives. |
| `config.yaml` | The numbers the framing fixes: `audio.clip_seconds: 30`, `audio.window_seconds: 3`, `audio.n_windows: 10`, `split_sizes: {train: 443, valid: 197, test: 290}`, the 10 `genres`. |

Command to see the config the rest of the project will use:

```bash
.venv/Scripts/python -c "from src.utils import load_config; print(load_config())"
```

---

## 2. Concepts in depth

### 2.1 The question

> How well can models predict a 30-second clip's genre from audio alone, and
> how much of that accuracy survives an artist-aware split that removes leakage?

It has two halves. The first ("how well") is the usual benchmark question. The
second ("how much survives") is the scientific one: it asks whether the usual
benchmark number measures *genre recognition* or something easier.

### 2.2 Unit of prediction vs training row

These are different on purpose.

- **Unit of prediction = one 30 s clip.** That is what a user would hand the
  system, and it is what the GTZAN label describes. Every reported score counts
  clips.
- **Training row = one 3 s window.** Each clip is cut into 10 non-overlapping
  windows, and every window inherits its clip's label.

*Intuition.* Ten short views of a clip give the model ten times as many
examples to learn from, and genre cues (a drum pattern, a guitar timbre) are
visible in 3 s. At prediction time we combine the ten views so the decision is
about the whole clip.

*Math.* For clip $c$ with windows $w_1,\dots,w_{10}$, a model outputs a
probability vector $p(\cdot \mid w_i)$ over the 10 genres for each window. The
clip prediction is the **soft vote** (average of probabilities):

```latex
\hat{p}(g \mid c) = \frac{1}{10} \sum_{i=1}^{10} p(g \mid w_i),
\qquad
\hat{y}(c) = \arg\max_{g} \hat{p}(g \mid c)
```

*Worked example with this project's numbers.* `audio.sr = 22050` samples per
second, so a 30 s clip is $30 \times 22050 = 661{,}500$ samples and a 3 s window
is $3 \times 22050 = 66{,}150$ samples. With `mel.hop_length = 512` and
librosa's centred frames, one window becomes
$1 + \lfloor 66150 / 512 \rfloor = 130$ frames, so a log-mel window is a
$128 \times 130$ image (`mel.n_mels = 128`). Split C has 443 training clips,
so models train on $443 \times 10 = 4{,}430$ windows.

**The danger this creates:** windows from the same clip are near-copies of each
other. If any split assigns windows (rather than clips) to train and test, the
model sees part of every test clip during training. That is split A, kept only
to show how large this leak is.

### 2.3 Target

The target $y \in \{$blues, classical, country, disco, hiphop, jazz, metal,
pop, reggae, rock$\}$: one label per clip, taken from the GTZAN folder name.
It is a single-label, 10-class problem. With balanced classes, a model that
guesses has expected accuracy $1/10 = 10\%$; the `Dummy` baseline will measure
this on our real splits instead of assuming it.

Limits of the target: GTZAN labels were assigned by one person (Tzanetakis) and
genre is fuzzy (is a track "rock" or "metal"?). Sturm (2013) documents
mislabels. So even a perfect model cannot reach 100% against these labels, and
"accuracy" means "agreement with these particular labels".

### 2.4 Prediction time: what information exists then

Ask: *at the moment the system is used, what does it actually have?*

- **Available:** the 30 s of audio, sampled at 22,050 Hz, mono.
- **Not available:** artist name, album, file name, the clip's index in the
  dataset, other clips by the same artist, or the test set's label distribution.

Any feature or procedure that uses something on the second list is **leakage**:
it raises the score without improving the real system. Two subtle forms:

1. *Leakage through the split.* If an artist appears in train and test, the
   model can learn that artist's sound (voice, studio, mastering) and "recognise"
   the genre via the artist. In use, new music often comes from unseen artists,
   so this does not transfer.
2. *Leakage through preprocessing.* A scaler fitted on all data has looked at
   the test set's means and variances. Hence the rule: fit preprocessing on
   training data only, inside sklearn Pipelines.

### 2.5 Metric

Primary metric: **clip-level macro-F1**. For each genre $g$:

```latex
\mathrm{P}_g = \frac{TP_g}{TP_g + FP_g}, \qquad
\mathrm{R}_g = \frac{TP_g}{TP_g + FN_g}, \qquad
F1_g = \frac{2\,\mathrm{P}_g \mathrm{R}_g}{\mathrm{P}_g + \mathrm{R}_g}
```

```latex
\text{macro-F1} = \frac{1}{10} \sum_{g} F1_g
```

Macro-averaging gives every genre equal weight, so a model cannot hide a
failing genre behind good ones. Accuracy, per-genre F1 and the confusion
matrix are secondary.

**Uncertainty.** The split C test set has 290 clips, so a score has real
sampling noise. We report a 95% bootstrap CI (`bootstrap.n = 2000`), and we
resample **artists**, not clips, because clips by one artist are correlated: if
the model fails on one artist it tends to fail on all their clips.

### 2.6 Limits of the claim

What a result on split C *can* support: "on GTZAN-style 30 s excerpts from
artists not seen in training, model M agrees with the GTZAN labels with
macro-F1 X (95% CI [a, b])."

What it *cannot* support:
- that the model understands "genre" in general (GTZAN is ~1000 clips, mostly
  from before 2002, labelled by one person);
- that it works on other datasets, other eras, or other genre taxonomies;
- that differences between models smaller than the CI width are real.

### 2.7 How the rubric grades framing

The ES 335 rubric gives **10 points to problem formulation**. A clear framing
is one where a reader can answer: what is predicted, from what, at what time,
measured how, and what a result would mean. A project that reports "92%
accuracy" on a random split with no discussion of leakage has a weaker
formulation than one reporting a lower number on an honest split and explaining
the gap. Careful reasoning about *what the number means* is what is being
graded, not the size of the number.

---

## 3. Why this choice, and the alternatives we rejected

| Choice | Alternative rejected | Why |
|---|---|---|
| Predict per clip | Predict per window | Users and labels are per clip; window accuracy counts each clip 10 times and mixes easy and hard windows. |
| Train on 3 s windows | Train on full 30 s clips | 10x more rows; standard in GTZAN literature; full clips would leave 443 training examples for a CNN. |
| Soft vote | Majority (hard) vote | Keeps confidence information; avoids ties (e.g. 5 vs 5 windows). |
| Macro-F1 primary | Accuracy primary | Equal weight per genre; robust if a split is not perfectly balanced. |
| Split C main | Random split main | Removes artist leakage; matches the real use case; comparable with published work. |
| Same 443/197/290 sizes on A, B, B', C | Each split's natural size | Differences between splits then reflect leakage only, not training-set size. |
| Artist-cluster bootstrap | Clip bootstrap | Clips of one artist are not independent; clip bootstrap gives CIs that are too narrow. |

---

## 4. Code walkthrough

No model code yet. The framing appears in code as configuration:

- `config.yaml` lines 3-7: `audio` block. `n_windows: 10` must equal
  `clip_seconds / window_seconds`; later windowing code will check this.
- `config.yaml` lines 18-21: `split_sizes`. They sum to 930, the number of
  clips left after Sturm's fault filtering.
- `config.yaml` line 39: the ordered `genres` list. Its order defines the class
  index 0-9 used everywhere (labels, confusion matrices).
- `tests/test_utils.py` `test_config_loads`: asserts these framing numbers so an
  accidental edit fails the tests.

---

## 5. Pitfalls

1. **Reporting window accuracy as the result.** Inflated and not what a user
   experiences. Always aggregate to clips first.
2. **Splitting windows instead of clips** (split A). Test windows have siblings
   in training; the model partly memorises clips.
3. **Splitting clips but not artists** (split B). Same artist, same studio, often
   the same album in train and test; the model can learn the artist.
4. **Using the test set to choose anything.** Hyperparameters chosen on test are
   fitted to test noise; the score is then optimistic. We choose on validation,
   and the test set is locked until the `pre-test-freeze` tag exists.
5. **Fitting a scaler before splitting.** Leaks test statistics.
6. **Reading too much into small differences.** With 290 test clips, two
   models a couple of points apart may be indistinguishable; check the CIs.

---

## 6. Results from this step

No experiments yet. Fixed facts recorded:
- 10 genres, 10 windows per clip, 3 s windows of 66,150 samples.
- Split sizes 443 / 197 / 290 clips (930 total), identical for all four splits.

---

## 7. Viva questions with model answers

**Q1. Why is the unit of prediction a clip and not a window?**
The label and the user's input are both clips. Window scores count each clip
ten times and are not what a user sees. We train on windows for data volume,
then soft-vote the ten window probabilities into one clip prediction.

**Q2. Why should a random split overestimate accuracy on GTZAN?**
GTZAN has many clips per artist and some duplicates. A random split puts the
same artist in train and test, so the model can use artist-specific cues
(voice, production) that do not exist for unseen artists.

**Q3. What if we used the window-random split A as the main result?**
Windows of every test clip would be in training, so the model would partly be
recognising clips it has seen. The score would mostly measure memorisation.
We keep A only to quantify that leak.

**Q4. Why macro-F1 instead of accuracy?**
Macro-F1 averages per-genre F1, so every genre counts equally and a poor genre
cannot be hidden. On a perfectly balanced test set accuracy and macro-F1 are
often close, but split C is not guaranteed to be perfectly balanced per genre.

**Q5. What if two models differ by 1 point of macro-F1 on split C?**
That is likely within noise: the test set is 290 clips from a limited number of
artists. We compare CIs from the artist-cluster bootstrap and do not claim a
winner when they overlap heavily.

**Q6. Why bootstrap over artists rather than clips?**
Clips by one artist are correlated; resampling clips treats them as
independent and makes the CI too narrow. Resampling artists respects the
dependence structure of the data.

**Q7. What information may the model use at prediction time?**
Only the 30 s of audio. Not the file name, the artist, or statistics computed on
the test set.

**Q8. What if we kept the original 1,000 clips including faulty ones in C?**
Duplicates and mislabels would remain; duplicates across parts would leak, and
mislabelled test clips would cap the achievable score. The fault-filtered split
removes these, which is why it has 930 clips.

**Q9. Why give all splits the same sizes?**
So the only difference between A, B, B' and C is the kind of leakage. If C had
fewer training clips, a drop in score could just be less training data.

---

## 8. Further reading

- Tzanetakis & Cook (2002), *Musical genre classification of audio signals*,
  IEEE TSAP. https://doi.org/10.1109/TSA.2002.800560
- Sturm (2013), *The GTZAN dataset: its contents, its faults, their effects on
  evaluation, and its future use*. https://arxiv.org/abs/1306.1461
- Kereliuk, Sturm & Larsen (2015), *Deep learning and music adversaries*, IEEE
  Trans. Multimedia (the fault-filtered split). https://arxiv.org/abs/1507.04761
- Kaufman et al. (2012), *Leakage in data mining: formulation, detection, and
  avoidance*, ACM TKDD. https://doi.org/10.1145/2382577.2382579
- scikit-learn user guide, *Common pitfalls: data leakage*.
  https://scikit-learn.org/stable/common_pitfalls.html
