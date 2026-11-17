# Lesson 18: Post-freeze corrections, the refit result, and defending the project

After the one test run (lesson 17), the website was re-read against the result files. Several
sentences said more than the numbers support, and one label was wrong. This lesson explains what was
corrected and why, what the refit result means, how to read errors that cluster by artist, and ends
with the viva questions most likely to come up about the whole project.

Nothing in this step changed a model, a setting, a prediction or a score.

---

## 1. What we built

| file | change |
|---|---|
| `scripts/export_site_data.py` | exports, per model on C: test accuracy, C valid macro-F1, the train-only test macro-F1 (`c_extras`); the train-only paired comparisons (`paired_train_only`); the artist of every clip in the confusion-matrix lists; two more figures (`leakage_lab.png`, `rms_by_genre.png`) |
| `docs/index.html` | corrected comparator label and answer text; leakage gap split into B′ → D and D → C; new refit paragraph; written error analysis; accuracy and validation columns; cold-start motivation; new limitations; contribution statement; four more figures |
| `src/projectpage.py` | the Streamlit Project tab shows the new artist column in its clip table |
| `DECISIONS.md` | the stale "four splits, 443/197/290" text is marked superseded; a dated post-freeze entry records every correction and the refit result |

Commands (from the repo root, `python` = the venv's Python):

```bash
python scripts/export_site_data.py     # docs/data/site.json and docs/img/ from results/test
python -m pytest -q                     # 151 tests, including the Project tab rendering site.json
python scripts/check_lesson_refs.py     # lesson line references still point at their code
```

The site itself has no build step: push `docs/` and GitHub Pages serves it.

---

## 2. Concepts in depth

### 2.1 What may change after the test run

**Intuition.** A test set is only an honest estimate if nothing was chosen by looking at it. Once
test has been scored, every choice that touches a model is frozen, because any change would be
made *knowing* the test result. Wording is different: correcting a sentence that misreports a fixed
number does not choose anything. The protocol (DECISIONS.md, 2026-11-14) says exactly this: a
problem found after the test run is fixed in a commit marked post-freeze, and both versions are
reported.

| allowed after the freeze | not allowed |
|---|---|
| fixing a wrong label or an overclaim | retuning, retraining, changing the epoch count |
| reporting a result that was computed but not shown (the train-only scores) | switching the headline to the number that looks better |
| writing an error analysis of the fixed predictions | dropping clips or genres that score badly |

**Why "switch the headline to the train-only CNN" is forbidden.** The train-only CNN scores 0.594
on C test, the refit CNN 0.552. Choosing 0.594 *because* it is higher is selection on test, the
same mistake as tuning on test. Its value would be biased upward by the choice itself.

### 2.2 Paired comparison: why "the CNN beats the SVM" was wrong

**Intuition.** Two models scored on the same 290 clips share the same easy and hard clips. If a
hard artist lands in a bootstrap resample, *both* models score lower. Comparing their separate
intervals counts that shared difficulty twice. A paired bootstrap resamples artists once and
computes the *difference* on each resample, so shared difficulty cancels.

**Math.** Let $b = 1..B$ index resamples of artist clusters ($B = 2000$), and $F_m^{(b)}$ the
macro-F1 of model $m$ on resample $b$:

```latex
\Delta^{(b)} = F_{\text{CNN}}^{(b)} - F_{\text{SVM}}^{(b)}, \qquad
\text{CI}_{95\%} = \big[\, q_{0.025}(\Delta^{(b)}),\; q_{0.975}(\Delta^{(b)}) \,\big], \qquad
\text{share} = \tfrac{1}{B}\textstyle\sum_b \mathbf{1}[\Delta^{(b)} > 0]
```

For the CNN, $F_{\text{CNN}}^{(b)}$ is the mean of its three runs on the same resample
(`scripts/evaluate_part.py:103`, `seed_means`).

**Worked example (C test, final models).** CNN − SVM = +0.011, 95% interval −0.083 to +0.076,
and the CNN is ahead in 52% of resamples: a coin flip. The old site said the CNN "beats" the SVM.
Against logistic regression the difference is −0.016 (−0.107 to +0.056); against XGBoost +0.034
(−0.065 to +0.098). Only kNN is clearly behind (+0.119, +0.028 to +0.190). So the honest sentence
is "no clear winner among logistic regression, the SVM and XGBoost".

### 2.3 Splitting a gap with a controlled ladder

**Intuition.** The step B′ → C changed two things at once: artist leakage was removed, *and* five
random draws were swapped for one fixed published partition. Split D (lesson 16) sits between them:
same pool and randomness as B′, but artist-disjoint. So the B′ → C gap splits into two parts, each
caused by one change.

**Math.** With $\bar F_S$ the mean macro-F1 over runs on split $S$:

```latex
\bar F_{B'} - \bar F_{C} \;=\; \underbrace{(\bar F_{B'} - \bar F_{D})}_{\text{artist leakage}}
\;+\; \underbrace{(\bar F_{D} - \bar F_{C})}_{\text{published vs random partition}}
```

**Worked example (CNN, train-only, test).** B′ 0.748, D 0.654, C 0.594. Artist leakage
B′ − D = 0.094 (95% interval 0.060 to 0.159); partition D − C = 0.060 (−0.042 to +0.153).
Together 0.153 up to rounding. The old site called all 0.153 "leakage"; only 0.094 is, and the
0.060 is within noise. Every model shows the same direction (D − C between 0.045 and 0.081, all
intervals including zero), so C may be a somewhat hard partition, but one fixed split cannot show
that for certain.

Both sides of these differences must come from the same kind of model. The old summary put the
refit headline (0.552) next to a gap computed from train-only models, so a reader adding them got
0.70, which is no number we measured.

### 2.4 The refit result

**What happened.** The protocol fixed before the tag said: refit each final model on train + valid
(639 clips), with the settings chosen on valid. On C test:

| model | train only | refit on train + valid |
|---|---|---|
| CNN (mean of 3 runs) | 0.594 | 0.552 |
| CNN seeds 0 / 1 / 2 | 0.567 / 0.607 / 0.609 | 0.550 / 0.576 / 0.530 |
| SVM | 0.488 | 0.541 |
| logistic regression | 0.515 | 0.568 |

Every CNN seed got worse; the classical models got better. Train-only, CNN − SVM is +0.107
(+0.022 to +0.197, CNN ahead in 99.5% of resamples); refit, +0.011. So "which model is best"
depends on the refit choice.

**Two hypotheses (neither tested).**

1. *The epoch count does not transfer.* An epoch is 10 random crops per training clip in batches of
   64, so the number of gradient steps per epoch grows with the training set:

   ```latex
   \text{steps per epoch} = \left\lceil \frac{10\,N}{64} \right\rceil:\quad
   N = 442 \Rightarrow 70, \qquad N = 639 \Rightarrow 100
   ```

   Fourteen epochs are 980 steps on train but 1,400 on train + valid, 43% more training at the same
   learning rate. Whether that is "too much" or simply "different" is unknown.
2. *The train-only model was a selected checkpoint; the refit model is not.* Validation macro-F1
   swings by up to 0.25 between consecutive epochs (lesson 17, DECISIONS 2026-11-14). The train-only
   run keeps the epoch that scored best on valid. The refit run stops at the same epoch *number*
   but lands wherever its own oscillating trajectory happens to be. A classical model has no
   trajectory, so refitting it simply adds data.

Testing either would mean new training runs chosen after seeing test, so they stay hypotheses.
The site says "a likely reason, not tested".

### 2.5 Errors that cluster by artist

**Intuition.** Clips by one artist share recording era, production, instruments and voice. If the
model misreads that sound, it misreads all of that artist's clips. Errors are then not 290
independent events but closer to 119 (the number of artist clusters in C test).

**Math.** The usual *design effect* for clusters of average size $m$ with within-cluster correlation
$\rho$ gives the effective sample size

```latex
n_{\text{eff}} = \frac{n}{1 + (m - 1)\,\rho}
```

With $n = 290$ clips in 119 clusters, $m \approx 2.4$; at the extreme $\rho = 1$ (an artist's clips
always succeed or fail together), $n_{\text{eff}} = 119$. This is why intervals resample artists:
resampling clips would pretend we had 290 independent pieces of evidence.

**Worked example (final CNN seed 0, C test, artists from gtzan-bind).**
- 20 of 32 rock clips are called country; 15 of those 20 are Led Zeppelin (9 of their 11 clips) or
  the Rolling Stones (6 of 7).
- Every Tony Williams clip (8) and every Dexter Gordon clip (6), both jazz, is called country.
- C's test blues is three artists: Robert Johnson (17 clips), Clifton Chenier (12), Albert Collins
  (2). The SVM calls 18 of the 31 blues clips jazz; logistic regression 17. Their blues F1 is 0.10 or
  lower (kNN included); the three CNN runs reach 0.36 to 0.68.
- Counter-example: the 15 disco clips called blues come from 15 different artists, so that error is
  about the genre (or the model's idea of it), not one performer.

Consequence for interpretation: "the CNN is good at blues" really means "the CNN handles Robert
Johnson and Clifton Chenier", two performers.

### 2.6 Naming the comparator

The test protocol named the RBF SVM as the classical comparator before the tag. Logistic regression
turned out higher on C valid (0.468 vs 0.458) and on C test (0.568 vs 0.541). The site therefore
shows both: the SVM labelled as the pre-registered comparator, and the best classical model *on
test* labelled as such. Calling whichever model wins on test "the comparator" would be the winner's
curse (lesson 13): the best of several noisy scores is biased upward.

### 2.7 Why the scores plateau near 0.55

**Intuition.** Logistic regression, the SVM, XGBoost and the CNN are very different learners, yet
they land between 0.52 and 0.57 on C test. When different models stop at the same place, suspect
the data. Here the data gives each genre very few *artists*: what a model calls "rock" is the sound
of the bands it was shown.

| genre | artists in C train | C test is mostly | mean F1 (5 models) |
|---|---|---|---|
| rock | 7 (Queen, Morphine, The Stone Roses, Sting, Jethro Tull, ...) | Led Zeppelin, the Rolling Stones, Ani DiFranco | 0.16 |
| blues | 7 (Hot Toddy, Kelly Joe Phelps, Stevie Ray Vaughan, Buckwheat Zydeco, ...) | Robert Johnson, Clifton Chenier | 0.17 |
| jazz | 15 (mostly Joe Lovano, Coleman Hawkins) | Tony Williams, Dexter Gordon, Miles Davis, Joe Henderson | 0.47 |
| classical | 29 | many performers | 0.94 |

Counts are distinct gtzan-bind names in `splits/metadata.csv` (credit variants separately); mean F1
is the final models' per-genre test F1 averaged over kNN, logistic regression, SVM, XGBoost and the
CNN's three-run mean.

**Checking the claim instead of trusting it.** If "few artists" were the whole story, F1 would rise
with the number of training artists across all genres. Spearman's rank correlation measures that:

```latex
\rho_s = 1 - \frac{6 \sum_{g} d_g^2}{n (n^2 - 1)}, \qquad
d_g = \operatorname{rank}(\text{artists}_g) - \operatorname{rank}(\text{F1}_g), \quad n = 10
```

Measured: $\rho_s = 0.50$ for the five-model mean ($p = 0.14$), 0.58 for the SVM, 0.36 for the CNN.
Positive but weak, and with only 10 genres not significant. Disco has 25 training artists and is
middling (0.44); classical is easy partly because it sounds unlike every other genre. So the honest
claim is: few artists is a likely main cause for rock and blues, not a law across genres.

Supporting evidence: the CNN scores 0.75 with artists shared (B′) and 0.65 without (D); errors
arrive in whole artists (2.5); the SVM's learning curve still rises with more training artists.

**What would help (not run, test is frozen):** pretrained audio embeddings (knowledge of many more
artists; possible overlap with GTZAN's sources), more artists per genre from a larger dataset,
a steadier CNN (seed averaging, learning-rate decay, artist-grouped CV for epochs, refit by steps),
augmentation that hides recording sound, richer temporal features.

---

## 3. Why this choice, and the alternatives we rejected

- **Correct the text, keep every number.** Rejected: leaving the overclaims (a grader reading the
  paired intervals would see the contradiction); deleting the paragraph (hides the main comparison).
- **Report both train-only and refit models.** Rejected: showing only the refit headline (hides that
  the model ranking depends on it); switching the headline to train-only (selection on test, 2.1).
- **Split the gap with D.** Rejected: one "B′ − C leakage" number (confounds two changes, 2.3).
- **Write the error analysis from fixed predictions with artist names.** Rejected: only the
  automatically generated "most frequent mistake" sentence (says *what*, never *why*); listening
  tests (no time before the deadline, and would need a protocol of their own).
- **Show the leakage lab and the loudness figure.** They were computed before the freeze on
  validation and data only; showing them adds evidence without new choices.
- **Not done:** a refit with a scaled epoch count, or ensembling the three CNN seeds. Both are new
  models chosen after test.

---

## 4. Code walkthrough

**`scripts/export_site_data.py`**

- `c_extras` (line 63): for each model on C, the mean test accuracy over its runs (from
  `results/test/scores.csv`, rows `set == "final"` once the test run exists), the mean C valid
  macro-F1 (from `results/val/summary.csv`), and the train-only test macro-F1 (`head["ladder"]`,
  key `C/<model>`). Means over runs matter only for the CNN (three runs); the rest have one.
- `c_block` (line 73) merges those fields into each model's row, so the page's table can show them.
- `confusion_runs` (line 84): line 87 reads `splits/metadata.csv` and maps clip id to artist
  (`""` when gtzan-bind has none); line 99 stores each cell as `[clip id, top score, artist]`.
- `main` (line 122): line 130 adds `paired_train_only`, the paired CNN comparisons of the train-only
  models, which the refit paragraph quotes.

**`docs/index.html`** (script section)

- `CLASSICAL`, `signed` (lines 617-618): the four classical models; a number with an explicit sign
  and a true minus sign, for differences.
- `bestClassical` (line 619): the classical model with the highest test macro-F1; drives the tile.
- `tiedModels` (line 622): classical models whose paired interval against the CNN includes zero.
  The answer, the summary and the takeaway all call it, so the three cannot disagree. The old
  takeaway compared *separate* intervals for overlap; this is the paired test of 2.2.
- `listText` (line 627): "a, b and c", lower-casing "Logistic regression" mid-sentence.
- `ladderSentence` (line 631): B′, D and C means of the train-only CNN, then the B′ → D and D → C
  drops with their intervals, adding "within noise" when an interval includes zero.
- `renderRefit` (line 749): the refit paragraph from `train_only_macro_f1`, `macro_f1` and the two
  paired comparisons; falls back to one sentence if the site shows validation numbers.
- `showCell` (line 810): appends the artist to each clip in the list.

**`scripts/evaluate_part.py`** (unchanged, for reference): `seed_means` (line 103) averages the
runs and their bootstrap arrays; `drops` (line 118) and `paired_on_C` (line 130) compute the
differences used above.

---

## 5. Pitfalls

- **Fixing results after the test.** The temptation after a disappointing number is to "fix" the
  model. Any such fix is a choice informed by test. Fix wording, never models.
- **Mixing model families in one sentence.** Refit headline + train-only gap = a number nobody
  measured. Always compare like with like.
- **Hard-coded labels.** "Best classical model (RBF SVM)" was typed before the test run and became
  false after it. Labels that depend on results should be computed from them.
- **Overlapping intervals as a test.** Two separate 95% intervals can overlap while the paired
  difference is clearly positive, and vice versa. Use the paired interval.
- **Reading a genre's F1 as a genre property.** With 1-3 artists per genre in a part, it describes
  those artists.
- **Over-reading untested explanations.** The refit hypotheses and the lopsided-interval
  explanation are labelled "likely" and "not tested". Do not state them as facts in the viva.

---

## 6. Results from this step

All numbers from `results/test/headline.json`, `results/test/scores.csv`,
`results/test/predictions/`, `results/val/summary.csv`.

| model on C | test macro-F1 (final) | 95% interval | test accuracy | valid macro-F1 | test macro-F1 (train only) |
|---|---|---|---|---|---|
| logistic regression | 0.568 | 0.475 to 0.641 | 0.583 | 0.468 | 0.515 |
| CNN (3 runs) | 0.552 | 0.467 to 0.583 | 0.581 | 0.620 | 0.594 |
| RBF SVM | 0.541 | 0.459 to 0.611 | 0.555 | 0.458 | 0.488 |
| XGBoost | 0.518 | 0.433 to 0.581 | 0.538 | 0.419 | 0.482 |
| kNN | 0.433 | 0.347 to 0.495 | 0.445 | 0.424 | 0.453 |
| dummy (stratified) | 0.092 | 0.052 to 0.118 | 0.100 | 0.128 | 0.103 |

Paired, final models: CNN − SVM +0.011 (−0.083 to +0.076); CNN − logistic regression −0.016
(−0.107 to +0.056). Train-only: CNN − SVM +0.107 (+0.022 to +0.197).

CNN ladder (train only, test): B′ 0.748, D 0.654, C 0.594; B′ − D 0.094 (0.060 to 0.159),
D − C 0.060 (−0.042 to +0.153).

Figures now on the site: `docs/img/rms_by_genre.png`, `docs/img/leakage_lab.png`,
`docs/img/permutation_importance.png`, `docs/img/analysis_calibration.png` (with
`mel_per_genre.png`, `val_svm_heatmap.png`, `analysis_learning_curves.png` from before).

---

## 7. Viva questions with model answers

**Q1. In one sentence, what is your answer to the question?**
On artists it has never heard, the best models reach a clip macro-F1 of about 0.55 (CNN 0.55,
interval 0.47 to 0.58), far above the 0.09 chance floor; when artists are shared between train and
test, the same CNN scores 0.75, and about 0.09 of that is artist leakage.

**Q2. Why is the CNN the headline if logistic regression scored higher on test?**
The headline model was fixed in the protocol before the test run: the CNN, mean of three runs,
refit on train + valid. Picking the model that happens to win on test would bias the headline
upward. And the difference (−0.016, −0.107 to +0.056) is not distinguishable from zero.

**Q3. Why did refitting on more data make the CNN worse?**
We don't know; it was not tested. Two candidates: the epoch count chosen for 442 clips gives 43%
more gradient steps on 639 clips, and the train-only model was the best-on-valid checkpoint of an
oscillating trajectory while the refit model stops at a fixed epoch number. Testing either needs new
training runs after test, so we report the result and leave it open.

**Q4. Why resample artists, not clips, in the bootstrap?**
Clips by one artist succeed or fail together (all 8 Tony Williams clips are called country). They
are not independent, so resampling clips treats ~119 pieces of evidence as 290 and gives intervals
that are too narrow.

**Q5. What does split D add that B′ and C do not?**
B′ → C removes artist leakage *and* switches to one fixed published partition. D keeps B′'s pool,
sizes and randomness and removes only artist leakage, so B′ → D measures leakage alone (0.094 for
the CNN) and D → C measures the published partition (0.060, within noise).

**Q6. Why are all splits exactly 442/197/290?**
So that a lower score on a cleaner split cannot be explained by less training data. Leakage is the
only systematic difference.

**Q7. What if you had used k-fold cross-validation over windows to tune?**
The leakage lab measured it: k-fold over training windows reports 0.973 for the SVM, because every
held-out window has sibling windows of the same clip in training; the setting it picks really
scores 0.433 on valid.

**Q8. What if you normalised each clip's loudness (per-clip normalisation) instead of a fixed dB
reference?**
The models would lose absolute level, which differs by genre (classical around −28 dBFS, pop around
−13). Scores would likely fall for genres separated by loudness, and the remaining accuracy would
rely more on musical content. It would need retraining everything and was not done; it is the main
"shortcut" limitation.

**Q9. What if you used 5 s windows?**
On validation the SVM scored 0.477 at 5 s against 0.458 at 3 s, a gap of a few clips, and the
settings were tuned at 3 s. Switching after seeing that would be selection on noise and would
invalidate all tuning, so 3 s was kept (DECISIONS 2026-10-30).

**Q10. Why macro-F1 and not accuracy?**
Accuracy hides a failing genre: the classical models have blues F1 of 0.10 or lower while logistic regression, the SVM and XGBoost have accuracy
0.54-0.58. Macro-F1 weights every genre equally. Accuracy is still reported as a secondary
number.

**Q11. Is the 0.06 D → C gap real?**
Not shown: its interval is −0.04 to +0.15. All five models point the same way (0.045 to 0.081), so
C may be a somewhat hard partition, but one fixed split cannot establish it.

**Q12. What did you do, and what did the AI do?**
Answer in your own words from `AI_USE.md` and the "Who did what" section of the site: which
choices were yours (question, main split, metric, decisions), and how you checked the code it wrote
(tests, rebuilding results, reading diffs).

**Q13. What if the test set had been used to pick the CNN's epoch?**
The test score would be optimistic in the same way the validation score is (CNN valid 0.620 vs test
0.594, train-only), and there would be no untouched data left to measure it.

**Q14. Why are your models stuck around 0.55, and what would you do next?**
Different model families tie at the same level, which points at the data: each genre is learned
from few artists (rock and blues from 7 each), and the test part has different ones; those two
genres score worst (mean F1 0.16 and 0.17). The link across all ten genres is weak (Spearman 0.50,
not significant), so it is a main cause, not the only one. Next: pretrained audio embeddings and
more artists per genre; for the CNN, seed averaging and a steadier training schedule.

---

## 8. Further reading

- B. L. Sturm, "The GTZAN dataset: Its contents, its faults, their effects on evaluation, and its
  future use," 2013. https://arxiv.org/abs/1306.1461
- C. Kereliuk, B. L. Sturm, J. Larsen, "Deep learning and music adversaries," IEEE Trans.
  Multimedia 17(11), 2015 (the fault-filtered split). https://doi.org/10.1109/TMM.2015.2478068
- A. Flexer, "A closer look on artist filters for musical genre classification," ISMIR 2007 (why
  artist-filtered evaluation lowers scores).
- B. Efron and R. Tibshirani, *An Introduction to the Bootstrap*, Chapman & Hall, 1993 (percentile
  intervals, paired comparisons).
- B. A. Nosek et al., "The preregistration revolution," PNAS 115(11), 2018 (why choices are fixed
  before seeing results). https://doi.org/10.1073/pnas.1708274114
- S. Kapoor and A. Narayanan, "Leakage and the reproducibility crisis in machine-learning-based
  science," Patterns 4(9), 2023. https://doi.org/10.1016/j.patter.2023.100804
