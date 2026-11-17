# Lesson 19: Presenting the results: one chart per question

The website already had the numbers. This step adds five charts so that a reader who never opens a
table can still see the answer. Each chart answers one question a reader actually asks, and each
chart type was picked because its geometry matches that question.

Nothing in this step changed a model, a setting, a prediction or a score. Every value drawn comes
from `docs/data/site.json`, which `scripts/export_site_data.py` wrote from the result files.

---

## 1. What we built

| file | change |
|---|---|
| `docs/index.html` | five new SVG charts drawn in plain JavaScript (no library): a 0-1 score scale in "The answer", a waterfall in "The leakage ladder", a paired-difference forest plot and a refit dumbbell in "Every model on split C", and a genre-by-model F1 heatmap in "Why the scores stop near 0.55". The step table under the ladder moved into a collapsible `<details>`, because the waterfall now shows the same numbers. |
| `DECISIONS.md` | dated entry: the five charts, why each form, what was rejected |
| `learn/glossary.md`, `learn/00_index.md` | new terms and this lesson |

How to view it (from the repo root; a server is needed because the page `fetch`es `data/site.json`,
which browsers block for `file://` pages):

```bash
cd docs
python -m http.server 8000      # then open http://localhost:8000
```

## 2. Concepts in depth

### 2.1 A chart is an answer to one question

Intuition: a reader looks at a chart for a few seconds. In that time they can judge **positions on a
common scale** very accurately, **lengths** well, and **colour shades** only roughly (Cleveland &
McGill 1984 ranked these "elementary perceptual tasks" by measured accuracy). So the most important
comparison in each chart is put on position or length, and colour is kept for the one chart (the
heatmap) where there are too many values to place on a line.

| reader's question | chart | what carries the answer |
|---|---|---|
| Is 0.55 good or bad? | score scale | position between chance (0.09) and perfect (1) |
| Where does the leaky score go? | waterfall | length of each orange bar |
| Is the CNN really ahead? | forest plot | which side of zero each interval lies on |
| Did refitting help? | dumbbell | direction and length of each line |
| Which genres are hard for everyone? | heatmap | dark or pale rows |

### 2.2 The waterfall: a total split into its parts

Intuition: the CNN scores 0.82 on the leakiest split A and 0.59 on split C. A reader wants to know
which kind of leakage accounts for which part of the 0.23 gap. A waterfall starts with a full bar at
the first total, hangs one floating bar per step, and ends with a full bar at the last total.

Math. With ladder means $m_A, m_B, m_{B'}, m_D, m_C$ (all train-only CNN, mean over runs), the steps
are differences of consecutive means, so they add up exactly:

```latex
m_A - m_C = (m_A - m_B) + (m_B - m_{B'}) + (m_{B'} - m_D) + (m_D - m_C)
```

Worked example (CNN, from `site.json` `ladder.cnn` and `drops`):

```latex
0.8188 - 0.0427 - 0.0283 - 0.0936 - 0.0599 = 0.5943
```

So the chart's bars are 0.82, −0.04 (A→B, within-clip), −0.03 (B→B′, faults), −0.09 (B′→D, artist),
−0.06 (D→C, published split vs random artist-disjoint draws), ending at 0.59. Only the artist step
has a 95% interval that excludes zero (0.061 to 0.159), which the tooltip shows. We checked before
drawing that the exported `drops[...].diff` equals the difference of the ladder means, so the chart
and the step table can never disagree.

### 2.3 The forest plot: a difference and its interval

Intuition: two models' separate intervals overlapping does **not** mean they are tied, and not
overlapping is not the only way to be different. The right object is the interval of the
**difference**, computed on the same bootstrap resamples (the paired bootstrap of lesson 17). A forest
plot draws one row per comparison: a dot at the difference and a line for its interval, with a
vertical line at zero.

Math. For bootstrap resample $b$ (artists drawn with replacement),

```latex
\Delta^{(b)} = F1_{\text{CNN}}^{(b)} - F1_{\text{other}}^{(b)}, \qquad
\text{CI}_{95} = [q_{0.025}(\Delta^{(b)}),\ q_{0.975}(\Delta^{(b)})]
```

The row is labelled "CNN ahead" if the lower end is above 0, "CNN behind" if the upper end is below
0, and "tie" otherwise. The same rule is used by the page's text (`tiedModels`), so the picture and
the sentences agree.

Worked example (test, split C, refit models): CNN − kNN = +0.119 (0.028 to 0.190): ahead. CNN − logistic
regression = −0.016 (−0.107 to 0.056), CNN − SVM = +0.011 (−0.083 to 0.076), CNN − XGBoost = +0.034
(−0.065 to 0.098): all three ties. The dummies are left out of this chart: their differences (+0.46
and +0.53) would stretch the axis and squash the four rows that matter; the text says every model is
far above them.

### 2.4 The dumbbell: before and after for each item

Intuition: refitting on 639 instead of 442 clips moved each model by a few hundredths, some up, some
down. Two dots per row (hollow = trained on 442, filled = refit on 639) joined by a line show
direction and size at once.

Worked example (test macro-F1, train-only → refit): kNN 0.453 → 0.433 (−0.02), logistic regression
0.515 → 0.568 (+0.05), SVM 0.488 → 0.541 (+0.05), XGBoost 0.482 → 0.518 (+0.04), CNN 0.594 → 0.552
(−0.04). Two models got worse after refitting, kNN and the CNN, and their filled dots sit left of
their hollow ones.

### 2.5 The heatmap: a table you can read by shade

Intuition: 10 genres × 7 runs is 70 numbers. As a table it must be read cell by cell; as a heatmap
the eye finds pale rows (hard genres) at once. Rows are sorted hardest first, so the plateau
argument ("each genre is learned from a handful of artists") is visible at the top of the chart.

Colour: a **sequential** scale (one hue, light to dark) because F1 is a magnitude from 0 to 1 with
no meaningful midpoint. A diverging scale (two hues around a centre) would suggest that some value,
say 0.5, is a natural "neutral", which it is not. Each cell's number is printed when the cell is wide
enough, so the colour is never the only way to read a value.

The "mean" column uses exactly the definition of the plateau table below it:

```latex
\bar F_g = \tfrac{1}{5}\Big( F_g^{\text{kNN}} + F_g^{\text{LR}} + F_g^{\text{SVM}} + F_g^{\text{XGB}}
  + \tfrac{1}{3}\textstyle\sum_{s=0}^{2} F_g^{\text{CNN},s} \Big)
```

Worked example: rock = (0.09 + 0.21 + 0.19 + 0.21 + (0.11 + 0.00 + 0.24)/3) / 5 = 0.16, the same
0.16 as in the table. Blues is the instructive exception: 0.05 to 0.10 for the four classical models,
but 0.36 to 0.68 for the three CNN runs, so blues is hard for the handcrafted features, not for
every model.

### 2.6 Zero baselines: when an axis may start above zero

Bars encode a value by **length**, so a bar axis must start at 0, or a bar twice as long would not
mean twice the value. The waterfall and the existing model bar chart start at 0. Dots encode by
**position**, so a dot chart may zoom: the dumbbell's axis runs 0.40 to 0.60 to make differences
of 0.02 to 0.05 visible, and the forest plot is centred on the range of differences with zero always
included.

## 3. Why this choice, and the alternatives we rejected

- **All five charts in one hue (plus orange for drops), not a colour per model.** Each chart compares
  models by position, and models are named by text labels. Adding seven categorical colours would
  add a legend to decode and would need checking for colour-blind separation, for no extra
  information.
- **Waterfall rejected alternative: a single stacked bar** of 0.82 split into "survives on C" plus
  four leakage segments. It breaks when a step is negative (the score can rise between splits; for
  some models it does), and segments of 0.03 are too thin to label on a phone. A waterfall draws a
  rise as an upward light-blue bar with no special case.
- **Forest plot rejected alternative: the existing bars with overlapping intervals.** Overlap of two
  separate intervals is the wrong test (lesson 18, paired vs overlapping intervals).
- **Heatmap rejected alternative: small-multiple bar charts of per-genre F1, one per model.** Seven
  panels of ten bars make the reader compare across panels, which is harder than comparing along a
  row.
- **No charting library.** The page already draws its charts in about 40 lines of SVG code each;
  a library (D3, Chart.js) would add a dependency and a second style for five charts.

## 4. Code walkthrough (`docs/index.html`)

- **L909-915 `xGrid`**: draws vertical grid lines and tick labels for horizontal charts. The line at
  0 gets the darker `baseline` class.
- **L916 `range`**: tick values from `a` to `b` in steps `s`, rounded to four decimals so 0.1 + 0.2
  does not print as 0.30000000000000004.
- **L917 `label`**: one SVG text node; a `weight` makes it bold and full ink colour. Text never uses
  a series colour, so it stays readable in both themes.
- **L923 `renderScale`**: the 0-1 track is a thick rounded line; the 95% interval is a lighter
  segment drawn over it; three dots mark chance (`headline.floor`), the headline and split A
  (`ladder.cnn.A`). The headline label sits above the track and the others below, so labels cannot
  collide even on a phone.
- **L943-944 `STEPS`, `renderFall`**: `cols` lists A, the four steps and C. For a step, the bar runs
  from the previous mean to the next (`hi`, `lo`); its colour is orange when the score falls and light
  blue when it rises. The dashed line joins each bar's end level to the next bar. The tooltip reads
  the interval from `S.drops`, and negates it because a drop is a negative change.
- **L978 `renderPairs`**: the axis limits are the interval ends rounded outward to 0.05 and always
  include −0.05 to 0.05, so zero is never at the edge. The verdict text uses the same rule as
  `tiedModels`. A transparent full-width `rect` per row is the hover and keyboard target, larger than
  the dot.
- **L1004 `renderRefitChart`**: hidden when the page shows validation results (`fit_on_C` is not
  `train+valid`), because there is then no refit to compare. The legend is drawn inside the SVG
  because two series need one.
- **L1032-1040 `HEAT_COLS`, `genreMean`, `renderGenreHeat`**: `genreMean` reproduces the plateau
  table's averaging; rows are sorted by it. Each cell's colour step is
  `round(F1 × (steps − 1))` on the page's existing 13-step blue ramp (`--seq`), and the text turns white
  on the darker steps for contrast. Numbers are hidden only when cells are narrower than 30 px; the
  tooltip still gives every value.
- Hooks: `renderAnswer` calls `renderScale`, `renderLadder` calls `renderFall` (so the waterfall
  follows the model buttons), `renderModels` calls `renderPairs` and `renderRefitChart`, and
  `renderAll` calls `renderGenreHeat`. Every chart is redrawn on resize and theme change, because
  colours are read from CSS variables at draw time.

## 5. Pitfalls

- **Computing a number in the page that differs from the exported one.** The waterfall uses ladder
  means; we checked that `drops[...].diff` equals their differences (0.0427, 0.0283 and 0.0936
  exactly; 0.0598 against 0.0599 for D→C, a rounding difference). If the export changed to a paired-by-seed difference, the bars and the
  table could disagree; the tooltip shows the exported interval for that reason.
- **A caption claiming more than the chart shows.** The first draft of the heatmap caption said the
  same rows are pale for every model. Blues contradicts it (CNN 0.36 to 0.68), so the caption names
  blues as the exception. Write captions after looking at the data, not before.
- **Truncating a bar axis.** Starting the waterfall's axis at 0.5 would make the −0.03 step look
  like a third of the score.
- **Colour as the only code.** Without printed numbers or a table, readers with colour-vision
  deficiency, or anyone printing in grey, could not read the heatmap.
- **Mixing train-only and refit numbers without saying so.** The scale chart puts the refit headline
  (0.55) beside the train-only split A score (0.82); the caption says so. The waterfall uses
  train-only numbers throughout, so its C bar is 0.59, not the headline 0.55.

## 6. Results from this step

No new numbers: every value is from `docs/data/site.json` (test, generated 2026-11-17). The charts
show the CNN waterfall 0.82 → 0.59 with the artist step (−0.09) the largest; the CNN tied with logistic
regression, the SVM and XGBoost and ahead of kNN; refitting helping logistic regression, the SVM and
XGBoost (+0.04 to +0.05) and lowering kNN (−0.02) and the CNN (−0.04); rock (mean F1 0.16) and blues
(0.17) the hardest genres, classical (0.94) and metal (0.78) the easiest. The page was rendered in
headless Edge at 1100 px and at its minimum width (481 px layout), with no horizontal overflow.

## 7. Viva questions with model answers

1. **Why a waterfall and not just the dot chart above it?** The dot chart shows levels; the
   waterfall shows the change at each step as a length, which is the quantity the ladder exists to
   measure. They answer different questions with the same numbers.
2. **The forest plot says "tie" for the SVM, but the bar chart shows the CNN 0.01 higher. Which is
   right?** Both: 0.01 is the point estimate, and the paired 95% interval (−0.083 to 0.076) contains
   zero, so the data cannot say which is better on this test set.
3. **Why does the dumbbell axis not start at zero?** It encodes values by position, not length, so a
   zoomed axis does not distort ratios; at 0 to 1 the 0.02 to 0.05 changes would be invisible.
4. **What if you used a red-to-blue diverging scale in the heatmap?** It would imply a meaningful
   midpoint (for example F1 = 0.5) and make cells just below it look "bad". F1 has no such neutral
   value here; a sequential scale is correct.
5. **What if a step in the waterfall were positive?** It is drawn as a light-blue bar rising from the
   previous level, and its label shows a plus sign. The stacked-bar alternative could not show this.
6. **Why leave the dummies out of the forest plot?** Their differences (about +0.46 and +0.53) would
   set the axis range and compress the four comparisons that are actually in doubt; the page states
   separately that all models beat them.
7. **Why sort heatmap rows by the mean rather than alphabetically?** The question is "which genres
   are hard", so ordering by difficulty puts the answer at the top. Alphabetical order would scatter
   the pale rows.
8. **The scale chart shows 0.82 next to 0.55. Is that a fair comparison?** Only as context: 0.82 is the
   train-only CNN on split A, and 0.55 is the refit CNN on C. The like-for-like train-only gap is
   0.82 → 0.59, shown in the waterfall, and the caption says which is which.

## 8. Further reading

- W. S. Cleveland and R. McGill, "Graphical perception: theory, experimentation, and application to
  the development of graphical methods," JASA, 1984. https://doi.org/10.1080/01621459.1984.10478080
- C. O. Wilke, *Fundamentals of Data Visualization* (free online), chapters on visualising
  uncertainty and on colour scales. https://clauswilke.com/dataviz/
- T. Munzner, *Visualization Analysis and Design*, CRC Press, 2014 (marks, channels, effectiveness).
- MDN, SVG tutorial (the drawing primitives the page uses). https://developer.mozilla.org/en-US/docs/Web/SVG/Tutorial
- G. Cumming, "The new statistics: why and how," Psychological Science, 2014 (plotting estimates
  with intervals). https://doi.org/10.1177/0956797613504966
