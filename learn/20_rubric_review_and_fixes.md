# Lesson 20: Reviewing the project against the rubric, and closing the gaps

The project was graded strictly against the ES 335 rubric (80 shared points for ML reasoning, 20 for
the website and the viva), reading the **published** website and the repository on GitHub as a
grader would. This lesson records what was missing, what was added, and why each addition answers a
specific rubric line. As in lessons 18 and 19, nothing here changed a model, a setting, a prediction
or a score: every number added to the site was already in a result file.

---

## 1. What we built

| file | change |
|---|---|
| `docs/index.html` | (a) a "prediction time" paragraph in the Question; (b) gtzan-bind linked, and artists per genre (14 to 62) in the Data table; (c) the window-length result and the reason 3 s was kept; (d) the models table now gives each model's objective, key assumption, and the grid searched with the setting chosen on C; (e) a "why macro-F1" paragraph with the dummy example; (f) a multiple-comparisons note under the paired-difference chart; (g) the hero's untested explanation of the lopsided interval is labelled as untested; (h) the six-step pipeline list and the future-work list are collapsible, to shorten the page; (i) gtzan-bind and the software libraries in the references |
| `README.md` | gtzan-bind and the main libraries added to Citations |
| `DECISIONS.md` | dated entry for this review |
| `learn/19_presenting_results_charts.md` | line references updated (the page grew by 21 lines above the chart code) |

Check commands (repo root):

```bash
python -m pytest -q                 # 151 tests
cd docs && python -m http.server    # open http://localhost:8000 and read the page as a grader
```

The artist counts were computed with:

```python
b = pd.read_csv("splits/gtzan_bind.csv"); b["genre"] = b.path.str.split("/").str[0]
b.groupby("genre").artist_name.nunique()     # blues 14 ... disco 62
```

## 2. Concepts in depth

### 2.1 Grading from the website alone

Intuition: the brief says the grader judges understanding "from how clearly the website reasons from
the question to the conclusions". A choice that is justified only in `DECISIONS.md` counts as
*made* but not *explained*. So the review asked of every rubric line: **can a reader who never
opens the repository find the evidence on the page?**

Three gaps failed that test even though the work had been done: the window-length comparison, the
chosen hyperparameters, and the training objectives. Each was fixed by copying the existing result
onto the page, not by new work.

### 2.2 Prediction-time information

The rubric asks that "prediction-time information" be clear. The question to answer is: **what does
the model see at the moment it predicts?** Here, only 30 s of audio. The file name `rock.00012`
contains the label, and the artist is known for every clip, so the page now states that neither is
ever an input. Leakage through inputs (a feature that encodes the label) is different from leakage
through the split (the same artist on both sides). The page now separates the two explicitly.

### 2.3 Why macro-F1, with a number

Macro-F1 averages per-genre F1:

```latex
\text{macro-F1} = \frac{1}{K}\sum_{k=1}^{K} \frac{2\,P_k R_k}{P_k + R_k}
```

A model that always predicts one genre has $R_k = 0$ for the other nine, so their F1 is 0 and
macro-F1 is about $\frac{1}{10} \cdot F1_{\text{that genre}}$. On C test the most-frequent dummy gets
**accuracy 0.107** but **macro-F1 0.019**. For real models on nearly balanced test genres (26 to 32
clips), the two agree closely (CNN: accuracy 0.58, macro-F1 0.55). That is the honest justification:
macro-F1 guards against giving up on a genre, and costs nothing here.

### 2.4 Multiple comparisons

With $m$ comparisons each at 95%, the chance that at least one interval misses its true value can
reach

```latex
1 - 0.95^{m}, \quad m = 6 \Rightarrow 1 - 0.95^{6} \approx 0.26
```

(if the comparisons were independent; ours share the same test clips, so the true figure is lower).
A Bonferroni correction would use $1 - 0.05/m$ intervals, which are wider. Wider intervals cannot
turn a tie into a win, so the three ties stay ties; only a clear result (CNN vs kNN, lower end
0.028) could lose its clarity. The page now says the comparisons are descriptive.

### 2.5 Objective and assumption: what each model believes

A model's **objective** is what training minimises; its **assumption** is what must be true of the
data for that objective to find a good answer. Worked from the table now on the site:

- Logistic regression minimises cross-entropy with an L2 penalty. It assumes each genre's log-odds
  is linear in the 57 features. Chosen on C: $C = 0.1$ (strong regularisation), which fits a small
  training set with correlated MFCC features.
- The RBF SVM assumes genre changes smoothly with distance in feature space. Chosen: $C = 100$,
  $\gamma = 0.001$: a wide kernel (smooth boundary) combined with a high penalty on margin
  violations (fitted closely to the training windows).
- kNN assumes nearby windows share a genre and every standardised feature counts equally. Chosen:
  $k = 3$.
- XGBoost assumes genre depends on thresholds and interactions. Chosen: depth 3, rate 0.1, early
  stopping at 54 trees: small, shallow trees, again a sign of a small dataset.
- The CNN assumes local time-frequency patterns identify genre wherever they occur (translation
  equivariance along time). Epochs chosen per run: 14, 14, 25.

## 3. Why this choice, and the alternatives we rejected

- **Copy existing results onto the page rather than run anything new.** The test part has been
  scored; any new experiment now would be chosen after seeing it.
- **Collapse long lists rather than delete them.** The brief asks for concision, but the future-work
  list and the pipeline steps are evidence a grader may want. A collapsible block keeps them one click
  away while shortening the page a skimming reader sees.
- **Rejected: regenerating the matplotlib figures with dark backgrounds.** It would re-run analysis
  scripts after the freeze for a cosmetic gain; the white figure panels are readable in both themes.
- **Rejected: a Bonferroni-corrected re-analysis.** It would be a new, post-test analysis choice; a
  sentence stating that the comparisons are uncorrected is honest and sufficient.

## 4. Code walkthrough (`docs/index.html`)

All changes are static HTML except one string:

- **Line 186**: the "At prediction time" paragraph.
- **Line 277**: the window-length result (0.435, 0.458, 0.477, 0.449) and the reason for 3 s.
- **Line 291**: the four-column models table; values from `config.yaml` (`grids`, `cnn`) and
  `results/chosen_params.json` (`C` → `"0"`).
- **Line 312**: the macro-F1 paragraph; numbers from `site.json` (`c.models`, dummy most frequent and
  CNN).
- **Line 387**: the multiple-comparisons note under the paired-difference chart.
- **Line 727** (in `renderAnswer`): the hero's interval sentence now says "A possible reason, not
  tested".

## 5. Pitfalls

- **Writing a number from memory.** Every value added was read from a file in this session (the
  grids from `config.yaml`, the chosen settings from `chosen_params.json`, artist counts from
  `splits/gtzan_bind.csv`). The artist counts per C part match the existing plateau table (rock 7,
  blues 7, jazz 15, classical 29 in train), which cross-checks the computation.
- **Citing a source by guess.** gtzan-bind is cited by its Hugging Face identifier, which is what
  `src/fetch_meta.py` actually downloads, rather than by a paper we did not check.
- **Moving the goalposts after test.** Labelling the lopsided-interval explanation as untested, and
  the comparisons as uncorrected, reduces claims; it adds none.

## 6. Results from this step

No new scores. The review estimated about 89/100 for the published version (72/80 shared, about
17/20 deliverable, the viva part uncertain). The gaps addressed are worth roughly 5 to 7 points under
that estimate; the viva (individual understanding) cannot be fixed on the page. Artists per genre in
the whole dataset: blues 14, classical 42, country 48, disco 62, hip-hop 29, jazz 23, metal 32, pop
27, reggae 26, rock 23.

## 7. Viva questions with model answers

1. **What does the model see at prediction time?** Only the 30 s of audio, cut into ten 3 s windows.
   No artist, no file name (which contains the genre), no metadata.
2. **5 s windows scored higher on validation. Why not use them?** +0.019 is a few of 197 clips, the
   settings were tuned at 3 s, and changing after seeing the validation score is selection on noise;
   it would also have meant retuning every model.
3. **Why macro-F1 if the classes are balanced?** It costs nothing when classes are balanced and
   catches a model that abandons a genre (dummy: accuracy 0.107, macro-F1 0.019).
4. **What if you corrected for multiple comparisons?** Intervals widen; the ties stay ties, and the
   CNN's lead over kNN might become uncertain. No conclusion on the page depends on a narrow win.
5. **What does C = 0.1 for logistic regression tell you?** Strong regularisation was best: with 442
   training clips and correlated features, a simpler boundary generalised better on validation.
6. **What if the CNN were given the artist as an input?** It would answer a different question and
   would fail the cold-start use case, where artists are new; on split C it could not even help,
   because test artists never appear in training.
7. **Which part of your grade can the website not secure?** Individual understanding: the site says
   an AI assistant wrote most of the code, so the viva must show I can explain and change it.

## 8. Further reading

- ES 335 project guide and rubric (course page).
- J. Demšar, "Statistical comparisons of classifiers over multiple data sets," JMLR 7, 2006.
  https://jmlr.org/papers/v7/demsar06a.html
- scikit-learn user guide, "Metrics and scoring: classification metrics."
  https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics
- gtzan-bind dataset card. https://huggingface.co/datasets/seungheondoh/gtzan-bind
