# Lesson 22: Shortening the website without losing evidence

The rubric's brief asks for a site that is "clear and concise" yet lets a grader judge the whole
project from it alone. Those pull in opposite directions. This lesson records how the page was cut by
about a fifth of its visible text while every fact and number stayed on it, and how that was checked.

No model, setting, prediction or score changed.

---

## 1. What we built

| file | change |
|---|---|
| `docs/index.html` | duplicated text removed or merged; secondary material moved into click-to-open blocks; sentences shortened; unused CSS for the removed "page organised" strip and step list deleted |
| `learn/19_presenting_results_charts.md`, `learn/20_rubric_review_and_fixes.md` | line references re-synced |
| `DECISIONS.md` | dated entry |

What moved where, section by section:

| section | change |
|---|---|
| Question | "How the page is organised" strip removed (the sticky navigation bar already lists the sections); the motivation paragraphs shortened |
| Answer | the answer paragraph no longer repeats the best-classical paired difference (the forest plot shows it); the leakage sentence keeps only the artist-leakage cost with its interval |
| Data | the "why these faults matter" box became one sentence above the list; the artist-matching bullet merged into "many clips per artist"; the 57-feature breakdown moved into the variables table |
| Method | the evaluation paragraph no longer lists the five splits that the cards below list again; the collapsed "six steps" block was removed after its unique facts (no k-fold over windows, how the test lock works, soft vote, artist bootstrap) became five short bullets; the SVM grid and learning-curve figures are one click away |
| Ladder, Models | the reading guides keep only what the chart legend does not say; the model table is collapsed; the refit paragraph no longer repeats the numbers the dumbbell chart shows |
| Errors | the four findings kept, wording tightened; the "resample artists" note joined the first finding |
| Why ~0.55 | "Other limits we can see" moved into Limitations (it overlapped with three items there); "other results agree" is one sentence |
| Limitations | the two lists merged: ten items, each one or two sentences |
| Key terms, References | collapsible, with a summary naming every term or source |

## 2. Concepts in depth

### 2.1 Progressive disclosure

Intuition: a reader should see the argument first and the supporting detail on demand. HTML's
`<details>`/`<summary>` element does this with no JavaScript: the summary line is always visible
and says what is inside, and one click opens it. A grader skimming sees the claims; a grader
checking opens the evidence.

The rule used here: **claims, results and limitations stay visible; reference material and second
figures for the same point go behind a summary that names them.**

### 2.2 Measuring "shorter" honestly

"Shorter" was measured, not guessed. The page was rendered in headless Edge (so the sentences that
JavaScript writes are included), and words were counted outside tables, charts and code:

```latex
\text{reduction} = 1 - \frac{\text{visible words after}}{\text{visible words before}}
= 1 - \frac{\approx 3{,}880}{4{,}921} \approx 21\%
```

Total prose including collapsed blocks went from 5,372 to 4,620 words. That 14% is the repetition
that was actually deleted. The rest of the visible reduction is material moved behind a summary.

### 2.3 Checking that nothing was lost

Every number in the rendered old page was compared with the rendered new page. Exactly one number
disappeared: "0.04", the lower end of the D → C step's interval in the old answer paragraph. It is
still on the page in the waterfall tooltip and the step table, and the answer now says "within
noise". One number was **corrected** rather than lost: the per-genre note said genres have "13 to 32"
test clips, but the test counts are 26 to 32 (blues 31, classical 31, country 30, disco 29,
hip-hop 27, jazz 27, metal 27, pop 30, reggae 26, rock 32).

## 3. Why this choice, and the alternatives we rejected

- **Rejected: deleting the future-work list, the extra figures or the glossary.** Each is evidence a
  grader may want; collapsing keeps them.
- **Rejected: cutting Limitations to five items.** Limitations are graded directly ("discussed
  without overclaiming"); merging duplicates was enough.
- **Rejected: a separate "details" page.** The brief says the grader should judge from the website;
  one page with click-to-open blocks keeps everything on it.
- **Kept visible on purpose:** the question, prediction-time inputs, the models table, the test
  lock, the leakage lab, every chart, the error findings, the pretrained check with its caveats,
  Limitations and "Who did what".

## 4. Code walkthrough (`docs/index.html`)

- **`<details>` blocks**: the SVM grid and learning curves (Method), the step and chart tables
  (Ladder), the model table (Models), future work (Why ~0.55), Key terms and References. Each
  `<summary>` names its contents.
- **`renderAnswer`**: the sentence that repeated the best classical model's paired difference is gone;
  `ladderSentence` now returns one sentence (artist-leakage cost with its interval, and whether
  D → C is within noise).
- **`renderRefit`**: no longer builds the before → after numbers (the dumbbell chart shows them); it
  keeps the CNN-vs-SVM paired differences before and after refitting, the untested reason, and why the
  refit models stay the headline.
- **CSS**: the `.flow` and `.steps` rules were deleted with the elements that used them. Three of the
  rules spanned two lines; the continuation lines were found and removed too, so no stray CSS remains.

## 5. Pitfalls

- **Cutting a number that only appeared once.** The number-by-number comparison guards against this.
- **Collapsing a claim.** A claim hidden behind a click is easy to miss; only supporting material
  was collapsed.
- **Line-based deletion of multi-line CSS rules** leaves fragments that can break the styles after
  them. Checked and fixed.
- **Stale cross-references.** "See Limitations" and "learning curves in Method" were kept pointing at
  sections that still hold that content.

## 6. Results from this step

Visible prose: 4,921 → about 3,880 words (about 21% shorter). All prose including collapsed blocks:
5,372 → 4,620. All nine charts render; the HTML tags balance; one number moved to tooltip/table
only; one note corrected (26 to 32 test clips per genre).

## 7. Viva questions with model answers

1. **Why hide the glossary instead of deleting it?** It helps readers who need it, and costs others
   nothing once collapsed; the summary lists every term, so a grader knows it is there.
2. **How do you know nothing important was lost?** Every number on the old page was checked against
   the new one; the single missing value is still in the waterfall tooltip and step table.
3. **What stays visible no matter what, and why?** The claims and their limits: question, inputs,
   models, evaluation protocol, results, errors, limitations, contribution. These are what the rubric
   grades.
4. **What if the grader never clicks a summary?** They still see every claim, chart and limitation.
   What they miss is supporting detail (a second figure, a table view, definitions, citations).
5. **Why was "13 to 32" changed?** It was wrong for this page: those are not the test counts. The
   test counts are 26 to 32.

## 8. Further reading

- MDN, `<details>`: The Details disclosure element.
  https://developer.mozilla.org/en-US/docs/Web/HTML/Element/details
- Nielsen Norman Group, "Progressive disclosure." https://www.nngroup.com/articles/progressive-disclosure/
- GOV.UK, "Writing for GOV.UK" (plain-language guidance). https://www.gov.uk/guidance/content-design/writing-for-gov-uk
