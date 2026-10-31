# Lesson 15: Auditing by evidence, and what reading alone missed

Lesson 14 audited the code by reading it. This audit started there too, then checked the
*outputs*: it rebuilt artefacts, compared them byte for byte, and fed the system inputs chosen
to embarrass it. Nothing in training, features, splits or stored scores was wrong. Three real
bugs and one measurement problem turned up, and the measurement problem would have affected
the headline confidence interval had it survived to the freeze.

---

## 1. What we built

| file | what changed |
|---|---|
| `src/splits.py` | New `credit_clusters`: artist credits that share a performer are one cluster. |
| `src/evaluate.py` | Bootstrap reads `cluster` by default; `n`, `ci` and `seed` come from config. |
| `src/utils.py` | New `write_json`: atomic result files. |
| `src/train_classical.py`, `src/train_cnn.py` | Result files that mark a run finished use `write_json`. |
| `src/webapp.py` | Audio under `app.min_rms_db` is refused as silent. |
| `scripts/tune_all.sh` | Strips CR from the seed list. |
| `scripts/make_splits.py` | Summary gains `test_clips_credit_cluster_in_train`. |
| `scripts/check_lesson_refs.py` | New: checks, and with `--fix` repairs, named line references in lessons. |
| `scripts/analysis_val.py`, `src/data.py`, `src/features.py` | Hard-coded constants now read from config or from the data. |
| `config.yaml`, `requirements.txt` | `app.min_rms_db`, `audit.clip_threshold`; unused `torchaudio` removed. |
| `learn/*.md` | 29 line references repaired. |

Commands (from the repo root):

```bash
.venv/Scripts/python.exe -m pytest -q                       # all tests
.venv/Scripts/python.exe scripts/check_lesson_refs.py       # exit 1 if a reference is wrong
.venv/Scripts/python.exe scripts/check_lesson_refs.py --fix # repair them
.venv/Scripts/python.exe scripts/make_splits.py             # rebuilds splits and the summary
```

---

## 2. Concepts in depth

### 2.1 Reading finds suspects, evidence finds bugs

**Intuition.** Reading code tells you what might be wrong. Only running it tells you what is.
This audit's first suspect, the CR in `tune_all.sh`, *failed* its first test: under Git Bash
all three seeds matched. Had we stopped there, a real bug would have stayed. Had we fixed it
on suspicion alone, we would not have known when it matters. It took a second test, in the
right environment, to show both.

**The checks that cleared the pipeline.** Each can fail, which is why a pass means something:

| question | check | result |
|---|---|---|
| Do the splits come from the code? | Rebuild all 12 and compare bytes | All identical |
| Do cached features match the code? | Recompute 3 clips | Handcrafted within 0.0005, mels within 0.016 dB (1 step of float16) |
| Do stored scores reproduce? | Re-tune dummy, kNN, logreg and XGBoost on C and B' seed 2 | Same to 6 decimals, same params and tree counts |
| Are results self-consistent? | For each file: confusion trace = accuracy, per-genre mean = macro-F1 | All pass |
| Does a fresh clone work? | Run pytest in a clean clone | 101 passed, 14 skipped |

### 2.2 Why a stub test missed the CR bug

**Intuition.** Lesson 14 tested `tune_all.sh` with a stub Python, and the stub printed plain
`0 1 2`. Real Python on Windows prints `0 1 2\r\n`. The test could not fail for this bug,
which is exactly pitfall 2 of lesson 14, repeated by the same audit that wrote it down. A stub
proves only what it imitates.

**The mechanism.** The shell splits words on spaces, tabs and newlines, not on CR, and `$( )`
strips only trailing newlines:

```latex
\texttt{0 1 2\textbackslash r\textbackslash n}
\;\xrightarrow{\;\$(\;)\;}\;
\texttt{0 1 2\textbackslash r}
\;\xrightarrow{\text{split}}\;
\{\texttt{0},\ \texttt{1},\ \texttt{2\textbackslash r}\}
```

So the script looks for `C_s2\r_cnn.json`, which never exists. Python then reads `--seed 2\r`
without complaint, because `int("2\r")` is 2, and trains. Nothing fails. Seed 2 is simply
retrained on every resume, and it overwrites the model the web app serves.

**Worked example.** With every run finished, so nothing should train, under WSL:

| | commands run |
|---|---|
| before | `train_cnn --split C --seed 2^M`, `--summary`, `summarise_val` |
| after | `--summary`, `summarise_val` |

Git Bash removes the CR itself, which is why this hid. WSL does not.

### 2.3 What a cluster bootstrap assumes

**Intuition.** Clips by one performer are correlated, so they must be resampled together
(lesson 6). That only works if "one performer" is identified correctly. Split one performer
into two labels and the bootstrap treats correlated clips as independent. The interval comes
out too narrow, which is the very mistake it exists to prevent.

**Math.** For a mean over $n$ clips with common variance $\sigma^2$ and correlation $\rho$
inside clusters of size $m$, the variance is inflated by the design effect:

```latex
\mathrm{Var}(\bar{x}) \approx \frac{\sigma^2}{n}\,\bigl(1 + (m - 1)\rho\bigr)
```

Larger true clusters mean more inflation. Under-merging understates $m$, so it understates the
variance. Over-merging errs the safe way: a slightly wider interval.

**Worked example.** gtzan-bind has "Bob Marley & The Wailers" (32 clips) and "Bob Marley" (3).
Exact names give 321 artists; shared credits give 294. In split C:

| measure | exact names | credit clusters |
|---|---|---|
| Artists crossing parts | 13 | 15 |
| Clips in crossing artists | 74 | 122 |
| Test clips linked to train | 22 (7.6%) | 25 (8.6%) |
| Test clusters for the bootstrap | 116 | 113 |

The three new links are real. Two classical test clips credited to Daniel Barenboim link to
train's "Chicago Symphony Orchestra & Daniel Barenboim" (3 clips). One pop test clip,
"Babyface, Mariah Carey, Kenny G & Sheila E.", links to train's Mariah Carey (4 clips).

The change is small, and that is worth saying plainly: split C stays almost artist-disjoint,
and the published split is unchanged. The fix makes the interval honest, not different.

### 2.4 Connected components

**Intuition.** Linking is transitive. If credit A shares a name with B, and B with C, all three
belong together even though A and C share nothing. The groups are the connected components of
a graph whose edges are shared names.

**Math.** Define a relation on credits, then take its transitive closure:

```latex
a \sim b \iff \mathrm{names}(a) \cap \mathrm{names}(b) \neq \emptyset,
\qquad
\text{clusters} = \text{equivalence classes of the transitive closure of } \sim
```

**Worked example.** "Barrington Levy & Bounty Killer" and "Bounty Killer & Fugees" join through
Bounty Killer. "Big Youth & Dennis Brown", "Dennis Brown" and "Dennis Brown & Bongo Herman"
join through Dennis Brown. Union-find computes this in one pass: keep a parent for each name,
and joining two names points one root at the other.

### 2.5 Silence, decibels and an upper limit from data

**Intuition.** "Silent" cannot mean "all samples are zero", because real silent audio contains
dither. A guard for exact zeros let through audio no one can hear, and the model, which has
never seen anything like it, gave a confident answer. Softmax has no "I do not know" output.

**Math.** RMS level, relative to full scale:

```latex
L = 20 \log_{10} \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2}
\qquad
\text{refuse if } L < -60\ \text{dBFS}
\iff
\mathrm{RMS} < 10^{-60/20} = 0.001
```

**Worked example.** Random samples in $\{-1, 0, 1\}$ at 16 bits have RMS
$\sqrt{2/3}/32768$, about -92 dBFS. The old app called that "classical" at 99.7% with no "Not
sure". The quietest clip in GTZAN (`classical.00080`) is -42.4 dBFS, so the floor sits 17.6 dB
under anything the model trained on, and a test keeps it there.

### 2.6 Atomic writes

**Intuition.** `tune_all.sh` treats "the file exists" as "the run finished". A file that is
half written also exists. Writing to a temporary name and renaming makes "exists" imply
"complete", because a rename is one step. A truncated result crashed the dashboard with a
`JSONDecodeError` when tested.

---

## 3. Why this choice, and the alternatives we rejected

| choice | alternative rejected | reason |
|---|---|---|
| Shared-credit rule | Hand-written alias table | The table found lead names only and missed "Los Angeles Philharmonic & Leonard Bernstein". A rule also needs no hand-kept list. |
| `cluster` as the default column | Keep `artist` | A forgotten column now raises KeyError rather than quietly using exact names. |
| Keep `artist_groups` on exact names | Change it everywhere | That redraws every learning-curve subset, needing about 12 CNN fits, for analyses that report no final score. |
| Keep published split C | Rebuild C from clusters | It would no longer be the published split. |
| Floor of -60 dBFS, from data | Exact zero test | Dither passes it. |
| Reference check as a tool | As a pytest test | Moving code is not a bug; every edit above a reference would fail the suite. |
| Manual WSL check of the CR fix | A pytest test | Git Bash strips CR, so that test could not fail on this machine. |

---

## 4. Code walkthrough

**`src/splits.py`, L143-163 `credit_clusters`**

- `find` follows parent pointers to a root; `setdefault` makes a new name its own root.
- `credits = ...` splits each credit into names. Missing values are dropped here.
- The double loop points each later name's root at the first name's root.
- The last line maps every credit to its root and `reindex` restores the missing rows as NaN,
  which `_cluster_rows` then treats as a cluster of one.

**`src/evaluate.py`, L124-136 `cluster_bootstrap_ci`** and **L139-152 `paired_bootstrap_diff`**:
the signatures take `cluster_col="cluster"`, and `n`, `ci` and `seed` from config.

**`src/utils.py`, L42-46 `write_json`**: writes `<path>.tmp`, then `Path.replace` swaps it in.
The name ends in `.tmp`, so the `*.json` glob in `load_results` cannot pick it up.

**`src/webapp.py`, `load_audio`**: line 116 computes RMS in float64 and compares it with the
config floor, replacing `not y.any()`.

**`scripts/tune_all.sh`, line 25**: `tr -d '\r'` cleans the seed list.

**`scripts/make_splits.py`, `summarise`**: line 32 computes clusters over all clips, so a link
through a clip outside the test part still counts; line 44 adds the new measure.

**`scripts/check_lesson_refs.py`**: L19-27 `definitions` parses every tracked Python file with
`ast` and records where each function starts and ends. L30-49 `fix_line` compares the lesson's
numbers with those. L52-67 `main` tracks the file last named in a lesson, to settle names
defined twice, such as `main`.

---

## 5. Pitfalls

1. **A stub that does not imitate the failure.** See 2.2.
2. **Stopping at the first negative result.** The CR suspect failed in Git Bash and was still
   real under WSL. A negative result covers only the environment it ran in.
3. **`isin` and NaN.** Our first test said 28 test clips shared an artist. The training set
   contained NaN, and pandas matches NaN to NaN, so 6 clips with no artist counted as shared.
   The real count is 22. The script already used `dropna`; the test did not.
4. **Testing the bootstrap with the wrong unit.** Exact-name clusters would have run without
   any error, giving a slightly narrow interval with nothing to show for it.
5. **Trusting a confident answer on input unlike the training data.** A 99.7% score says
   nothing about whether the input resembles music.
6. **Line references drift.** Lesson 14 warned of this and shipped no check. Of 121 named
   references, 29 were wrong: 15 from this audit's edits, 4 from an edit to `app.py` during the
   audit, 8 already stale in lessons 4 and 5, and 2 counted from a decorator rather than `def`.
7. **Saying "nearly identical".** Re-running changed three JSONs by one line each. It was
   `grid_valid_macro_f1`, a field added after those files were first written, and every score
   matched. The difference had to be read, not assumed.

---

## 6. Results from this step

| check | result |
|---|---|
| Full test suite | 122 passed |
| Split files rebuilt from code | 12 of 12 byte-identical |
| Stored scores re-tuned, split C seed 0 | dummy 0.127695, kNN 0.424072, logreg 0.468146, XGBoost 0.419498: all equal the stored values |
| `analysis_calibration.json`, `analysis_voting.json` after the edits | Byte-identical |
| CNN temperatures after the edits | 1.37, 2.01, 1.75: unchanged |
| Lesson references | 121 checked, 29 repaired, 0 wrong |
| Credit clusters | 294, against 321 exact names |
| Test clips linked to train in C | 25, against 22 by exact name |

Credit-cluster counts for every split are in `results/split_summary.json`. Fixed: seed 2
retraining on every resume under WSL; confident answers on near-silent audio; half-written
results; split performers in the bootstrap; hard-coded constants; an unused dependency;
stale lesson references.

Not changed: every model, every feature, every split, every stored score.

---

## 7. Viva questions with model answers

**Q1. Does the credit-cluster change alter any reported result?**
No validation score uses it. It changes the unit the test-time bootstrap resamples and adds a
leakage count. It was fixed before the freeze, so no test score influenced it.

**Q2. What if the rule merges two unrelated artists?**
Over-merging widens the interval slightly, which errs on the safe side. All 17 merged groups
were read by hand and none was wrong. Band names such as "Earth, Wind & Fire" are split, but
into parts no other credit uses.

**Q3. Why not make split C fully disjoint by these clusters?**
Then it stops being the published split, and our results stop being comparable with prior work.
We report the 25 linked clips (8.6%) instead. A strict subset could be a sensitivity check.

**Q4. Why is validation not affected, given 7 valid clips share an artist with train?**
It is affected, and always was: valid and test scores carry a small residual artist leak. The
step from B' to C still removes nearly all of it. The table in 2.3 states the size.

**Q5. What if a real, very quiet recording falls under -60 dBFS?**
It is refused with a clear message. No GTZAN clip is within 17 dB of the floor, and a whole
track averaging -60 dBFS is close to inaudible at normal listening levels.

**Q6. Why did the model say "classical" for near silence?**
A softmax must spread probability 1 across the ten genres, so it cannot abstain. The likely reason
is that low mel energy sits closest to classical, the quietest genre (median -28.0 dBFS). We did
not test that explanation, and either way it is an extrapolation, not a judgement.

**Q7. Why is a rename atomic but a write not?**
A write reaches the file in pieces, so a crash can leave part of it. A rename changes one
directory entry, so a reader sees the old file or the new one. On Windows a rename can fail if
another program holds the target open, but result files are new at that moment.

**Q8. What if you run `tune_all.sh` from WSL with a Linux venv?**
Python prints LF, `tr` removes nothing, and everything behaves as before. The fix is harmless
where it is not needed.

**Q9. How do you know the edits changed no behaviour?**
The tests are not enough for that claim. We re-ran the changed paths and compared the outputs:
calibration and voting files were byte-identical, and the classical files differed only by a
field added since they were written.

---

## 8. Further reading

- B. L. Sturm (2013). *The GTZAN dataset: Its contents, its faults, their effects on
  evaluation, and its future use.* https://arxiv.org/abs/1306.1461
- A. C. Cameron and D. L. Miller (2015). *A practitioner's guide to cluster-robust inference.*
  Journal of Human Resources. https://jhr.uwpress.org/content/50/2/317
- R. Sedgewick and K. Wayne, *Algorithms*, section 1.5, Union-Find:
  https://algs4.cs.princeton.edu/15uf/
- Python, `pathlib.Path.replace`: https://docs.python.org/3/library/pathlib.html#pathlib.Path.replace
- GNU Bash manual, command substitution and word splitting:
  https://www.gnu.org/software/bash/manual/bash.html#Command-Substitution
- A. Nguyen, J. Yosinski and J. Clune (2015). *Deep neural networks are easily fooled.* CVPR.
  https://arxiv.org/abs/1412.1897
