# Lesson 16: Split D (random artist-grouped) and the widened SVM grid

The leakage ladder used to jump from B′ (clip-random, fault-filtered) straight to C (the published
artist-disjoint split). That jump changed **two** things at once, so a drop between them could not
be pinned on either. This step adds split D between them, which changes only one thing. It also
widens the SVM grid, because the old SVM optimum on C sat on a grid corner.

---

## 1. What we built

| file | what changed |
|---|---|
| `src/splits.py` | New `_grouped_parts`, `_balance_by_swaps`, `make_split_D`; `split_file` knows `"D"`. |
| `scripts/make_splits.py` | Writes `splits/D_seed0.csv` … `D_seed4.csv`; older split files are rebuilt byte-identically. |
| `tests/test_splits.py` | D joins the size, fault-filter and lock tests; three new D tests. |
| `src/train_classical.py`, `src/train_cnn.py` | `--split` accepts `D`. |
| `scripts/tune_all.sh`, `scripts/summarise_val.py` | D seeds 0–4 are part of the protocol and of the summary figure. |
| `config.yaml` | `grids.svm`: C {0.1, 1, 10, 100, 1000} × gamma {scale, 0.0001, 0.001, 0.01, 0.1}. |
| `DECISIONS.md` | Entries "Split D …" and "SVM grid widened …" (2026-11-14). |

Commands (from the repo root):

```bash
.venv/Scripts/python.exe scripts/make_splits.py                       # all splits, incl. D seeds 0-4
.venv/Scripts/python.exe -m pytest -q tests/test_splits.py            # split tests
bash scripts/tune_all.sh                                              # every model on every split (resumable)
.venv/Scripts/python.exe -m src.train_classical --split D --seed 2    # one D run
.venv/Scripts/python.exe scripts/summarise_val.py                     # table + figure over all splits
```

---

## 2. Concepts in depth

### 2.1 Confounding: why one comparison must change one thing

**Intuition.** You want to know whether a fertiliser helps. You put it on field 1, not on field 2,
and field 1 grows more. But field 1 also gets more sun. You cannot say which caused the
difference: sun and fertiliser are *confounded*. The fix is to compare two fields that differ
only in the fertiliser.

Our old ladder had exactly this problem at its last step:

| step | what changes |
|---|---|
| A → B | windows of one clip can no longer be on both sides |
| B → B′ | the 70 faulty clips (duplicates, mislabels, distortions) leave the pool |
| B′ → C | (1) artists can no longer be on both sides, **and** (2) five random draws become one fixed, hand-made partition |

Change (2) matters on its own. C's valid part has 13 pop clips and 24 rock, while a stratified
random draw would give about 20 of each. One fixed partition is also a single sample: it might be
easy or hard by luck. So if a model scores 0.77 on B′ and 0.46 on C, the 0.31 drop mixes
"artist leakage removed" with "this particular partition".

**Split D** keeps everything B′ has (same 929-clip pool, same 442/197/290 sizes, genre-stratified,
five random seeds) and adds one rule: every artist's clips stay in one part. Now:

```text
B′ − D  = effect of removing artist leakage (nothing else changed)
D  − C  = effect of using the published partition instead of a random artist-disjoint one
B′ − C  = (B′ − D) + (D − C)
```

The last line is an identity: the two steps add up to the old jump exactly, so nothing is lost.

### 2.2 What counts as "one artist": credit clusters

GTZAN gives no artist names. gtzan-bind (lesson 4) names an artist for most fault-filtered clips,
written as a credit string such as "Bob Marley & The Wailers". Two credits that share a performer
("Bob Marley", "Bob Marley & The Wailers") are the same artist for leakage purposes. Lesson 15
built `credit_clusters`, which joins credits that share a name into one cluster with union-find
(connected components). D uses those clusters as groups. A clip with no artist name becomes its
own group (`"__solo_" + clip_id`). That treats it as an artist we know nothing about. It is the
optimistic assumption, since such a clip could secretly share an artist with another one.

### 2.3 Group splitting with exact sizes is a packing problem

**Intuition.** Splitting clips at random to exact sizes is easy: shuffle, cut. Splitting
*groups* to exact sizes is like packing boxes of different weights into three bins with fixed
capacities 442, 197 and 290, while also keeping every genre at roughly its share in every bin.
A group of 30 reggae clips (Bob Marley) is one box that cannot be cut.

**The quota.** For part $p$ with $n_p$ clips out of $N = 929$ and genre $g$ with $N_g$ clips in
the pool, the ideal (generally fractional) number of genre-$g$ clips in $p$ is

```latex
q_{p,g} = \frac{n_p}{N} \, N_g
```

For example, reggae has $N_g = 86$ clips in the pool (counted from `splits/D_seed0.csv`), so
train's quota is $442/929 \times 86 \approx 40.9$. Bob Marley's cluster has 30 clips in the pool
and fits only in train: in valid ($197/929 \times 86 \approx 18.2$) or test ($\approx 26.8$) it
would overshoot. That is why
Bob Marley is in train on every D seed (a stated limit in DECISIONS.md).

**Distance from the quotas.** With $c_{p,g}$ the actual counts, the balance cost is

```latex
\text{cost} = \sum_{p}\sum_{g} \left(c_{p,g} - q_{p,g}\right)^2
```

**The algorithm** (`_grouped_parts`):

1. Order groups largest first: big boxes are hardest to place, so place them while there is room.
   Ties are shuffled with the seed.
2. For each group, list the parts with enough free room. Among them, keep those where the group
   leaves every genre at most `slack = 2` clips over quota. Pick one of those **at random,
   weighted by free room**. That is what plain random assignment would do, so different seeds
   give different splits.
3. If no part keeps the group within slack, put it where it adds least to the cost.
4. If a group has nowhere to go (all parts too full), redraw from the start, up to 50 tries.
5. Finally `_balance_by_swaps`: exchange two single-clip groups of different genres between two
   parts whenever that lowers the cost. Swaps keep sizes and artist-disjointness intact, because
   single-clip groups carry no artist across a part boundary.

### 2.4 Balance versus randomness: the trade-off we measured

A first version did step 2 by cost alone: always put the group where the cost is lowest. Balance
was excellent, but the big clusters landed in the same part on every seed, so any two seeds put
only 25–30% of clips in different parts. Five seeds that are nearly the same split would make
D's seed spread look small and overstate how stable the numbers are. Random placement within
the slack fixed it:

| split | worst genre, clips off quota | two seeds differ on |
|---|---|---|
| B′ (stratified random) | 0.8 | 56–62% of clips |
| D (final) | 3.6 | 54–64% of clips |
| C (published) | 6.6 | (one partition) |

So D is about as random as B′ and better balanced than C. A test checks every seed stays within 4
clips of quota, and another checks seeds differ on more than 30% of clips.

### 2.5 Grid edges: why the SVM grid was widened

**Intuition.** A grid search is a search over a box. If the best point is on the edge of the box,
the true best may lie outside it: the search was cut off, not finished. An examiner can then say
"your SVM is under-tuned, so your CNN's lead over it is inflated".

The old grid was C ∈ {0.1, 1, 10, 100} × gamma ∈ {scale, 0.001, 0.01, 0.1} (16 settings), and on
C the winner was C = 100, gamma = 0.001: the largest C and the smallest gamma, a corner. The new
grid adds one step past that corner in each direction, C = 1000 and gamma = 0.0001 (25 settings).

**Why this is allowed now.** Tuning on the valid part is what the valid part is for. The test part
has never been loaded (no `pre-test-freeze` tag). Widening the grid after seeing *test* results
would be cheating. Widening it after seeing *valid* results, before the freeze, is ordinary tuning.

**The C × gamma ridge.** For the RBF kernel $k(x, x') = \exp(-\gamma \lVert x - x' \rVert^2)$, a
smaller gamma makes the boundary smoother, and a larger C punishes training errors harder. The
two trade off: "large C, small gamma" and "smaller C, larger gamma" can give similar boundaries.
The good settings therefore lie along a diagonal ridge in the heatmap. On some seeds of B and D
the winner is C = 1000, gamma = 0.0001. That is the corner, but it is the same ridge one step
further along: (100, 0.001) and (1000, 0.0001) both multiply C by 10 and divide gamma by 10.
On C, the main split, the winner is interior (it has untried neighbours on every side), so the
corner worry is answered where it matters. We did not widen again for those other seeds: there
is always another edge, and the ladder only needs the same protocol on every split.

---

## 3. Why this choice, and the alternatives we rejected

- **Add D, keep C as the main result.** C is the published split that prior work reports, and the
  project question names it. D is context for reading C, not a replacement.
- **`StratifiedGroupKFold` (sklearn), rejected.** It makes k folds of roughly equal size. We need
  exactly 442/197/290 so that only leakage differs between splits, never training size.
- **`GroupShuffleSplit`, rejected.** It respects groups but ignores genre, so a part could get very
  few clips of some genre.
- **Cost-only placement, rejected.** It had the best balance, but the seeds were too alike (2.4).
- **An exact solver (integer programming), rejected.** It would give perfect balance but the same
  "one best split" problem: we want random draws, not the optimum.
- **Widening only on C, rejected.** All splits must share one protocol, so the grid changed
  everywhere and every SVM was retuned.

---

## 4. Code walkthrough

### `src/splits.py`

- `_grouped_parts` (line 95):
  - line 107 asserts that D uses the whole pool (442 + 197 + 290 = 929);
  - line 108 computes each part's share $n_p/N$;
  - line 110 groups clip indices by credit cluster.
- line 112: the retry loop. Each try draws a fresh order from the same `rng`, so the result is
  still fixed by the seed.
- lines 113–115: shuffle, then a *stable* sort by size. Large groups come first, and equal sizes
  keep their shuffled order. This is where the seed changes which group of size 5 goes first.
- line 130: `quota` is $q_{p,g}$ for the group's genres in each part with room.
- line 131: `room` keeps parts where every genre of the group stays within `quota + slack`.
- lines 132–134: a random choice weighted by `free[p]`, the room left in each part.
- lines 135–138: the fallback, the part with the smallest increase in squared distance. The
  expression is (cost after) − (cost before) for the affected genres only.
- line 144: once every group is placed, the single-clip swaps.
- `_balance_by_swaps` (line 148): `counts` is the genre × part table. The double loop tries every
  pair of single clips in different parts and genres. `cells` lists the only four table cells a
  swap touches, so `before`/`after` compare just those. The loop repeats until a full pass makes
  no swap; each swap strictly lowers a non-negative cost, so it must stop.
- `make_split_D` (line 175): line 185 turns missing artists into solo groups; then `_grouped_parts`
  runs on the C pool (`in_C`). `_expand` turns clips into ten window rows each.

### `tests/test_splits.py`

- `test_D_is_artist_disjoint_by_credit_cluster` (line 117): no credit cluster in two parts, for
  every seed.
- `test_D_genres_are_close_to_proportional` (line 125): every genre within 4 clips of its quota in
  every part. The comment there gives the measured worst case (3.6) and B′/C for comparison.
- `test_D_seeds_differ` (line 137): any two seeds put more than 30% of clips in different parts.

---

## 5. Pitfalls

- **Splitting by credit string instead of credit cluster.** "Bob Marley" and "Bob Marley & The
  Wailers" would be different groups, so the same singer could sit in train and test: leakage that
  a test on raw strings would not catch.
- **Choosing D's sizes freely.** If D had more training clips than B′, part of the B′ − D drop
  would be a training-size effect. Exact sizes rule that out.
- **Reading one seed.** D's SVM ranges from 0.469 to 0.676 over seeds (section 6). One
  artist-disjoint split is a single, noisy draw, which matters for how much weight C alone can bear.
- **Widening a grid after looking at test.** That turns the test part into a second validation set
  and makes its score optimistic. The order matters: widen, retune, *then* freeze.
- **Treating unknown-artist clips as one group.** Lumping every unnamed clip into one giant
  "unknown" group would force them all into one part and ruin the balance. A solo group per clip is
  the standard choice, and its optimism is stated in 2.2.

---

## 6. Results from this step

Split facts (from `results/split_summary.json` and the split files):
- every D seed has 442/197/290 clips;
- 0 test clips share a credit cluster with train (C: 25, B′ seed 0: 207);
- the worst genre is 3.6 clips off quota;
- two seeds differ on 54–64% of clips.

Valid clip macro-F1, mean over runs (5 seeds on B, B′ and D; one run on C). Read from
`results/val/*.json`; the full table is in `results/val/summary.csv` after `summarise_val.py`:

| model | B | B′ | D | C |
|---|---|---|---|---|
| kNN | 0.698 | 0.708 | 0.526 | 0.424 |
| logistic regression | 0.727 | 0.734 | 0.589 | 0.468 |
| RBF SVM | 0.751 | 0.769 | 0.596 | 0.458 |
| XGBoost | 0.728 | 0.741 | 0.560 | 0.419 |
| dummy (stratified) | 0.125 | 0.136 | 0.112 | 0.128 |

Reading it:
- **B′ → D**: every classical model loses 0.15–0.18 when only artist leakage is removed (SVM:
  0.769 → 0.596).
- **D → C**: a further 0.10–0.14 (SVM: 0.596 → 0.458). The published partition is harder than a
  typical random artist-disjoint one. These numbers cannot say why. Two candidate reasons: C's
  odd genre mix in valid (13 pop, 24 rock), and possible same-producer or same-album links that
  artist credits do not capture.
- **Seed spread on D is large**: SVM sd 0.079 (0.642, 0.469, 0.612, 0.581, 0.676); logistic
  regression sd 0.074. On B′ the SVM sd is 0.020. Grouping by artist makes the valid part depend
  on *which artists* land in it, so the split choice alone moves scores by several points.
- The CNN on D and B′ was still training when this lesson was written; lesson 17 and the site add
  those numbers.

SVM grid (25 settings):
- on C the winner is still C = 100, gamma = 0.001 (0.458), now interior;
- on D: seed 1 picks C = 0.1 with gamma = scale (an edge), seeds 3 and 4 pick C = 1000 with
  gamma = 0.0001 (the ridge corner);
- on B′: C = 10 on four of five seeds.

Heatmap: `results/figures/val_svm_heatmap.png`.

---

## 7. Viva questions with model answers

1. **Why add split D at all? Wasn't C enough?**
   C answers the main question, but the drop from B′ to C mixed two causes: removing artist
   leakage and using one fixed hand-made partition. D removes only the leakage, with the same
   pool, sizes and five random seeds as B′. B′ − D isolates artist leakage, and D − C isolates
   the partition.

2. **D scores higher than C. Doesn't that mean C has leakage you missed?**
   No, leakage would push C *up*, not down. C is harder than a typical artist-disjoint draw. With
   one partition we cannot say why: its genre mix, or links between clips that credits don't show.
   The point is that a single fixed split is one draw from a wide distribution. D's SVM ranges
   0.469–0.676 over five seeds.

3. **Why is Bob Marley always in train? Is that a problem?**
   His 30 reggae clips exceed valid's and test's reggae quotas (about 18 and 27), so only train can
   hold them. It is a limit: D seeds never test on Bob Marley. It is stated in DECISIONS.md. It
   does not create leakage, because the cluster is in one part.

4. **What if you placed groups to minimise imbalance only?**
   We tried it. Balance was best, but large clusters landed in the same part on every seed, so the
   seeds overlapped heavily (only 25–30% of clips differed). The seed spread would understate
   split-to-split variation. Random placement within a 2-clip slack keeps the seeds as different as
   B′'s.

5. **Isn't widening the SVM grid after seeing results a form of p-hacking?**
   It would be if we looked at test. We looked only at the valid part, which exists for tuning, and
   the test part is locked by the missing `pre-test-freeze` tag. Leaving a corner optimum
   unexplored would instead make the SVM look weaker than it is and flatter the CNN.

6. **Some D seeds still pick a corner (C = 1000, gamma = 0.0001). Why not widen again?**
   That corner lies on the same C × gamma ridge as the interior optimum on C: both multiply C by 10
   and divide gamma by 10. On C, the main result, the optimum is interior. We stop because there is
   always another edge; what matters is that all splits share one grid.

7. **What if D had used `StratifiedGroupKFold`?**
   Its folds are roughly equal, so we could not hit 442/197/290 exactly. Training size would then
   differ between B′ and D, and part of the drop would be a learning-curve effect, not leakage.

8. **What if an unnamed clip actually shares an artist with a test clip?**
   Then some leakage remains and D's scores are slightly optimistic. It is the same assumption C
   and the bootstrap make. It cannot make the B′ → D drop look larger than it really is, only
   smaller.

---

## 8. Further reading

- Kereliuk, Sturm & Larsen (2015), *Deep learning and music adversaries* (the fault-filtered,
  artist-disjoint split): <https://arxiv.org/abs/1507.04761>
- Sturm (2013), *The GTZAN dataset: its contents, faults, effects on evaluation, and future use*:
  <https://arxiv.org/abs/1306.1461>
- Flexer (2007), *A closer look on artist filters for musical genre classification* (ISMIR):
  <https://ismir2007.ismir.net/proceedings/ISMIR2007_p341_flexer.pdf>
- scikit-learn, group-aware splitters (`GroupShuffleSplit`, `StratifiedGroupKFold`):
  <https://scikit-learn.org/stable/modules/cross_validation.html#group-k-fold>
- Hsu, Chang & Lin, *A Practical Guide to Support Vector Classification* (C × gamma grids):
  <https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf>
