# Lesson 14: A whole-code audit, and the bug I shipped

A full read of the code, with each suspected bug reproduced before it was changed.
The main find was a loudness bug in the web app's AAC path. Nothing in training,
features or stored scores changed, so every number in lessons 7 to 12 still stands.

---

## 1. What we built

| file | what changed |
|---|---|
| `src/webapp.py` | `decode_with_av` rewritten: channels averaged, librosa's resampler, damaged packets skipped, streams that change midway handled. |
| `src/splits.py` | `is_test_unlocked` stays locked without git. `load_split` rejects an unknown part. |
| `app.py` | Works from any folder, survives unfinished results, picks up new results within a minute, explains missing files. |
| `scripts/tune_all.sh` | Resumes: finished runs are skipped. |
| `Makefile` | `analysis` and `app` work; `cnn` points at `tune`. |
| `.gitattributes`, `.gitignore` | Scripts keep LF. All uploadable audio formats and `*.pt` are ignored. |
| `tests/test_webapp.py`, `tests/test_splits.py` | 14 new tests; the lock test now survives the tag. |
| `learn/*.md` | Line references re-anchored to the code; stale claims corrected. |

Commands (from the repo root):

```bash
.venv/Scripts/python.exe -m pytest -q                 # all tests
bash scripts/tune_all.sh                              # finishes the 7 missing CNN runs, then the summaries
.venv/Scripts/python.exe -m streamlit run app.py      # the app
```

---

## 2. Concepts in depth

### 2.1 Downmixing, and why 3 dB matters here

**Intuition.** A stereo file has two channels and the model wants one. Two sensible
recipes exist, and they differ in loudness. Averaging keeps the level of a sound that
is the same in both channels. Broadcast downmixing adds them with a gain of about
0.707, which is meant for speakers, where two channels of the same sound add up in
the room.

**Math.** For identical channels $L = R = x$:

```latex
\text{average: } \frac{L + R}{2} = x
\qquad
\text{FFmpeg float downmix: } \frac{L + R}{\sqrt{2}} = \sqrt{2}\,x
\qquad
20 \log_{10} \sqrt{2} \approx 3.01\ \text{dB}
```

**Why a level error is a model error.** `clip_mel` uses a fixed reference
(`mel.ref: 1.0`), so a louder file produces larger log-mel values, and the CNN was
trained on librosa's average. A gain error is therefore a training-serving skew
(lesson 13), not a cosmetic difference. Lesson 13's gain experiment showed that 12 dB
of gain changes 10 of 30 answers.

**Worked example.** Your AAC song, before and after the fix:

| | before | after |
|---|---|---|
| peak | 1.415 | 1.001 |
| level | -6.8 dB | -9.8 dB |
| scores | country 27%, jazz 22%, classical 15% | country 30%, jazz 19%, classical 17% |
| windows for country | 3 of 10 | 4 of 10 |

The peak of 1.415 was the clue: a mastered track peaks near 1.0, and 1.415 is almost
exactly $\sqrt{2}$. Measured on a synthetic tone, the error was +2.98 dB for stereo,
+10.64 dB for 5.1 and -0.01 dB for mono.

### 2.2 Why the earlier check missed it

The AAC check in lesson 13 encoded valid clips as **mono**, because the GTZAN clips are
mono. Mono has nothing to downmix, so it could not show the bug. Real music files are
stereo. A test proves only what its input can exercise: the check passed honestly and
still said nothing about the common case.

### 2.3 Reproduce first, then fix

Every suspicion became a failing test before any code changed: 13 tests failed. Two
of them turned out to be wrong tests, and that is the value of the habit:

- 5.1 came out at 5/6 of the expected level. Per-channel levels showed the encoder
  had filtered a 440 Hz tone out of the subwoofer channel, so the code was right.
- A guard for non-finite samples could never run: both decode paths go through
  librosa, which already refuses them. The guard was deleted.

### 2.4 Failing safe

The test lock must fail **closed**. With no git installed, `is_test_unlocked` used to
crash. That did not open the lock, but it took the dashboard down with it. It now
returns False. A security check should give the safe answer when it cannot find out.

### 2.5 Resumable pipelines

A pipeline that takes hours should not lose finished work when it stops.
`tune_all.sh` now skips a run whose result file exists. This is safe only because each
script writes its JSON last, after training, so an interrupted run leaves no file.

---

## 3. Why this choice, and the alternatives we rejected

| choice | alternative rejected | reason |
|---|---|---|
| Average channels after decoding | Ask FFmpeg for mono | Its float downmix is louder, by 3 dB for stereo. |
| `librosa.resample` | FFmpeg's resampler | One resampler for every format, so one less skew. |
| Skip damaged packets | Reject the file | Players skip them, and a part-downloaded song is still classifiable. |
| Refuse non-finite audio through librosa | Our own check | It could never run: dead code. |
| Skip finished runs | Rerun everything | The missing work is 7 CNN runs; a full rerun would repeat hours. |
| Leave the SVM `probability=True` warning | Move to `CalibratedClassifierCV` | It would change stored SVM probabilities. It is safe on our pin (sklearn 1.9.1). |

---

## 4. Code walkthrough

**`src/webapp.py`**

- **L63-93 `decode_with_av`**: L75 reads packets one at a time, so L78 can skip a bad
  one. L81-84 begin a new run when the rate or channels change; the resampler at L83
  keeps both and converts the sample format only. L86 averages channels, as
  `librosa.to_mono` does. L90 keeps what was decoded if the file ends in damage. L92
  resamples each run with `soxr_hq`, librosa's default.

**`src/splits.py`**

- **L104-110 `is_test_unlocked`**: L108 turns "git is not installed" into False.
- **L117-124 `load_split`**: the lock is checked first, then L121-122 reject an
  unknown part. A typo used to return an empty table and fail later, far from the cause.

**`app.py`**

- **L15**: moves to the repo root before importing `src`, because `config.yaml` and
  every path in it are relative.
- **L42**: `ttl=60`, so finished runs appear without a restart.
- **L136-147 `overview`**: each metric appears only if its numbers exist.
- **L240-250, the main block**: one `except FileNotFoundError` covers the model,
  mels and result files.

**`scripts/tune_all.sh`**

- `finished` tests for a result file. The classical line checks the `xgb` file
  because it is written last.

---

## 5. Pitfalls

1. **Testing only the convenient input.** Our data is mono, real uploads are stereo.
2. **Reading a clean result as proof.** A check that cannot fail for a given bug says
   nothing about that bug.
3. **Skipping a run after changing its settings.** `tune_all.sh` trusts existing files.
   After changing a grid or the model, delete the old files in `results/val`.
4. **Reloading instead of restarting the app** (lesson 13). New `app.py` with old
   `src/webapp.py` caused both of the errors reported this session.
5. **Fixing what would move stored scores.** The SVM warning is real, but changing
   that code now would invalidate tuned results, so it stays until an upgrade.
6. **Trusting line numbers in prose.** They drift. Check them by script.

---

## 6. Results from this step

| check | result |
|---|---|
| Full test suite | 115 passed |
| Failing tests written before the fixes | 13 |
| Stereo AAC level error | +2.98 dB before, fixed |
| 5.1 AAC level error | +10.64 dB before, fixed |
| 30 valid clips as stereo AAC, against the original | Same top genre on 30 of 30; largest score change median 0.009, worst 0.030 |
| The same as stereo M4A | Same top genre on 30 of 30; median 0.009, worst 0.030 |
| The same as mono AAC | Same top genre on 29 of 30; median 0.011, worst 0.063 |
| Resume logic with a stub Python | Ran exactly the 7 missing CNN runs, then the two summary steps |
| Stored validation results | Unchanged |

Fixed: AAC level; rate or channel changes midway; cut-off and damaged AAC; lock crash
without git; silent empty result on a mistyped part; app crash on unfinished results;
app failing outside the repo folder; stale dashboard; sample page failing without
audio; misleading missing-file message; a test that would fail once the tag exists;
restart-from-zero tuning; audio formats missing from `.gitignore`; CRLF risk in shell
scripts; stale lesson references; three contradictions in the docs.

---

## 7. Viva questions with model answers

**Q1. Why does loudness change the prediction at all?**
`clip_mel` uses a fixed dB reference, and normalisation uses fixed training stats, so
a louder file gives larger inputs. Random gain of up to 6 dB in training gives some
tolerance, but not full invariance.

**Q2. Should we have normalised loudness per clip instead?**
That would remove this sensitivity, but it is a change to the model's input. It would
need retraining and a fresh comparison on valid. We fixed the app to match the
evaluated pipeline rather than change the pipeline.

**Q3. How did you find a bug no test caught?**
An odd number in other work. A peak of 1.415 was implausible for mastered music and
close to $\sqrt{2}$, which pointed at a downmix gain. Then a controlled test
confirmed it: one signal through both paths, comparing RMS.

**Q4. Does this bug affect your reported results?**
No. Stored scores come from the WAV path through librosa. The bug was only in the
app's AAC and M4A path, added after those scores were measured.

**Q5. What if a damaged file decodes to mostly noise?**
It gets classified, and probably reads "Not sure". We chose availability over
strictness for a demo. For evaluation this would be wrong, and evaluation does not
use this path.

**Q6. What if you run `tune_all.sh` after changing the SVM grid?**
It would skip the SVM and reuse old results. That is why the script's header says to
delete those files first. A stronger design would store a hash of the settings in
each result file and compare it. That is more machinery than one project needs.

**Q7. Why keep a test that uses a stand-in split file?**
To check the unlocked path of the lock without reading a real test row. Testing the
lock must not itself breach it.

**Q8. What if git is missing when the freeze happens?**
`is_test_unlocked` returns False and the test part stays locked. The failure is
visible and safe, and the fix is to install git.

---

## 8. Further reading

- ITU-R BS.775, *Multichannel stereophonic sound system* (downmix coefficients):
  https://www.itu.int/rec/R-REC-BS.775
- FFmpeg documentation, swresample options: https://ffmpeg.org/ffmpeg-resampler.html
- PyAV documentation: https://pyav.org/docs/stable/
- librosa, `load` and `to_mono`: https://librosa.org/doc/latest/generated/librosa.load.html
- D. Sculley et al. (2015). *Hidden technical debt in machine learning systems*.
  NeurIPS. https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems
- E. Breck et al. (2017). *The ML test score*. IEEE Big Data.
  https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/
