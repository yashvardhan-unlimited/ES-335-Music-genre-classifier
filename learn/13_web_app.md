# Lesson 13: Serving the model in a local web app

A model in a results file helps nobody. This step wraps the CNN in a small web
app: upload audio or pick a sample clip, get a genre, and browse every
validation result on a dashboard. Building it raises questions that training
never did. Does the app compute exactly what we evaluated? What happens with a
file that is not music, or a genre the model has never heard? How do we demo
without touching the locked test part?

---

## 1. What we built

| file | what it does |
|---|---|
| `app.py` | The Streamlit interface: two tabs, Classify and Results. Layout and charts only. |
| `src/webapp.py` | All logic, with no Streamlit in it: load the served CNN, decode and check audio, classify, pick samples, read `results/val`. |
| `tests/test_webapp.py` | 32 tests, including one that runs the app's path over all 197 valid clips and one that drives the app itself. |
| `.streamlit/config.toml` | Localhost only, 50 MB upload limit, no usage statistics, no Deploy button, light theme. |
| `config.yaml` (`app:`) | Which split is served, the 10-minute cut, samples per genre, how many genres to name, the "Not sure" cut-off. |

Commands (from the repo root):

```bash
.venv/Scripts/python.exe -m streamlit run app.py     # then open http://localhost:8501
.venv/Scripts/python.exe -m pytest tests/test_webapp.py -q
```

Stop the app with Ctrl+C in that terminal. It needs `models/`, `features/mel/`
and `data/raw/`, which are all git-ignored and already on this computer.

---

## 2. Concepts in depth

### 2.1 Training-serving skew

**Intuition.** A model is a function of its input *as prepared during training*.
If the app prepares input even slightly differently, the model silently receives
something it never saw, and the accuracy we measured no longer describes the app.
Nothing crashes: it just gets worse. This is **training-serving skew**, and it is
the classic failure of deployed models.

Our CNN expects a normalised window $x$ computed per mel bin $b$ with the
statistics of the training clips:

```latex
x_{b,t} = \frac{m_{b,t} - \mu_b}{\sigma_b},
\qquad
\hat p(g) = \frac{1}{10} \sum_{w=1}^{10} \operatorname{softmax}\bigl(f(x^{(w)})\bigr)_g .
```

$m$ is the log-mel in dB, $\mu_b$ and $\sigma_b$ come from
`results/mel_norm_C_s2.json`, and $\hat p$ is the soft vote. Using the wrong
$\mu$, the wrong number of mel bins or a different dB reference would all give
plausible-looking output.

**Our defence is a test, not care.** `classify_mel` must reproduce the evaluated
model. The test runs it over all 197 valid clips and requires the macro-F1 to
equal the stored value. Result on this project: 0.6348 on both sides.

### 2.2 From a fixed clip to any length

Training clips are 30 s, but an upload can be anything. We reuse `window_starts`,
which places 10 evenly spaced 130-frame windows over whatever length it receives:

```latex
s_i = \operatorname{round}\!\left(\frac{i\,(T - 130)}{9}\right), \quad i = 0, \dots, 9 .
```

**Worked example.** A 30 s GTZAN clip has $T = 1292$ frames, so $s_i$ is 0, 129,
258, up to 1162: exactly the evaluation windows. A 3-minute song has
$T = 1 + \lfloor 180 \times 22050 / 512 \rfloor = 7752$ frames, so the windows
start about 847 frames apart, roughly every 19.7 s. They sample the whole song,
intro and outro included, instead of only its first 30 s. A 10 s clip gives
windows that overlap. Each window still lasts 3 s, which is the size the CNN
trained on.

### 2.3 Inputs a model cannot refuse: the open-set problem

Softmax always produces ten numbers that sum to 1, so a classifier has no way to
say "this is none of the above". Feed it speech, a bird call or K-pop and it
still names one of GTZAN's genres, often confidently. Classifiers trained on a
closed list of classes face this **open-set problem**. There are two related
risks in real uploads:

- **Unknown classes:** the music is not any of our 10 genres.
- **Distribution shift:** the genre is known, but recording style differs. GTZAN
  clips are mostly from before 2002, and modern masters are mixed much louder.

So the app says plainly that it knows only ten genres. The measured scores
describe GTZAN clips by unseen artists and say nothing about a modern upload.

### 2.4 Why the score is not a percentage to trust

Lesson 11 measured that our CNNs are overconfident. Expected calibration error
for the served run on valid is 0.077:

```latex
\mathrm{ECE} = \sum_{k=1}^{10} \frac{n_k}{n}\,\bigl|\mathrm{acc}_k - \mathrm{conf}_k\bigr| .
```

**Worked example.** Sample `blues.00009` is labelled blues. The model answers
jazz with a score of 87%, and 10 of 10 windows agree, yet it is wrong. A high
score means the windows agree with each other. It does not mean a 87% chance of
being right. The app therefore calls these "model scores", tells the reader to
treat them as a ranking, and reports window agreement next to them.

### 2.5 Which number does the app quote?

We serve the best of three CNN runs, picked by valid macro-F1: seed 2 at 0.635,
against 0.580 and 0.615. Picking on valid is allowed, since choosing is what
valid is for. But the maximum of noisy scores is biased upwards (lesson 12), an
effect often called the **winner's curse**. So the dashboard headline is the mean
over seeds, 0.610, not the 0.635 of the run we picked.

---

## 3. Why this choice, and the alternatives we rejected

| choice | alternative rejected | reason |
|---|---|---|
| Streamlit | Gradio | Gradio is excellent for one input and one output, but weaker for a dashboard with several sections. |
| Streamlit | Flask or FastAPI plus HTML | Far more code, with routes and JavaScript to maintain, for the same screens. |
| Local only | Public host (Hugging Face Spaces) | Needs an exception for model files and copyright-safe samples, since GTZAN must not be republished. Local was the user's choice. |
| Python model on the server | ONNX in the browser via `docs/` | Puts weights on the public site and needs mel extraction rewritten in JavaScript, a second pipeline that could drift (2.1). |
| Samples from valid, chosen at random | Hand-picked samples the model gets right | Hand-picking makes a misleading demo. Random selection keeps wrong answers, like `blues.00009`. |
| Samples from valid | Samples from test | Predicting test clips before the freeze breaks the lock. |
| One run: best on valid | Average of the three seeds | An ensemble is a new model with no measured score. The final model is decided at the freeze. |
| Raw scores, labelled | Temperature-scaled scores | On seed 2, cross-fitted scaling worsened clip ECE from 0.077 to 0.114. |
| "Not sure" under a top score of 0.7, chart kept | Always state a genre | Under 0.7 the served run was right on 42.7% of valid clips, so a flat answer would overclaim. |
| Cut-off on top score alone | Window agreement | Accuracy did not rise steadily with agreement (0.348 at 6 of 10 windows, 0.538 at 4 of 10). |
| 10 windows across the file | First 30 s only | Intros are often unrepresentative of a song. |
| AAC and M4A through PyAV, after librosa fails | Decoding every format with PyAV | librosa stays first, so WAV, MP3, FLAC and OGG keep the path the stored scores were measured on. |
| PyAV | System-wide ffmpeg, or `imageio-ffmpeg` | PyAV ships its own decoder, so nothing extra is installed by hand, and it reads uploads from memory. |

---

## 4. Code walkthrough

**`src/webapp.py`**

- **L45-58 `load_classifier`**: reads the per-seed scores from
  `C_cnn_summary.json` and takes the best with `max`, so the seed is derived from
  results, not hand-copied. L51 asserts the checkpoint's class order equals the
  config's, because a swapped order would silently mislabel every prediction. L54
  loads the normalisation stats saved from that run's own training clips.
- **L61-75 `decode_with_av`**: reads AAC and M4A, which libsndfile cannot. L64 rewinds
  the upload, because librosa has already read from it. L66 asks for mono 32-bit
  floats at the project rate, the same form `librosa.load` returns. L73-74 stop
  decoding at the 10-minute cut, since a 50 MB low-bitrate file can hold many hours.
- **L78-83 `decode`**: tries librosa first (L81, with the 10-minute cut), so the
  evaluated formats never touch PyAV, and falls back only if librosa fails.
- **L86-100 `load_audio`**: uploads are untrusted input, so this is the one place
  they are checked. L94-95 turn any decoder error into one readable `ValueError`,
  because each backend raises its own type. L96-97 reject anything shorter than one
  window. L98 uses `not max > 0` deliberately:
  the test is False for silence and also for NaN, since any comparison with NaN
  is False.
- **L103-112 `classify_mel`**: L109 cuts the windows, L110 normalises and adds the
  channel dimension, L111-112 run the network without gradients. This mirrors
  `MelWindowDataset` in eval mode, and the 197-clip test proves it.
- **L115-121 `summarise`**: the soft vote, then a count of windows that agree with
  the winner. Agreement is a diagnostic only and never changes the answer. L121
  sets `unsure` when the top score is under `app.min_confidence`. The genre is
  still computed, so nothing about the vote changes, only how the app words it.
- **L129-131 `classify_sample`**: uses the cached log-mel, the exact input the
  stored score used, so a sample's prediction equals the evaluated one.
- **L134-139 `sample_clips`**: L136 loads through `load_split(..., "valid")`, so
  test clips cannot enter. L138 samples with the config seed before any score
  exists to look at.
- **L142-153 `load_results`**: `RESULT_FILE` (L31) matches only per-run files, so
  analysis JSONs are skipped.
- **L156-161 `missing_runs`**: lists what the protocol expects but has not got.
  The dashboard shows this, so a partial table can never pass as complete.
- **L171-175 `ladder`**: L174-175 reindex to every model and split, so an unfinished
  cell stays as NaN and draws a gap instead of a line joining across it.

**`app.py`**

- **L33-55, the caches**: Streamlit reruns the whole script on every click.
  `cache_resource` keeps one model in memory; `cache_data` stores results by input.
  `max_entries=8` (L53) stops uploads from filling memory.
- **L66-75 `bars`**: `labelled` prints values on only the first rows, so the chart
  stays readable, and the tooltip and table carry the rest.
- **L78-94 `show_prediction`**: one function for both inputs. L81 picks the heading:
  "Not sure: best guess ..." or "Prediction: ...". L85-87 add the explanation. The
  chart is drawn either way. With a known genre it also says "Correct" or "Wrong".
- **L206-216 `results_tab`**: the validation banner comes first, then the
  unfinished-runs warning, and only then any number.

---

## 5. Pitfalls

1. **Re-implementing preprocessing in the app.** A copied function drifts from the
   original. We import `clip_mel`, `window_starts` and `GenreCNN` instead.
2. **Forgetting `model.eval()`.** Dropout would stay on and BatchNorm would use
   each batch's statistics, so predictions would be random and would depend on
   what else is in the batch.
3. **Normalising with stats from the wrong run.** Each run has its own stats file.
   Mixing them is skew, and nothing would crash.
4. **Letting test clips into the demo.** This is leakage by another route: looking
   at test predictions is a look at test (lesson 12). The sample list is safe by
   construction. **The upload box cannot protect you**, though: uploading a GTZAN
   file from split C's test part is still a peek, so do not do it before the freeze.
5. **Quoting 0.635.** That is the winner's score. Quote 0.610 +/- 0.028, and call
   it validation until test exists.
6. **Reading a demo as evidence.** A few good uploads prove nothing. The measured
   score comes from 197 clips.
7. **Publishing by accident.** `server.address = "localhost"` keeps the app on this
   computer. Do not change it on a shared network without thinking about uploads.
8. **Reading "Prediction" as a promise.** The cut-off catches weak answers, not all
   wrong ones: `blues.00009` scores 87% and is still wrong. A high score on music
   unlike GTZAN proves even less, since 0.7 was only measured on GTZAN clips.
9. **Testing a change against an old server.** `fileWatcherType = "none"` means a
   running app keeps its old code. Stop it with Ctrl+C and start it again after any
   change. Our first AAC browser test failed for exactly this reason.

---

## 6. Results from this step

Quoted from the code's output, all on **validation** data:

| check | result |
|---|---|
| Full test suite | 101 passed (69 earlier + 32 for the app) |
| App path on all 197 valid clips | 0.6348, equal to the stored score of run seed 2 |
| Served CNN run | seed 2, valid macro-F1 0.635, clip ECE 0.077 |
| Browser run, MP3 made from `jazz.00007` (a valid clip) | jazz 94%, blues 3%, classical 2%; 10 of 10 windows |
| Sample `blues.00009` | jazz 87%, blues 8%, classical 2%: wrong, and shown as wrong |
| 30 valid clips re-encoded as AAC, then as M4A | Same top genre on 29 of 30 in each format; largest score change per clip: median 0.011, worst 0.063 |
| Browser run, AAC and M4A made from `jazz.00004` (a valid clip) | Prediction jazz in both; 10 of 10 windows |
| Top score under 0.7, 197 valid clips, served run | 103 clips, accuracy 0.427 |
| Top score 0.7 or more, same clips | 94 clips, accuracy 0.883 |
| Same split on the other two runs | seed 0: 0.466 below, 0.875 above; seed 1: 0.360 below, 0.815 above |
| Unreadable file | "This file could not be read as audio." |
| 1 s file | "This audio is too short: at least 3 seconds are needed." |

Dashboard on split C (valid clip macro-F1):

| model | macro-F1 |
|---|---|
| CNN (mean of 3 runs) | 0.610, accuracy 0.618 |
| Logistic regression | 0.468 |
| RBF SVM | 0.458 |
| kNN | 0.424 |
| XGBoost | 0.419 |
| Dummy (stratified) | 0.128 |
| Dummy (most frequent) | 0.018 |

CNN per genre on split C, averaged over 3 runs: classical 0.90, hiphop 0.85, metal
0.83, disco 0.71, blues 0.59, jazz 0.57, reggae 0.55, pop 0.44, country 0.35,
rock 0.31.

The dashboard currently warns that 7 CNN runs are missing (`B_s3`, `B_s4`,
`Bp_s0` to `Bp_s4`). Those cells fill in once the runs finish. Figures shown in
the Analyses section come from `results/figures/`.

---

## 7. Viva questions with model answers

**Q1. How do you know the app computes the same thing you evaluated?**
A test runs the app's own `classify_mel` over all 197 valid clips and requires the
stored macro-F1, 0.6348. The code is shared too: `clip_mel`, `window_starts` and
`GenreCNN` are imported, not copied.

**Q2. What is training-serving skew?**
Input prepared differently at serving time than at training time. The model still
returns an answer, so nothing fails visibly. It just stops matching the measured
performance.

**Q3. Your app says 87% jazz on a blues clip. Is the model broken?**
No. The score is not calibrated and shows agreement between windows: 10 of 10
windows picked jazz here. The CNN's macro-F1 is about 0.61, so many clips will be
wrong, and blues is among its weaker genres at 0.59.

**Q4. Why not show only the samples it gets right?**
That would present a better model than the one we measured. Random valid clips
show typical behaviour, mistakes included.

**Q5. Why does the dashboard headline 0.610 when you serve a 0.635 model?**
0.635 is the maximum of three noisy scores, chosen on the same data, so it is
biased upwards. The mean over seeds is the fairer estimate of what this training
recipe gives.

**Q6. What if someone uploads a podcast?**
It gets one of the ten genres anyway. Softmax cannot refuse, which is the open-set
problem. To handle it, we would need a rejection rule, such as a threshold on a
calibrated score, tested on non-music audio.

**Q7. Could the app leak test data?**
The code cannot: samples come from `load_split("C", "valid")`, and scores come from
`results/val`. A person still can, by uploading a test clip by hand, so that is a
rule we must follow ourselves until the freeze.

**Q8. What if you hosted this publicly?**
We would need the model weights outside `models/`, a replacement for GTZAN samples
because that audio must not be republished, rate and size limits, and a privacy
note because uploads would leave the user's computer.

**Q9. What if an upload is 3 s long?**
It works. All 10 windows overlap almost completely, so we get little more than one
window's evidence and the vote adds almost nothing. Under 3 s, the app refuses.

**Q10. Does an AAC upload get the same answer as the original WAV?**
Not always. AAC is lossy and PyAV resamples with its own filter, so the log-mel
changes slightly: this is a small training-serving skew. We measured it instead of
assuming it away: on 30 valid clips the top genre held on 29, and no genre's score
moved by more than 0.063. That is 30 clips at one bitrate, not a scored evaluation,
so 0.635 still describes WAV input only. Why not put a real `.aac` in the tests?
The tests encode one in memory, so no audio is committed.

**Q11. Why does the app say "Not sure" at 0.7, and why not a round 0.5?**
This is selective prediction: the model may hold back an answer it cannot support.
We measured accuracy by top score on the 197 valid clips. It stays near 0.43 from
0.4 up to 0.7, then jumps to about 0.85. A cut-off of 0.5 would still state a
genre on the 49 clips between 0.5 and 0.7, where the model is right under half the
time. All three runs show the same split. The price is coverage: 103 of 197 valid
clips (52%) now say "Not sure". It only changes the wording, never the predicted
genre, so it cannot change macro-F1. The cut-off comes from valid, never test, and
because it was picked on these clips, 0.883 is slightly optimistic.

---

## 8. Further reading

- Streamlit documentation: https://docs.streamlit.io
- Streamlit, app testing: https://docs.streamlit.io/develop/concepts/app-testing
- Vega-Altair documentation: https://altair-viz.github.io
- D. Sculley et al. (2015). *Hidden technical debt in machine learning systems*.
  NeurIPS. https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems
- M. Zinkevich. *Rules of Machine Learning* (training-serving skew):
  https://developers.google.com/machine-learning/guides/rules-of-ml
- C. Guo et al. (2017). *On calibration of modern neural networks*. ICML.
  https://arxiv.org/abs/1706.04599
- D. Hendrycks, K. Gimpel (2017). *A baseline for detecting misclassified and
  out-of-distribution examples in neural networks*. ICLR.
  https://arxiv.org/abs/1610.02136
- Y. Geifman, R. El-Yaniv (2017). *Selective classification for deep neural
  networks*. NeurIPS. https://arxiv.org/abs/1705.08500
- OWASP, *File Upload Cheat Sheet*:
  https://cheatsheetseries.owasp.org/cheatsheets/File_Upload_Cheat_Sheet.html
