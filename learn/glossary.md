# Glossary

**Artist-cluster bootstrap.** A bootstrap that resamples whole artists (with all their clips) instead of single clips, because clips by one artist are correlated. Gives honest confidence intervals.

**Artist-disjoint split.** A split in which no artist has clips in more than one part (train, valid, test). Prevents the model from scoring by recognising artists.

**Baseline.** A simple, well-understood model (here kNN and logistic regression) that any more complex model must beat to justify its complexity.

**Bias-variance trade-off.** Simple models err systematically (bias); flexible models change a lot with the training sample (variance). Hyperparameters such as k in kNN or C in logistic regression move a model along this trade-off.

**BLAS thread count.** How many CPU threads the linear-algebra library uses. It changes floating-point summation order, so iterative solvers such as lbfgs can stop at slightly different points; we fix it to 1 for reproducible results.

**Bootstrap.** Estimating the uncertainty of a statistic by recomputing it on many datasets drawn with replacement from the observed one; the spread of those values gives a confidence interval.

**Chroma.** A 12-value per-frame vector of energy in each pitch class (C to B), with octaves folded together; describes harmony.

**Clip.** One ~30 s GTZAN audio excerpt with one genre label; the unit of prediction in this project.

**Clipping.** Distortion when a signal exceeds the largest representable value and its peaks are cut flat. We count samples with |x| >= 0.999 (`clip_share`).

**Cluster bootstrap.** A bootstrap that resamples whole clusters (here artists) with replacement, keeping correlated clips together.

**Confidence interval (95%).** A range produced by a procedure that, over repeated samples, contains the true value 95% of the time. Here estimated with the bootstrap percentile method.

**Config file.** `config.yaml`: the single place where all constants (seed, audio settings, sizes, paths) are defined.

**Confusion matrix.** K x K table counting clips of each true class (rows) by predicted class (columns); the diagonal holds correct predictions.

**Cross-entropy loss.** -log of the probability a model gives to the true class; the loss logistic regression minimises. Confident wrong predictions cost the most.

**Curse of dimensionality.** In many dimensions, data is sparse and distances between points become similar, so "nearest" neighbours are not really near.

**Data leakage.** Information available during training or evaluation that would not be available at real prediction time, making scores look better than real performance.

**dBFS.** Decibels relative to full scale: 20 log10(amplitude), so 0 dBFS is the loudest possible digital level and real levels are at or below 0.

**DCT.** Discrete cosine transform; rewrites a sequence as a sum of cosines. Applied to a log-mel frame it gives MFCCs.

**Decision boundary.** The surface in feature space where a classifier switches from one predicted class to another; for logistic regression it is a hyperplane.

**Distance weighting.** A kNN option in which each neighbour's vote is weighted by 1/distance, so closer neighbours count more.

**Dummy classifier.** A model that ignores the features (always the most frequent class, or random draws with the training class frequencies); it gives the score floor.

**Exact repetition.** The same audio file present twice in a dataset; found here as identical md5 hashes (14 pairs in GTZAN).

**F1 score.** Harmonic mean of precision and recall for one class: 2PR / (P + R).

**Fault-filtered split.** The published GTZAN split (Kereliuk, Sturm & Larsen 2015) that removes duplicated, mislabelled and distorted clips and keeps artists disjoint: 443/197/290 clips.

**Fixed transform.** A feature computation that uses only the clip itself and preset constants; it cannot leak between splits.

**Floor.** The score a model with no information reaches under the chosen metric; measured here with dummy classifiers (valid macro-F1 0.018 and 0.128 on split C).

**Frame.** A short slice of audio (here 2048 samples, 93 ms) analysed as one spectrum; frames advance by the hop length (512 samples).

**Group splitting.** Assigning all examples that share a group (clip, song, artist) to the same part, so the model is tested on unseen groups.

**GTZAN.** Genre dataset of Tzanetakis & Cook (2002): 1000 clips of 30 s, 10 genres, 22,050 Hz mono, collected 2000-2001 from CDs, radio and microphone recordings.

**Hann window.** A taper that goes smoothly to zero at both ends of a frame, reducing spectral leakage in the STFT.

**HPSS.** Harmonic-percussive source separation: median filtering of the spectrogram along time (harmonic) and frequency (percussive) to split a signal into two parts.

**Hyperparameter.** A setting chosen before training rather than learned from data (k in kNN, C in logistic regression); we choose them on the valid part only.

**i.i.d..** Independent and identically distributed: examples drawn separately from one distribution; the assumption behind unbiased test estimates.

**Inverse regularisation strength (C).** sklearn's C: it multiplies the data loss against a fixed L2 penalty, so small C means strong regularisation and large C means weak.

**k-nearest neighbours (kNN).** Classifies a point by a vote among the k training points closest to it; no model is fitted.

**L2 penalty.** Adds (1/2) times the sum of squared weights to the loss, pulling weights towards zero to limit overfitting.

**Lazy learning.** Learning that defers all work to prediction time (kNN only stores the training data), so fitting is instant but prediction is slow.

**lbfgs.** Limited-memory BFGS, a quasi-Newton optimiser; sklearn's default solver for logistic regression. It stops when the gradient is below a tolerance or at max_iter.

**Learned statistic.** A quantity computed from many clips (mean, std, PCA axes, selected features); must be fitted on training data only.

**Log-mel spectrogram.** A time-frequency picture of audio: short-time Fourier magnitudes pooled into mel-spaced bands and converted to dB.

**Logit.** The raw score w^T x + b of one class before the softmax turns all class scores into probabilities.

**Macro-F1.** The unweighted mean of the per-class F1 scores, so each genre counts equally regardless of how many clips it has.

**Majority vote.** Clip prediction = the class most windows predict; ties broken by mean probability.

**md5.** A 128-bit hash of a file's bytes; identical files have identical md5, so it detects exact duplicates and fingerprints a dataset copy.

**Mel scale.** A frequency scale spaced roughly like human pitch perception: fine at low frequencies, coarse at high ones.

**MFCC.** Mel-frequency cepstral coefficient: DCT of a log-mel frame; low coefficients summarise the spectral envelope (timbre).

**Micro-F1.** F1 computed from TP, FP and FN summed over all classes; equals accuracy in single-label classification.

**Mislabeling.** A clip whose genre label is wrong or doubtful; it caps achievable accuracy and teaches the model wrong mappings.

**Multinomial logistic regression.** A linear model with one weight vector per class whose scores go through a softmax; trained by minimising cross-entropy.

**Nyquist frequency.** Half the sampling rate, the highest frequency a sampled signal can represent (11,025 Hz for GTZAN).

**Onset strength.** A per-frame measure of how much spectral energy increases, i.e. how likely a note or hit starts; the input to tempo estimation.

**Overfitting.** Fitting the training data, noise included, better than the pattern generalises; seen as a large train-valid gap (kNN k=1: train macro-F1 1.0, valid 0.409).

**Paired bootstrap.** Bootstrapping the difference between two models using the same resamples for both, so shared test-set difficulty cancels.

**Percentile interval.** Bootstrap CI taken as the 2.5th and 97.5th percentiles of the resampled statistic.

**Pinned version.** An exact package version (`name==X.Y.Z`) in `requirements.txt`, so everyone installs the same code.

**Pipeline.** An sklearn object chaining preprocessing and a model, so preprocessing such as scaling is refit on the training rows of each fit and never sees held-out data.

**Precision.** Of the clips predicted as a class, the share that truly belong to it: TP / (TP + FP).

**PredefinedSplit.** sklearn splitter with a fixed train/validation assignment (-1 = always train, 0 = validation fold).

**PYTHONHASHSEED.** Environment variable fixing the randomisation of Python's string hashing; read at interpreter start.

**Recall.** Of the clips truly in a class, the share the model found: TP / (TP + FN).

**Refit.** Retraining the selected hyperparameter setting after a search; we refit on train only (GridSearchCV refit=False) so valid stays held out.

**Reproducibility.** Another person obtaining the same results from the same code, data and settings.

**RMS.** Root-mean-square of the samples; a measure of average signal level (loudness), reported in dBFS as `rms_db`.

**Sampling distribution.** The distribution a statistic would have over many independent samples of the same size; its spread is the statistic's uncertainty.

**Sampling rate.** Number of samples stored per second of audio; GTZAN uses 22,050 Hz.

**Seed.** The starting state of a pseudo-random number generator; the same seed produces the same sequence of "random" numbers.

**Soft vote.** Combining window predictions by averaging their class probabilities, then taking the most probable class for the clip.

**Softmax.** Turns K scores into probabilities: exp(z_g) divided by the sum of exp(z_h) over all classes.

**Spectral bandwidth.** The magnitude-weighted spread of frequencies around the spectral centroid.

**Spectral centroid.** The magnitude-weighted mean frequency of a spectrum; perceived as brightness.

**Spectral roll-off.** The frequency below which a fixed share (85%) of a frame's spectral energy lies.

**Split A / B / B' / C.** Our four splits: A random windows; B random clips; B' random clips from the fault-filtered pool; C the published artist-filtered split. Same sizes, different leakage.

**Standardisation.** Rescaling each feature to mean 0 and standard deviation 1 using training-set statistics, so no feature dominates by its units.

**STFT.** Short-time Fourier transform: Fourier transforms of overlapping windowed frames, giving a time-frequency representation.

**Stratification.** Sampling each class in proportion to its share of the pool, so every part has the same class mix.

**Test lock.** `load_split` refuses the test part unless allow_test=True and the git tag pre-test-freeze exists; ensures the test set is used once.

**Training row.** One example the model is fit on; here one 3 s window, labelled with its clip's genre.

**Unit of prediction.** The object a prediction and its score refer to; here one 30 s clip.

**Virtual environment (venv).** A self-contained Python installation for one project, isolating its packages from others.

**Weighted-F1.** Per-class F1 averaged with weights equal to each class's share of true examples.

**Window.** A 3 s non-overlapping segment of a clip (10 per clip); the training row.

**Within-clip leakage.** Windows of one clip in both train and test; the model sees near-copies of test windows during training.

**Zero-crossing rate.** Fraction of consecutive samples whose sign changes; high for noisy or bright sounds.
