# Glossary

**Artist-cluster bootstrap.** A bootstrap that resamples whole artists (with all their clips) instead of single clips, because clips by one artist are correlated. Gives honest confidence intervals.

**Artist-disjoint split.** A split in which no artist has clips in more than one part (train, valid, test). Prevents the model from scoring by recognising artists.

**Baseline.** A simple, well-understood model (here kNN and logistic regression) that any more complex model must beat to justify its complexity.

**Best iteration.** The boosting round (counted from 0) at which the validation loss was lowest during early stopping; XGBoost then predicts with trees up to that round only. Our chosen XGBoost model stopped at best iteration 53, i.e. 54 rounds.

**Bias-variance trade-off.** Simple models err systematically (bias); flexible models change a lot with the training sample (variance). Hyperparameters such as k in kNN or C in logistic regression move a model along this trade-off.

**BLAS thread count.** How many CPU threads the linear-algebra library uses. It changes floating-point summation order, so iterative solvers such as lbfgs can stop at slightly different points; we fix it to 1 for reproducible results.

**Boosting.** Building an ensemble sequentially, each new weak model (here a shallow tree) correcting the errors of the sum of the previous ones. It mainly reduces bias, unlike bagging, which averages independent models to reduce variance.

**Bootstrap.** Estimating the uncertainty of a statistic by recomputing it on many datasets drawn with replacement from the observed one; the spread of those values gives a confidence interval.

**Chroma.** A 12-value per-frame vector of energy in each pitch class (C to B), with octaves folded together; describes harmony.

**Clip.** One ~30 s GTZAN audio excerpt with one genre label; the unit of prediction in this project.

**Clipping.** Distortion when a signal exceeds the largest representable value and its peaks are cut flat. We count samples with |x| >= 0.999 (`clip_share`).

**Cluster bootstrap.** A bootstrap that resamples whole clusters (here artists) with replacement, keeping correlated clips together.

**Column subsampling (colsample_bytree).** Growing each boosted tree on a random share of the features (0.8 here); makes trees less alike and reduces variance.

**Confidence interval (95%).** A range produced by a procedure that, over repeated samples, contains the true value 95% of the time. Here estimated with the bootstrap percentile method.

**Config file.** `config.yaml`: the single place where all constants (seed, audio settings, sizes, paths) are defined.

**Confusion matrix.** K x K table counting clips of each true class (rows) by predicted class (columns); the diagonal holds correct predictions.

**Cross-entropy loss.** -log of the probability a model gives to the true class; the loss logistic regression minimises. Confident wrong predictions cost the most.

**Curse of dimensionality.** In many dimensions, data is sparse and distances between points become similar, so "nearest" neighbours are not really near.

**Data leakage.** Information available during training or evaluation that would not be available at real prediction time, making scores look better than real performance.

**dBFS.** Decibels relative to full scale: 20 log10(amplitude), so 0 dBFS is the loudest possible digital level and real levels are at or below 0.

**DCT.** Discrete cosine transform; rewrites a sequence as a sum of cosines. Applied to a log-mel frame it gives MFCCs.

**Decision boundary.** The surface in feature space where a classifier switches from one predicted class to another; for logistic regression it is a hyperplane.

**Decision tree.** A model that predicts by a sequence of yes/no threshold questions on single features, ending in a leaf. Trained greedily by choosing the split that most reduces impurity (or, in XGBoost, most reduces the loss).

**Distance weighting.** A kNN option in which each neighbour's vote is weighted by 1/distance, so closer neighbours count more.

**Dual problem.** The SVM optimisation rewritten in terms of one multiplier alpha_i per training point, bounded by 0 and C. Data enter it only through dot products, which is what makes the kernel trick possible.

**Dummy classifier.** A model that ignores the features (always the most frequent class, or random draws with the training class frequencies); it gives the score floor.

**Early stopping.** Stopping iterative training when a validation loss has not improved for a set number of rounds (patience; 50 here) and keeping the model from the best round. Turns the number of trees into a quantity chosen from data.

**Exact repetition.** The same audio file present twice in a dataset; found here as identical md5 hashes (14 pairs in GTZAN).

**F1 score.** Harmonic mean of precision and recall for one class: 2PR / (P + R).

**Fault-filtered split.** The published GTZAN split (Kereliuk, Sturm & Larsen 2015) that removes duplicated, mislabelled and distorted clips and keeps artists disjoint: 443/197/290 clips.

**Fixed transform.** A feature computation that uses only the clip itself and preset constants; it cannot leak between splits.

**Floor.** The score a model with no information reaches under the chosen metric; measured here with dummy classifiers (valid macro-F1 0.018 and 0.128 on split C).

**Frame.** A short slice of audio (here 2048 samples, 93 ms) analysed as one spectrum; frames advance by the hop length (512 samples).

**Gamma (RBF).** The inverse width of the RBF kernel's Gaussian bumps. Large gamma makes each support vector influence only its close neighbourhood (risk of overfitting); small gamma gives smooth, nearly linear boundaries. sklearn's "scale" equals 1/(number of features x variance of X), 0.0175 on our standardised features.

**Gini impurity.** 1 minus the sum of squared class shares in a node: the chance two random points from the node have different labels. 0 for a pure node, 0.9 for an even 10-class mix.

**Gradient boosting.** Boosting viewed as gradient descent in function space: each new tree is fit to the negative gradient of the loss at the current predictions, and added multiplied by the learning rate.

**Group splitting.** Assigning all examples that share a group (clip, song, artist) to the same part, so the model is tested on unseen groups.

**GTZAN.** Genre dataset of Tzanetakis & Cook (2002): 1000 clips of 30 s, 10 genres, 22,050 Hz mono, collected 2000-2001 from CDs, radio and microphone recordings.

**Hann window.** A taper that goes smoothly to zero at both ends of a frame, reducing spectral leakage in the STFT.

**Hinge loss.** max(0, 1 - y f(x)): zero for points correctly classified outside the margin, growing linearly inside or beyond it. The SVM's loss; only points with non-zero hinge loss (support vectors) shape the boundary.

**HPSS.** Harmonic-percussive source separation: median filtering of the spectrogram along time (harmonic) and frequency (percussive) to split a signal into two parts.

**Hyperparameter.** A setting chosen before training rather than learned from data (k in kNN, C in logistic regression); we choose them on the valid part only.

**i.i.d..** Independent and identically distributed: examples drawn separately from one distribution; the assumption behind unbiased test estimates.

**Inverse regularisation strength (C).** sklearn's C: it multiplies the data loss against a fixed L2 penalty, so small C means strong regularisation and large C means weak.

**k-nearest neighbours (kNN).** Classifies a point by a vote among the k training points closest to it; no model is fitted.

**Kernel trick.** Replacing every dot product in an algorithm by a kernel, so a linear method works in a much richer feature space and draws non-linear boundaries in the original space.

**Kernel.** A function K(x, x') returning the dot product of two points after a (possibly infinite-dimensional) feature map, without computing that map. Acts as a similarity measure.

**L2 penalty.** Adds (1/2) times the sum of squared weights to the loss, pulling weights towards zero to limit overfitting.

**Lazy learning.** Learning that defers all work to prediction time (kNN only stores the training data), so fitting is instant but prediction is slow.

**lbfgs.** Limited-memory BFGS, a quasi-Newton optimiser; sklearn's default solver for logistic regression. It stops when the gradient is below a tolerance or at max_iter.

**Learned statistic.** A quantity computed from many clips (mean, std, PCA axes, selected features); must be fitted on training data only.

**Learning rate (shrinkage).** In boosting, the factor (0.05 or 0.1 here) multiplying each new tree's contribution. Smaller values regularise but need more trees; halving it roughly doubled the trees early stopping chose.

**Log-mel spectrogram.** A time-frequency picture of audio: short-time Fourier magnitudes pooled into mel-spaced bands and converted to dB.

**Logit.** The raw score w^T x + b of one class before the softmax turns all class scores into probabilities.

**Macro-F1.** The unweighted mean of the per-class F1 scores, so each genre counts equally regardless of how many clips it has.

**Majority vote.** Clip prediction = the class most windows predict; ties broken by mean probability.

**Margin (SVM).** The gap between a separating hyperplane and the nearest training points of either class, of width 2/||w||. The SVM chooses the boundary with the widest margin.

**max_depth.** The deepest a single boosted tree may grow; a depth-d tree can model interactions among up to d features. Deeper trees fit faster and overfit sooner.

**md5.** A 128-bit hash of a file's bytes; identical files have identical md5, so it detects exact duplicates and fingerprints a dataset copy.

**Mel scale.** A frequency scale spaced roughly like human pitch perception: fine at low frequencies, coarse at high ones.

**MFCC.** Mel-frequency cepstral coefficient: DCT of a log-mel frame; low coefficients summarise the spectral envelope (timbre).

**Micro-F1.** F1 computed from TP, FP and FN summed over all classes; equals accuracy in single-label classification.

**Mislabeling.** A clip whose genre label is wrong or doubtful; it caps achievable accuracy and teaches the model wrong mappings.

**mlogloss.** Multiclass log loss (cross-entropy): mean of -log(probability given to the true class). A model knowing nothing scores ln 10 = 2.303 with 10 classes. XGBoost's early-stopping metric here.

**Multinomial logistic regression.** A linear model with one weight vector per class whose scores go through a softmax; trained by minimising cross-entropy.

**Nyquist frequency.** Half the sampling rate, the highest frequency a sampled signal can represent (11,025 Hz for GTZAN).

**One-vs-one (OvO).** Multiclass strategy training one binary classifier per pair of classes (45 for 10 genres) and predicting by their votes. sklearn's SVC uses it.

**Onset strength.** A per-frame measure of how much spectral energy increases, i.e. how likely a note or hit starts; the input to tempo estimation.

**Overfitting.** Fitting the training data, noise included, better than the pattern generalises; seen as a large train-valid gap (kNN k=1: train macro-F1 1.0, valid 0.409).

**Paired bootstrap.** Bootstrapping the difference between two models using the same resamples for both, so shared test-set difficulty cancels.

**Pairwise coupling.** Combining the 45 pairwise probabilities of a one-vs-one model into one probability vector over all classes that agrees best with them (Wu, Lin & Weng 2004).

**Percentile interval.** Bootstrap CI taken as the 2.5th and 97.5th percentiles of the resampled statistic.

**Pinned version.** An exact package version (`name==X.Y.Z`) in `requirements.txt`, so everyone installs the same code.

**Pipeline.** An sklearn object chaining preprocessing and a model, so preprocessing such as scaling is refit on the training rows of each fit and never sees held-out data.

**Platt scaling.** Turning an SVM's decision values into probabilities with a fitted sigmoid 1/(1 + exp(A f + B)); sklearn fits A and B on internal 5-fold cross-validation scores.

**Precision.** Of the clips predicted as a class, the share that truly belong to it: TP / (TP + FP).

**PredefinedSplit.** sklearn splitter with a fixed train/validation assignment (-1 = always train, 0 = validation fold).

**PYTHONHASHSEED.** Environment variable fixing the randomisation of Python's string hashing; read at interpreter start.

**RBF kernel.** Radial basis function (Gaussian) kernel, exp(-gamma ||x - x'||^2): 1 for identical points, near 0 for distant ones. Its decision function is a weighted sum of Gaussian bumps on the support vectors.

**Recall.** Of the clips truly in a class, the share the model found: TP / (TP + FN).

**Refit.** Retraining the selected hyperparameter setting after a search; we refit on train only (GridSearchCV refit=False) so valid stays held out.

**Reproducibility.** Another person obtaining the same results from the same code, data and settings.

**RMS.** Root-mean-square of the samples; a measure of average signal level (loudness), reported in dBFS as `rms_db`.

**Sampling distribution.** The distribution a statistic would have over many independent samples of the same size; its spread is the statistic's uncertainty.

**Sampling rate.** Number of samples stored per second of audio; GTZAN uses 22,050 Hz.

**Seed.** The starting state of a pseudo-random number generator; the same seed produces the same sequence of "random" numbers.

**Slack variable.** xi_i >= 0 in the soft-margin SVM: how far training point i violates the margin. Their sum, multiplied by C, is added to the objective.

**Soft margin.** An SVM that allows margin violations at a cost C per unit of slack, so it works on overlapping classes. Small C tolerates violations (simpler boundary), large C punishes them (closer fit).

**Soft vote.** Combining window predictions by averaging their class probabilities, then taking the most probable class for the clip.

**Softmax.** Turns K scores into probabilities: exp(z_g) divided by the sum of exp(z_h) over all classes.

**Spectral bandwidth.** The magnitude-weighted spread of frequencies around the spectral centroid.

**Spectral centroid.** The magnitude-weighted mean frequency of a spectrum; perceived as brightness.

**Spectral roll-off.** The frequency below which a fixed share (85%) of a frame's spectral energy lies.

**Split A / B / B' / C.** Our four splits: A random windows; B random clips; B' random clips from the fault-filtered pool; C the published artist-filtered split. Same sizes, different leakage.

**Standardisation.** Rescaling each feature to mean 0 and standard deviation 1 using training-set statistics, so no feature dominates by its units.

**STFT.** Short-time Fourier transform: Fourier transforms of overlapping windowed frames, giving a time-frequency representation.

**Stochastic gradient boosting.** Gradient boosting in which each tree is grown on a random subsample of the training rows (subsample 0.8 here), and here also of the features; reduces variance and speeds up training (Friedman 2002).

**Stratification.** Sampling each class in proportion to its share of the pool, so every part has the same class mix.

**Support vector machine (SVM).** A maximum-margin classifier: minimises ||w||^2/2 plus C times the hinge losses; with a kernel it draws non-linear boundaries.

**Support vector.** A training point with a non-zero dual multiplier: on or inside the margin, or misclassified. The SVM's decision function depends only on these points (2,113 of 4,420 windows in our best SVM).

**Test lock.** `load_split` refuses the test part unless allow_test=True and the git tag pre-test-freeze exists; ensures the test set is used once.

**Training row.** One example the model is fit on; here one 3 s window, labelled with its clip's genre.

**Unit of prediction.** The object a prediction and its score refer to; here one 30 s clip.

**Virtual environment (venv).** A self-contained Python installation for one project, isolating its packages from others.

**Weighted-F1.** Per-class F1 averaged with weights equal to each class's share of true examples.

**Window.** A 3 s non-overlapping segment of a clip (10 per clip); the training row.

**Within-clip leakage.** Windows of one clip in both train and test; the model sees near-copies of test windows during training.

**XGBoost.** A gradient-boosted tree library (Chen & Guestrin 2016) using second-order (Newton) steps, L2-penalised leaf values and histogram-based split finding.

**Zero-crossing rate.** Fraction of consecutive samples whose sign changes; high for noisy or bright sounds.
