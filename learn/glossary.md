# Glossary

**Ablation.** Removing or changing one component (a feature group, the window length) while keeping everything else fixed and retraining, to measure what that component contributes.

**Adam.** An optimiser that keeps running averages of each parameter's gradient and squared gradient and steps by their ratio, giving every parameter its own step size. Used with learning rate 0.001.

**AdamW.** Adam with decoupled weight decay: the decay is subtracted from the weights directly instead of being added to the gradient (where Adam would rescale it).

**Affine transformation.** A map of the form z = a x + b (per feature here). Standardisation is affine, so standardising data that was already affinely rescaled gives the same result as standardising the original data.

**Artist-cluster bootstrap.** A bootstrap that resamples whole artists (with all their clips) instead of single clips, because clips by one artist are correlated. Gives honest confidence intervals.

**Artist-disjoint split.** A split in which no artist has clips in more than one part (train, valid, test). Prevents the model from scoring by recognising artists.

**Backpropagation.** Computing the gradient of the loss with respect to every weight of a network by applying the chain rule layer by layer from the output back to the input.

**Baseline.** A simple, well-understood model (here kNN and logistic regression) that any more complex model must beat to justify its complexity.

**Batch normalisation.** Standardising each channel over the mini-batch, then rescaling with learned gamma and shifting with learned beta; uses running averages at evaluation time. Stabilises and speeds up training.

**Best iteration.** The boosting round (counted from 0) at which the validation loss was lowest during early stopping; XGBoost then predicts with trees up to that round only. Our chosen XGBoost model stopped at best iteration 53, i.e. 54 rounds.

**Bias-variance trade-off.** Simple models err systematically (bias); flexible models change a lot with the training sample (variance). Hyperparameters such as k in kNN or C in logistic regression move a model along this trade-off.

**BLAS thread count.** How many CPU threads the linear-algebra library uses. It changes floating-point summation order, so iterative solvers such as lbfgs can stop at slightly different points; we fix it to 1 for reproducible results.

**Boosting.** Building an ensemble sequentially, each new weak model (here a shallow tree) correcting the errors of the sum of the previous ones. It mainly reduces bias, unlike bagging, which averages independent models to reduce variance.

**Bootstrap.** Estimating the uncertainty of a statistic by recomputing it on many datasets drawn with replacement from the observed one; the spread of those values gives a confidence interval.

**Calibration.** A classifier is calibrated if, among predictions made with confidence p, a fraction p is correct. Our CNNs are over-confident per window (every fitted temperature is above 1).

**Channel.** One of the stacked 2-D maps a convolution layer outputs (one per filter); our blocks have 32, 64, 128 and 128.

**Chroma.** A 12-value per-frame vector of energy in each pitch class (C to B), with octaves folded together; describes harmony.

**Clip.** One ~30 s GTZAN audio excerpt with one genre label; the unit of prediction in this project.

**Clipping.** Distortion when a signal exceeds the largest representable value and its peaks are cut flat. We count samples with |x| >= 0.999 (`clip_share`).

**Cluster bootstrap.** A bootstrap that resamples whole clusters (here artists) with replacement, keeping correlated clips together.

**Column subsampling (colsample_bytree).** Growing each boosted tree on a random share of the features (0.8 here); makes trees less alike and reduces variance.

**Confidence interval (95%).** A range produced by a procedure that, over repeated samples, contains the true value 95% of the time. Here estimated with the bootstrap percentile method.

**Config file.** `config.yaml`: the single place where all constants (seed, audio settings, sizes, paths) are defined.

**Confusion matrix.** K x K table counting clips of each true class (rows) by predicted class (columns); the diagonal holds correct predictions.

**Convolution (2-D).** Sliding a small learned filter over an image and computing, at every position, a weighted sum of the local patch; the output is a feature map showing where the pattern occurs.

**Cross-entropy loss.** -log of the probability a model gives to the true class; the loss logistic regression minimises. Confident wrong predictions cost the most.

**Cross-fitting.** Fitting a quantity on one half of the data and evaluating it on the other half, then swapping, so the evaluation never uses data the fit saw. We cross-fit the CNN temperature over the two valid artist halves.

**Curse of dimensionality.** In many dimensions, data is sparse and distances between points become similar, so "nearest" neighbours are not really near.

**Data augmentation.** Randomly transforming training inputs in ways that keep the label (crops, gain, masking) so the model sees more varied data and overfits less. Never applied to evaluation data.

**Data leakage.** Information available during training or evaluation that would not be available at real prediction time, making scores look better than real performance.

**dBFS.** Decibels relative to full scale: 20 log10(amplitude), so 0 dBFS is the loudest possible digital level and real levels are at or below 0.

**DCT.** Discrete cosine transform; rewrites a sequence as a sum of cosines. Applied to a log-mel frame it gives MFCCs.

**Decision boundary.** The surface in feature space where a classifier switches from one predicted class to another; for logistic regression it is a hyperplane.

**Decision tree.** A model that predicts by a sequence of yes/no threshold questions on single features, ending in a leaf. Trained greedily by choosing the split that most reduces impurity (or, in XGBoost, most reduces the loss).

**Distance weighting.** A kNN option in which each neighbour's vote is weighted by 1/distance, so closer neighbours count more.

**Dropout.** During training, setting each unit to zero with probability p (0.3 here) and scaling the rest by 1/(1-p), so the network cannot rely on any single unit. Off at evaluation.

**Dual problem.** The SVM optimisation rewritten in terms of one multiplier alpha_i per training point, bounded by 0 and C. Data enter it only through dot products, which is what makes the kernel trick possible.

**Dummy classifier.** A model that ignores the features (always the most frequent class, or random draws with the training class frequencies); it gives the score floor.

**Early stopping.** Stopping iterative training when a validation loss has not improved for a set number of rounds (patience; 50 here) and keeping the model from the best round. Turns the number of trees into a quantity chosen from data.

**Epoch.** One pass over the training data; for our CNN 10 random crops per training clip (4,420 items on split C).

**Exact repetition.** The same audio file present twice in a dataset; found here as identical md5 hashes (14 pairs in GTZAN).

**Expected calibration error (ECE).** The bin-share-weighted mean gap between accuracy and mean confidence over confidence bins (10 equal-width bins here). 0 means perfectly calibrated; CNN seed 1 drops from 0.196 to 0.038 per window after temperature scaling.

**F1 score.** Harmonic mean of precision and recall for one class: 2PR / (P + R).

**Fault-filtered split.** The published GTZAN split (Kereliuk, Sturm & Larsen 2015) that removes duplicated, mislabelled and distorted clips and keeps artists disjoint: 443/197/290 clips.

**Feature map.** The output image of one convolution filter: large where the filter's pattern is present in the input.

**Fixed transform.** A feature computation that uses only the clip itself and preset constants; it cannot leak between splits.

**Floor.** The score a model with no information reaches under the chosen metric; measured here with dummy classifiers (valid macro-F1 0.018 and 0.128 on split C).

**Frame.** A short slice of audio (here 2048 samples, 93 ms) analysed as one spectrum; frames advance by the hop length (512 samples).

**Gamma (RBF).** The inverse width of the RBF kernel's Gaussian bumps. Large gamma makes each support vector influence only its close neighbourhood (risk of overfitting); small gamma gives smooth, nearly linear boundaries. sklearn's "scale" equals 1/(number of features x variance of X), 0.0175 on our standardised features.

**Gini impurity.** 1 minus the sum of squared class shares in a node: the chance two random points from the node have different labels. 0 for a pure node, 0.9 for an even 10-class mix.

**Global average pooling.** Replacing each feature map by its mean, giving one number per channel; removes the need for a large dense layer and makes the output independent of where patterns occur.

**Gradient boosting.** Boosting viewed as gradient descent in function space: each new tree is fit to the negative gradient of the loss at the current predictions, and added multiplied by the learning rate.

**Group splitting.** Assigning all examples that share a group (clip, song, artist) to the same part, so the model is tested on unseen groups.

**GroupKFold.** sklearn cross-validation splitter that keeps all rows of one group (clip, artist, patient) in the same fold, so no group is on both sides of a split.

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

**KFold.** sklearn splitter dividing rows into k folds, each used once for validation. With several rows per source and no grouping it leaks; with shuffle=True every window of a clip lands in a random fold.

**L2 penalty.** Adds (1/2) times the sum of squared weights to the loss, pulling weights towards zero to limit overfitting.

**Lazy learning.** Learning that defers all work to prediction time (kNN only stores the training data), so fitting is instant but prediction is slow.

**lbfgs.** Limited-memory BFGS, a quasi-Newton optimiser; sklearn's default solver for logistic regression. It stops when the gradient is below a tolerance or at max_iter.

**Learned statistic.** A quantity computed from many clips (mean, std, PCA axes, selected features); must be fitted on training data only.

**Learning curve.** Validation score as a function of training-set size. Still rising at full size means the model is data-limited (variance); flat means more of the same data will not help (bias). We grow the training set by artists, not clips.

**Learning rate (shrinkage).** In boosting, the factor (0.05 or 0.1 here) multiplying each new tree's contribution. Smaller values regularise but need more trees; halving it roughly doubled the trees early stopping chose.

**Log-mel spectrogram.** A time-frequency picture of audio: short-time Fourier magnitudes pooled into mel-spaced bands and converted to dB.

**Logit.** The raw score w^T x + b of one class before the softmax turns all class scores into probabilities.

**Macro-F1.** The unweighted mean of the per-class F1 scores, so each genre counts equally regardless of how many clips it has.

**Majority vote.** Clip prediction = the class most windows predict; ties broken by mean probability.

**Margin (SVM).** The gap between a separating hyperplane and the nearest training points of either class, of width 2/||w||. The SVM chooses the boundary with the widest margin.

**Max-pooling.** Keeping the largest value in each small block (2 x 2 here), halving height and width; adds tolerance to small shifts and grows the receptive field.

**max_depth.** The deepest a single boosted tree may grow; a depth-d tree can model interactions among up to d features. Deeper trees fit faster and overfit sooner.

**md5.** A 128-bit hash of a file's bytes; identical files have identical md5, so it detects exact duplicates and fingerprints a dataset copy.

**Mel scale.** A frequency scale spaced roughly like human pitch perception: fine at low frequencies, coarse at high ones.

**Metric mismatch.** Selecting hyperparameters by one score (e.g. window accuracy) while reporting another (clip macro-F1). Not leakage, but it can pick settings that are worse on the reported metric.

**MFCC.** Mel-frequency cepstral coefficient: DCT of a log-mel frame; low coefficients summarise the spectral envelope (timbre).

**Micro-F1.** F1 computed from TP, FP and FN summed over all classes; equals accuracy in single-label classification.

**Mini-batch.** The group of training examples (64 here) whose average loss gives one gradient step.

**Mislabeling.** A clip whose genre label is wrong or doubtful; it caps achievable accuracy and teaches the model wrong mappings.

**mlogloss.** Multiclass log loss (cross-entropy): mean of -log(probability given to the true class). A model knowing nothing scores ln 10 = 2.303 with 10 classes. XGBoost's early-stopping metric here.

**Multinomial logistic regression.** A linear model with one weight vector per class whose scores go through a softmax; trained by minimising cross-entropy.

**Nyquist frequency.** Half the sampling rate, the highest frequency a sampled signal can represent (11,025 Hz for GTZAN).

**One-vs-one (OvO).** Multiclass strategy training one binary classifier per pair of classes (45 for 10 genres) and predicting by their votes. sklearn's SVC uses it.

**Onset strength.** A per-frame measure of how much spectral energy increases, i.e. how likely a note or hit starts; the input to tempo estimation.

**Over-confidence.** Predicted confidence systematically higher than accuracy; shows up as reliability-diagram points below the diagonal and as spikes in cross-entropy on wrong predictions.

**Overfitting.** Fitting the training data, noise included, better than the pattern generalises; seen as a large train-valid gap (kNN k=1: train macro-F1 1.0, valid 0.409).

**Padding.** Adding a border (of zeros) around the input so a convolution's output keeps the input's size.

**Paired bootstrap.** Bootstrapping the difference between two models using the same resamples for both, so shared test-set difficulty cancels.

**Pairwise coupling.** Combining the 45 pairwise probabilities of a one-vs-one model into one probability vector over all classes that agrees best with them (Wu, Lin & Weng 2004).

**Percentile interval.** Bootstrap CI taken as the 2.5th and 97.5th percentiles of the resampled statistic.

**Pinned version.** An exact package version (`name==X.Y.Z`) in `requirements.txt`, so everyone installs the same code.

**Pipeline.** An sklearn object chaining preprocessing and a model, so preprocessing such as scaling is refit on the training rows of each fit and never sees held-out data.

**Platt scaling.** Turning an SVM's decision values into probabilities with a fitted sigmoid 1/(1 + exp(A f + B)); sklearn fits A and B on internal 5-fold cross-validation scores.

**Precision.** Of the clips predicted as a class, the share that truly belong to it: TP / (TP + FP).

**PredefinedSplit.** sklearn splitter with a fixed train/validation assignment (-1 = always train, 0 = validation fold).

**Preprocessing leakage.** Fitting a learned preprocessing step (scaler, PCA, feature selection, oversampling) on data that includes the evaluation rows, so their information shapes the model.

**PYTHONHASHSEED.** Environment variable fixing the randomisation of Python's string hashing; read at interpreter start.

**Random crop.** A training window taken at a random start position in the clip (any of 1,163 offsets for our 130-frame crops), so the model sees a different slice of each clip every epoch.

**RBF kernel.** Radial basis function (Gaussian) kernel, exp(-gamma ||x - x'||^2): 1 for identical points, near 0 for distant ones. Its decision function is a weighted sum of Gaussian bumps on the support vectors.

**Recall.** Of the clips truly in a class, the share the model found: TP / (TP + FN).

**Receptive field.** The input region that one unit of a network layer depends on; 46 mel bins x 46 frames (about 1.07 s) for our last conv block.

**Refit.** Retraining the selected hyperparameter setting after a search; we refit on train only (GridSearchCV refit=False) so valid stays held out.

**Reliability diagram.** A plot of accuracy against mean confidence per confidence bin; a calibrated model lies on the diagonal.

**ReLU.** Rectified linear unit, max(0, z): the non-linearity after each convolution; passes positive values, zeroes negative ones.

**Reported score.** The number a procedure itself produces as its estimate of performance (e.g. GridSearchCV's best_score_). In lesson 09 it is compared with the true valid score of the chosen setting.

**Reproducibility.** Another person obtaining the same results from the same code, data and settings.

**RMS.** Root-mean-square of the samples; a measure of average signal level (loudness), reported in dBFS as `rms_db`.

**Sampling distribution.** The distribution a statistic would have over many independent samples of the same size; its spread is the statistic's uncertainty.

**Sampling rate.** Number of samples stored per second of audio; GTZAN uses 22,050 Hz.

**Seed variance.** The spread of results when only the random seed changes (initial weights, crops, augmentation, batch order); 0.580 to 0.635 valid macro-F1 for our three CNN seeds.

**Seed.** The starting state of a pseudo-random number generator; the same seed produces the same sequence of "random" numbers.

**Selection bias (in model selection).** The optimism of a score that was itself used to choose among candidates: the best of many noisy scores is likely to be lucky. Also called the winner's curse.

**Selection score.** The score a tuning procedure maximises to choose hyperparameters. It only needs to rank settings in the right order; leakage that biases settings unequally changes that order.

**Slack variable.** xi_i >= 0 in the soft-margin SVM: how far training point i violates the margin. Their sum, multiplied by C, is added to the objective.

**Soft margin.** An SVM that allows margin violations at a cost C per unit of slack, so it works on overlapping classes. Small C tolerates violations (simpler boundary), large C punishes them (closer fit).

**Soft vote.** Combining window predictions by averaging their class probabilities, then taking the most probable class for the clip.

**Softmax.** Turns K scores into probabilities: exp(z_g) divided by the sum of exp(z_h) over all classes.

**Spearman rank correlation.** Pearson correlation of the ranks of two sets of values: 1 for the same order, 0 for unrelated, -1 for reversed. Used in lesson 09 to compare how selection scores and true scores rank the 16 SVM settings.

**SpecAugment.** Augmentation for spectrograms that blanks random bands of frequencies and random stretches of time (2 masks each here, up to 15 bins and 20 frames).

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

**Temperature scaling.** Dividing all logits by one learned number T before the softmax (T > 1 softens, T < 1 sharpens). It changes confidences but never which class has the highest score of a single window.

**Test lock.** `load_split` refuses the test part unless allow_test=True and the git tag pre-test-freeze exists; ensures the test set is used once.

**Training row.** One example the model is fit on; here one 3 s window, labelled with its clip's genre.

**Translation equivariance.** Shifting the input shifts a convolution's output by the same amount, because the same filter is used everywhere. Natural along time in a spectrogram, only partly along frequency.

**Unit of prediction.** The object a prediction and its score refer to; here one 30 s clip.

**Virtual environment (venv).** A self-contained Python installation for one project, isolating its packages from others.

**Weight decay.** Adding lambda times each weight to its gradient (an L2 penalty, lambda = 0.0001 here) so the weights stay small.

**Weight sharing.** Using the same filter weights at every position of the input, which makes a convolution's parameter count independent of the image size.

**Weighted-F1.** Per-class F1 averaged with weights equal to each class's share of true examples.

**Window.** A 3 s non-overlapping segment of a clip (10 per clip); the training row.

**Within-clip leakage.** Windows of one clip in both train and test; the model sees near-copies of test windows during training.

**XGBoost.** A gradient-boosted tree library (Chen & Guestrin 2016) using second-order (Newton) steps, L2-penalised leaf values and histogram-based split finding.

**Zero-crossing rate.** Fraction of consecutive samples whose sign changes; high for noisy or bright sounds.
