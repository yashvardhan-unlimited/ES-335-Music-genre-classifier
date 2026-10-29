# Lessons (reading order)

1. [01_problem_framing.md](01_problem_framing.md): the question, unit of prediction vs training row, target, prediction-time information, metric, limits of the claim.
2. [02_reproducibility_setup.md](02_reproducibility_setup.md): repo layout, venv, pinned versions, config, seeds, Makefile, git as evidence.
3. [03_data_and_gtzan_faults.md](03_data_and_gtzan_faults.md): what GTZAN is, licence, digital audio basics, the file audit and its numbers, GTZAN's faults, reading the EDA figures.
4. [04_splits_and_leakage.md](04_splits_and_leakage.md): i.i.d., leakage, group splitting, the A -> B -> B' -> C ladder and its measured leakage, identical sizes, stratification, the test lock.
5. [05_audio_features.md](05_audio_features.md): sampling, framing and Hann windows, STFT, mel filterbank, dB, MFCCs, chroma, spectral features, HPSS, tempo, fixed vs learned transforms.
6. [06_evaluation_metrics_bootstrap.md](06_evaluation_metrics_bootstrap.md): confusion matrix, precision/recall/F1, macro vs micro vs weighted, clip vs window, soft vs majority vote, bootstrap, artist-cluster resampling, paired comparisons, limits.
7. [07_baselines_knn_logreg.md](07_baselines_knn_logreg.md): why floors and baselines, dummy classifiers, standardisation, kNN (distance, voting, bias-variance in k, curse of dimensionality, lazy learning), multinomial logistic regression (softmax, cross-entropy, L2 and C, linear boundaries), the C-valid results and the k plot.
8. [08_svm_xgboost.md](08_svm_xgboost.md): maximum margin, hinge loss and C, kernels and the kernel trick, RBF and gamma, reading the C x gamma heatmap, one-vs-one, Platt scaling, decision trees and Gini, gradient boosting (XGBoost), learning rate, depth, subsampling, early stopping, why trees need no scaling, the C-valid results.
9. [09_leakage_lab.md](09_leakage_lab.md): measured leakage: scaler fit on train+valid, KFold over windows (0.973 reported vs 0.433 true), selection by window accuracy; reported vs true scores, setting-dependent bias, Spearman ranking, why an inner scaler cancels pre-scaling, how to spot leakage in notebooks.

Terms: [glossary.md](glossary.md)
