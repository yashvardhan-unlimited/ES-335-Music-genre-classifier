# Glossary

**Artist-cluster bootstrap.** A bootstrap that resamples whole artists (with all their clips) instead of single clips, because clips by one artist are correlated. Gives honest confidence intervals.

**Artist-disjoint split.** A split in which no artist has clips in more than one part (train, valid, test). Prevents the model from scoring by recognising artists.

**Bootstrap.** Estimating the uncertainty of a statistic by recomputing it on many datasets drawn with replacement from the observed one; the spread of those values gives a confidence interval.

**Chroma.** A 12-value per-frame vector of energy in each pitch class (C to B), with octaves folded together; describes harmony.

**Clip.** One ~30 s GTZAN audio excerpt with one genre label; the unit of prediction in this project.

**Clipping.** Distortion when a signal exceeds the largest representable value and its peaks are cut flat. We count samples with |x| >= 0.999 (`clip_share`).

**Confidence interval (95%).** A range produced by a procedure that, over repeated samples, contains the true value 95% of the time. Here estimated with the bootstrap percentile method.

**Config file.** `config.yaml`: the single place where all constants (seed, audio settings, sizes, paths) are defined.

**Data leakage.** Information available during training or evaluation that would not be available at real prediction time, making scores look better than real performance.

**dBFS.** Decibels relative to full scale: 20 log10(amplitude), so 0 dBFS is the loudest possible digital level and real levels are at or below 0.

**DCT.** Discrete cosine transform; rewrites a sequence as a sum of cosines. Applied to a log-mel frame it gives MFCCs.

**Exact repetition.** The same audio file present twice in a dataset; found here as identical md5 hashes (14 pairs in GTZAN).

**Fault-filtered split.** The published GTZAN split (Kereliuk, Sturm & Larsen 2015) that removes duplicated, mislabelled and distorted clips and keeps artists disjoint: 443/197/290 clips.

**Fixed transform.** A feature computation that uses only the clip itself and preset constants; it cannot leak between splits.

**Frame.** A short slice of audio (here 2048 samples, 93 ms) analysed as one spectrum; frames advance by the hop length (512 samples).

**GTZAN.** Genre dataset of Tzanetakis & Cook (2002): 1000 clips of 30 s, 10 genres, 22,050 Hz mono, collected 2000-2001 from CDs, radio and microphone recordings.

**Hann window.** A taper that goes smoothly to zero at both ends of a frame, reducing spectral leakage in the STFT.

**HPSS.** Harmonic-percussive source separation: median filtering of the spectrogram along time (harmonic) and frequency (percussive) to split a signal into two parts.

**Learned statistic.** A quantity computed from many clips (mean, std, PCA axes, selected features); must be fitted on training data only.

**Log-mel spectrogram.** A time-frequency picture of audio: short-time Fourier magnitudes pooled into mel-spaced bands and converted to dB.

**Macro-F1.** The unweighted mean of the per-class F1 scores, so each genre counts equally regardless of how many clips it has.

**md5.** A 128-bit hash of a file's bytes; identical files have identical md5, so it detects exact duplicates and fingerprints a dataset copy.

**Mel scale.** A frequency scale spaced roughly like human pitch perception: fine at low frequencies, coarse at high ones.

**MFCC.** Mel-frequency cepstral coefficient: DCT of a log-mel frame; low coefficients summarise the spectral envelope (timbre).

**Mislabeling.** A clip whose genre label is wrong or doubtful; it caps achievable accuracy and teaches the model wrong mappings.

**Nyquist frequency.** Half the sampling rate, the highest frequency a sampled signal can represent (11,025 Hz for GTZAN).

**Onset strength.** A per-frame measure of how much spectral energy increases, i.e. how likely a note or hit starts; the input to tempo estimation.

**Pinned version.** An exact package version (`name==X.Y.Z`) in `requirements.txt`, so everyone installs the same code.

**PYTHONHASHSEED.** Environment variable fixing the randomisation of Python's string hashing; read at interpreter start.

**Reproducibility.** Another person obtaining the same results from the same code, data and settings.

**RMS.** Root-mean-square of the samples; a measure of average signal level (loudness), reported in dBFS as `rms_db`.

**Sampling rate.** Number of samples stored per second of audio; GTZAN uses 22,050 Hz.

**Seed.** The starting state of a pseudo-random number generator; the same seed produces the same sequence of "random" numbers.

**Soft vote.** Combining window predictions by averaging their class probabilities, then taking the most probable class for the clip.

**Spectral bandwidth.** The magnitude-weighted spread of frequencies around the spectral centroid.

**Spectral centroid.** The magnitude-weighted mean frequency of a spectrum; perceived as brightness.

**Spectral roll-off.** The frequency below which a fixed share (85%) of a frame's spectral energy lies.

**STFT.** Short-time Fourier transform: Fourier transforms of overlapping windowed frames, giving a time-frequency representation.

**Training row.** One example the model is fit on; here one 3 s window, labelled with its clip's genre.

**Unit of prediction.** The object a prediction and its score refer to; here one 30 s clip.

**Virtual environment (venv).** A self-contained Python installation for one project, isolating its packages from others.

**Window.** A 3 s non-overlapping segment of a clip (10 per clip); the training row.

**Zero-crossing rate.** Fraction of consecutive samples whose sign changes; high for noisy or bright sounds.
