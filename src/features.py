"""Fixed per-clip audio transforms: windowing, 57 handcrafted features, log-mel.

Nothing here uses statistics across clips, so nothing here can leak between splits.
"""
import numpy as np
import librosa

from src.utils import load_config

CFG = load_config()
SR = CFG["audio"]["sr"]
N_CLIP = CFG["audio"]["clip_seconds"] * SR      # 661,500
N_WIN = CFG["audio"]["window_seconds"] * SR     # 66,150
N_WINDOWS = CFG["audio"]["n_windows"]           # 10
STFT = dict(n_fft=CFG["mel"]["n_fft"], hop_length=CFG["mel"]["hop_length"])
assert N_WIN * N_WINDOWS == N_CLIP


def load_clip(path: str, sr: int = SR) -> tuple[np.ndarray, bool]:
    """Load mono float32 and zero-pad or trim to exactly 30 s. Returns (y, padded)."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    padded = y.size < N_CLIP
    y = np.pad(y, (0, N_CLIP - y.size)) if padded else y[:N_CLIP]
    return y.astype(np.float32), padded


def windows(y: np.ndarray, n_windows: int = N_WINDOWS) -> np.ndarray:
    """(661,500,) -> (n_windows, 661,500 / n_windows) non-overlapping windows (a view, no copy).

    Default 10 windows of 3 s; the window-length ablation uses 30, 6 and 3 (1, 5, 10 s).
    """
    assert y.size == N_CLIP, f"expected {N_CLIP} samples, got {y.size}"
    assert N_CLIP % n_windows == 0, f"{n_windows} windows do not divide the clip evenly"
    return y.reshape(n_windows, N_CLIP // n_windows)


def clip_tempo(y: np.ndarray, sr: int = SR) -> float:
    """Tempo (BPM) of the full clip from librosa's beat tracker."""
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=STFT["hop_length"])
    return float(np.atleast_1d(tempo)[0])


def feature_names() -> list[str]:
    """Names of the 57 handcrafted features, in column order."""
    names = [f"mfcc{i}_{s}" for i in range(1, 21) for s in ("mean", "var")]
    for base in ("chroma", "centroid", "bandwidth", "rolloff", "zcr", "rms", "harm_rms", "perc_rms"):
        names += [f"{base}_mean", f"{base}_var"]
    return names + ["tempo"]


def feature_group(name: str, groups: dict[str, list[str]] = CFG["analysis"]["feature_groups"]) -> str:
    """Group of one feature name: its stem (name minus _mean/_var, mfcc numbers dropped) looked up
    in config analysis.feature_groups, e.g. 'mfcc7_var' -> 'mfcc', 'harm_rms_mean' -> 'hpss'."""
    stem = name.removesuffix("_mean").removesuffix("_var").rstrip("0123456789")
    matches = [g for g, stems in groups.items() if stem in stems]
    if len(matches) != 1:
        raise ValueError(f"feature {name!r} (stem {stem!r}) is in {len(matches)} groups")
    return matches[0]


def _mv(x: np.ndarray) -> list[float]:
    return [float(np.mean(x)), float(np.var(x))]


def handcrafted_features(window: np.ndarray, sr: int, clip_tempo: float,
                         harmonic: np.ndarray, percussive: np.ndarray) -> np.ndarray:
    """57 features of one window; harmonic/percussive are the matching slices of the
    clip-level HPSS, clip_tempo is the clip-level tempo (both computed once per clip)."""
    S = np.abs(librosa.stft(window, **STFT))
    mfcc = librosa.feature.mfcc(y=window, sr=sr, n_mfcc=CFG["mfcc"]["n_mfcc"],
                                n_mels=CFG["mel"]["n_mels"], **STFT)
    f = []
    for row in mfcc:
        f += _mv(row)
    f += _mv(librosa.feature.chroma_stft(S=S**2, sr=sr, **STFT))
    f += _mv(librosa.feature.spectral_centroid(S=S, sr=sr, **STFT))
    f += _mv(librosa.feature.spectral_bandwidth(S=S, sr=sr, **STFT))
    f += _mv(librosa.feature.spectral_rolloff(S=S, sr=sr, **STFT))
    frames = dict(frame_length=STFT["n_fft"], hop_length=STFT["hop_length"])
    f += _mv(librosa.feature.zero_crossing_rate(window, **frames))
    f += _mv(librosa.feature.rms(y=window, **frames))
    f += _mv(librosa.feature.rms(y=harmonic, **frames))
    f += _mv(librosa.feature.rms(y=percussive, **frames))
    f.append(clip_tempo)
    return np.asarray(f, dtype=np.float32)


def clip_mel(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """Full-clip log-mel (n_mels x frames) in dB with a fixed reference, float16."""
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=CFG["mel"]["n_mels"], **STFT)
    return librosa.power_to_db(S, ref=CFG["mel"]["ref"]).astype(np.float16)


def clip_handcrafted(y: np.ndarray, window_counts: list[int]) -> dict[int, np.ndarray]:
    """57 features per window for each window count (e.g. 10 -> 10 x 57), sharing one
    clip-level HPSS and tempo between all of them."""
    harmonic, percussive = librosa.effects.hpss(y)
    tempo = clip_tempo(y)
    return {n: np.stack([handcrafted_features(w, SR, tempo, h, p)
                         for w, h, p in zip(windows(y, n), windows(harmonic, n), windows(percussive, n))])
            for n in window_counts}


def clip_features(path: str) -> tuple[np.ndarray, np.ndarray, bool]:
    """All features of one clip: (10 x 57 handcrafted, log-mel, padded)."""
    y, padded = load_clip(path)
    return clip_handcrafted(y, [N_WINDOWS])[N_WINDOWS], clip_mel(y), padded
