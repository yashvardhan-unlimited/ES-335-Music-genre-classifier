"""Extract handcrafted window features and full-clip log-mels for every usable clip.

Run from repo root: python scripts/extract_features.py
Writes features/handcrafted.npy, features/index.csv, features/feature_names.json,
features/mel/<clip_id>.npy.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import N_WINDOWS, clip_features, feature_names  # noqa: E402
from src.utils import get_logger, load_config  # noqa: E402

log = get_logger("extract_features")


def process(clip_id: str, path: str, mel_dir: Path) -> tuple[np.ndarray, bool]:
    """Features of one clip; its log-mel is written straight to disk."""
    warnings.filterwarnings("ignore")
    feats, mel, padded = clip_features(path)
    np.save(mel_dir / f"{clip_id}.npy", mel)
    return feats, padded


def pipeline_figure(path: str, out: Path) -> None:
    """Waveform -> spectrogram -> mel-spectrogram -> MFCCs for the first 3 s window of one clip."""
    import librosa
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.features import CFG, SR, STFT, load_clip, windows

    w = windows(load_clip(path)[0])[0]
    spec = librosa.amplitude_to_db(np.abs(librosa.stft(w, **STFT)), ref=1.0)
    mel = librosa.power_to_db(librosa.feature.melspectrogram(
        y=w, sr=SR, n_mels=CFG["mel"]["n_mels"], **STFT), ref=CFG["mel"]["ref"])
    mfcc = librosa.feature.mfcc(y=w, sr=SR, n_mfcc=CFG["mfcc"]["n_mfcc"], n_mels=CFG["mel"]["n_mels"], **STFT)
    hop = STFT["hop_length"]

    fig, ax = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    librosa.display.waveshow(w, sr=SR, ax=ax[0])
    ax[0].set(title=f"1. Waveform ({Path(path).stem}, window 0: {w.size:,} samples)", ylabel="amplitude")
    im = librosa.display.specshow(spec, sr=SR, hop_length=hop, x_axis="time", y_axis="hz", ax=ax[1])
    fig.colorbar(im, ax=ax[1], format="%+.0f dB")
    ax[1].set(title=f"2. STFT magnitude, Hann window n_fft={STFT['n_fft']}, hop={hop} "
                    f"({spec.shape[0]} bins x {spec.shape[1]} frames)")
    im = librosa.display.specshow(mel, sr=SR, hop_length=hop, x_axis="time", y_axis="mel", ax=ax[2])
    fig.colorbar(im, ax=ax[2], format="%+.0f dB")
    ax[2].set(title=f"3. Log-mel spectrogram ({mel.shape[0]} mel bands, power_to_db ref=1.0)")
    im = librosa.display.specshow(mfcc, sr=SR, hop_length=hop, x_axis="time", ax=ax[3])
    fig.colorbar(im, ax=ax[3])
    ax[3].set(title=f"4. MFCCs (DCT of log-mel, first {mfcc.shape[0]} coefficients)", ylabel="coefficient")
    ax[3].set_yticks(np.arange(0, 20, 4) + 0.5, np.arange(1, 21, 4))
    plt.tight_layout()
    plt.savefig(out, dpi=110)
    plt.close()


def main() -> None:
    cfg = load_config()
    out = Path(cfg["paths"]["features"])
    mel_dir = out / "mel"
    mel_dir.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(Path(cfg["paths"]["splits"]) / "metadata.csv")
    clips = meta[meta["load_ok"]].reset_index(drop=True)

    t0 = time.perf_counter()
    jobs = (delayed(process)(c, p, mel_dir) for c, p in zip(clips["clip_id"], clips["path"]))
    results = Parallel(n_jobs=-1, return_as="generator")(jobs)
    results = list(tqdm(results, total=len(clips), desc="clips"))

    X = np.concatenate([f for f, _ in results]).astype(np.float32)
    index = pd.DataFrame({
        "clip_id": np.repeat(clips["clip_id"], N_WINDOWS).to_numpy(),
        "window_idx": np.tile(np.arange(N_WINDOWS), len(clips)),
        "genre": np.repeat(clips["genre"], N_WINDOWS).to_numpy(),
        "clip_padded": np.repeat([p for _, p in results], N_WINDOWS),
    })
    np.save(out / "handcrafted.npy", X)
    index.to_csv(out / "index.csv", index=False)
    (out / "feature_names.json").write_text(json.dumps(feature_names(), indent=1))

    log.info("handcrafted %s, %d mels, %d padded clips, non-finite values: %d",
             X.shape, len(list(mel_dir.glob("*.npy"))), int(index.groupby("clip_id")["clip_padded"].first().sum()),
             int((~np.isfinite(X)).sum()))
    log.info("total runtime %.1f s", time.perf_counter() - t0)
    pipeline_figure(clips["path"].iloc[0], Path(cfg["paths"]["results"]) / "figures" / "feature_pipeline.png")


if __name__ == "__main__":
    main()
