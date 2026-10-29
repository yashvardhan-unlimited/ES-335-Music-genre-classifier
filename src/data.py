"""GTZAN file listing and per-file audit (load check, length, md5, level, clipping)."""
import hashlib
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from src.utils import get_logger, load_config

log = get_logger(__name__)

CLIP_THRESHOLD = 0.999  # |x| at or above this counts as a clipped sample


def list_clips(audio_dir: str | Path) -> pd.DataFrame:
    """One row per .wav file: clip_id ('blues.00000'), genre, path (posix, as given)."""
    rows = [
        {"clip_id": p.stem, "genre": p.parent.name, "path": p.as_posix()}
        for p in sorted(Path(audio_dir).glob("*/*.wav"))
    ]
    return pd.DataFrame(rows, columns=["clip_id", "genre", "path"])


def audit_clip(path: str | Path, sr: int) -> dict:
    """Load one file and describe it. Never raises on unreadable audio.

    Returns load_ok, n_samples, duration_s, md5 (of the file bytes), peak (max |x|),
    clip_share (fraction of samples with |x| >= 0.999), rms_db (20 log10 RMS, dBFS),
    and error (empty string when load_ok).
    """
    with open(path, "rb") as f:
        md5 = hashlib.file_digest(f, "md5").hexdigest()
    out = {"load_ok": False, "n_samples": 0, "duration_s": np.nan, "md5": md5,
           "peak": np.nan, "clip_share": np.nan, "rms_db": np.nan, "error": ""}
    try:
        y, _ = librosa.load(path, sr=sr, mono=True)
    except Exception as e:  # corrupt files raise different errors per backend
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    if y.size == 0:
        out["error"] = "empty audio"
        return out
    a = np.abs(y)
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    out.update(
        load_ok=True,
        n_samples=int(y.size),
        duration_s=y.size / sr,
        peak=float(a.max()),
        clip_share=float(np.mean(a >= CLIP_THRESHOLD)),
        rms_db=20 * np.log10(max(rms, 1e-10)),
    )
    return out


def build_metadata(cfg: dict | None = None) -> pd.DataFrame:
    """Audit every file one at a time; write splits/metadata.csv and results/load_errors.csv."""
    cfg = cfg or load_config()
    clips = list_clips(cfg["paths"]["audio_dir"])
    audits = []
    for i, path in enumerate(clips["path"]):
        audits.append(audit_clip(path, cfg["audio"]["sr"]))
        if (i + 1) % 100 == 0:
            log.info("audited %d/%d", i + 1, len(clips))
    meta = pd.concat([clips, pd.DataFrame(audits)], axis=1)

    errors = meta.loc[~meta["load_ok"], ["clip_id", "genre", "path", "error"]]
    errors.to_csv(Path(cfg["paths"]["results"]) / "load_errors.csv", index=False)
    meta.drop(columns="error").to_csv(Path(cfg["paths"]["splits"]) / "metadata.csv", index=False)
    log.info("%d files, %d loaded, %d failed", len(meta), meta["load_ok"].sum(), len(errors))
    return meta


if __name__ == "__main__":
    build_metadata()
