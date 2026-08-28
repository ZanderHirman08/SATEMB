"""Wrapper around the Clay v1.5 foundation model for encoding Sentinel-2 chips.

Clay is metadata-conditioned: alongside pixels it wants per-band center
wavelengths (which bands, physically, is this?) and a timestamp/location
(when and where on Earth is this?). That conditioning is itself one of the
more interesting things to explain in the write-up -- the embedding isn't
just "what does this patch look like", it's "what does this patch look like,
*given* it's these specific bands at this specific place and time".

Reference docs (check these if the encoder call signature below has drifted
since this was written -- Clay is an actively developed research repo):
  https://clay-foundation.github.io/model/getting-started/basic_use.html
  https://huggingface.co/made-with-clay/Clay
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch
import yaml

CHECKPOINT_URL = (
    "https://huggingface.co/made-with-clay/Clay/resolve/main/v1.5/clay-v1.5.ckpt"
)
PLATFORM = "sentinel-2-l2a"

# Order must match src/stac_utils.py:S2_BANDS.
BAND_NAMES = ["blue", "green", "red", "rededge1", "rededge2", "rededge3",
              "nir", "nir08", "swir16", "swir22"]


def download_checkpoint(dest_dir: str = "checkpoints") -> Path:
    """Download the Clay v1.5 checkpoint from HuggingFace if not already cached."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    ckpt_path = dest / "clay-v1.5.ckpt"
    if not ckpt_path.exists():
        import urllib.request

        print(f"Downloading Clay v1.5 checkpoint (~1.3GB) to {ckpt_path} ...")
        urllib.request.urlretrieve(CHECKPOINT_URL, ckpt_path)
    return ckpt_path


def _find_metadata_yaml() -> Path:
    """Locate metadata.yaml inside the installed claymodel package.

    We search rather than hardcode the path since the package's internal
    layout has moved between versions of the repo.
    """
    spec = importlib.util.find_spec("claymodel")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError(
            "claymodel is not installed. Run: "
            "pip install git+https://github.com/Clay-foundation/model.git@main"
        )
    pkg_root = Path(list(spec.submodule_search_locations)[0])
    candidates = list(pkg_root.rglob("metadata.yaml"))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find metadata.yaml under {pkg_root}. "
            "Check the current claymodel package layout at "
            "https://github.com/Clay-foundation/model"
        )
    return candidates[0]


def load_model(ckpt_path: Path, device: str = "cuda"):
    """Load the Clay v1.5 module from a local checkpoint and set eval mode."""
    from claymodel.module import ClayMAEModule

    model = ClayMAEModule.load_from_checkpoint(str(ckpt_path))
    model.eval()
    model.to(device)
    return model


def load_band_stats(platform: str = PLATFORM):
    """Read per-band wavelength (nm) and normalization mean/std for `platform`
    out of Clay's own metadata.yaml, so we don't hand-transcribe numbers that
    could silently drift out of sync with a newer checkpoint.
    """
    meta_path = _find_metadata_yaml()
    with open(meta_path) as f:
        meta = yaml.safe_load(f)

    platform_meta = meta[platform]
    wavelengths = [platform_meta["bands"]["wavelength"][b] for b in BAND_NAMES]
    means = [platform_meta["bands"]["mean"][b] for b in BAND_NAMES]
    stds = [platform_meta["bands"]["std"][b] for b in BAND_NAMES]
    return (
        torch.tensor(wavelengths, dtype=torch.float32),
        torch.tensor(means, dtype=torch.float32),
        torch.tensor(stds, dtype=torch.float32),
    )


def normalize_chips(chips: np.ndarray, means: torch.Tensor, stds: torch.Tensor) -> torch.Tensor:
    """chips: (N, C, H, W) raw reflectance -> normalized torch tensor."""
    x = torch.from_numpy(chips).float()
    return (x - means[None, :, None, None]) / stds[None, :, None, None]


def make_time_latlon_tensors(dates, lats, lons) -> tuple[torch.Tensor, torch.Tensor]:
    """Build Clay's (batch, 4) time and (batch, 4) latlon conditioning tensors.

    Clay encodes time as sin/cos of (week-of-year, hour-of-day) and location
    as sin/cos of (lat, lon) -- see basic_use.html for the exact encoding if
    this needs adjusting. `dates` is a sequence of pandas.Timestamp-like.
    """
    time_feats, latlon_feats = [], []
    for date, lat, lon in zip(dates, lats, lons):
        week = date.isocalendar().week / 52.0 * 2 * np.pi
        hour = getattr(date, "hour", 12) / 24.0 * 2 * np.pi
        time_feats.append([np.sin(week), np.cos(week), np.sin(hour), np.cos(hour)])

        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        latlon_feats.append([np.sin(lat_rad), np.cos(lat_rad), np.sin(lon_rad), np.cos(lon_rad)])

    return (
        torch.tensor(time_feats, dtype=torch.float32),
        torch.tensor(latlon_feats, dtype=torch.float32),
    )


@torch.no_grad()
def encode_batch(model, chips_norm, time_feats, latlon_feats, wavelengths, device: str = "cuda"):
    """Run one batch through Clay's encoder and return (batch, embed_dim) embeddings.

    Clay's encoder returns patch tokens plus a class token; we take the mean
    over patch tokens (excluding the class token) as the whole-chip embedding
    -- a standard choice for downstream similarity/clustering use, though the
    class token alone is a reasonable alternative worth comparing in notebook 03.
    """
    datacube = {
        "pixels": chips_norm.to(device),
        "time": time_feats.to(device),
        "latlon": latlon_feats.to(device),
        "waves": wavelengths.to(device),
        "gsd": torch.tensor(10.0, device=device),
    }
    embeddings, *_ = model.model.encoder(datacube)
    patch_tokens = embeddings[:, 1:, :]  # drop class token
    return patch_tokens.mean(dim=1).cpu().numpy()
