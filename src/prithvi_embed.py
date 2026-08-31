"""Wrapper around IBM/NASA's Prithvi-EO-2.0 foundation model, for the
cross-model replication check (does the Front Range "terrain axis" finding
hold on a second, independently-trained foundation model, or is it specific
to Clay?).

The real model object (`terratorch.models.backbones.prithvi_mae.PrithviViT`)
exposes `img_size`, `in_chans`, `embed_dim`, `num_frames`, and `pretrained_bands`
as plain attributes -- discovered by loading the model in Colab and running
`dir(model)`, not assumed from documentation, after an earlier guess at a
`pretrained_cfg`-style config object turned out wrong. `pretrained_bands` is a
list of `HLSBands` enum members (semantic names: BLUE, RED, ...), not raw
Sentinel-2 band codes, so BAND_TO_S2 below maps them to the STAC asset codes
this project's stac_utils.py already uses.

Normalization mean/std are NOT attributes on the model object itself (checked
directly -- not present in `dir(model)`), so unlike everything else in this
module they can't be read off the live model at runtime. They're hardcoded
below as PRITHVI_V2_MEAN/STD, sourced from terratorch's own registration code
(terratorch/models/backbones/prithvi_vit.py, PRITHVI_V2_MEAN/PRITHVI_V2_STD)
rather than from a web summary -- these numbers happened to match an earlier,
otherwise-unreliable web lookup exactly, which is corroborating, not proof;
if a chip's post-normalization values look implausible, re-verify against
that file directly rather than trusting this comment.

Reference docs (check these first if something below has drifted):
  https://github.com/IBM/terratorch/blob/main/terratorch/models/backbones/prithvi_vit.py
  https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M
  https://github.com/NASA-IMPACT/Prithvi-EO-2.0
"""

from __future__ import annotations

import numpy as np
import torch

MODEL_ID = "prithvi_eo_v2_300"

# Sourced from terratorch/models/backbones/prithvi_vit.py: PRITHVI_V2_MEAN / STD.
PRITHVI_V2_MEAN = [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0]
PRITHVI_V2_STD = [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0]

# HLSBands enum member name -> Sentinel-2 L2A STAC asset code (Planetary
# Computer). Order doesn't matter here; discover_config() reorders this to
# match model.pretrained_bands's actual order.
HLS_BAND_TO_S2 = {
    "BLUE": "B02",
    "GREEN": "B03",
    "RED": "B04",
    "NIR_NARROW": "B8A",
    "SWIR_1": "B11",
    "SWIR_2": "B12",
}


def load_model(device: str = "cuda"):
    """Build the pretrained Prithvi-EO-2.0-300M backbone via terratorch."""
    try:
        from terratorch.registry import BACKBONE_REGISTRY
    except ImportError as e:
        raise ImportError(
            "Could not import BACKBONE_REGISTRY from terratorch.registry -- "
            "run `!pip install -q terratorch` first, or the package layout "
            "has changed. Run `import terratorch; print(dir(terratorch))` to "
            "find the real import path, then update src/prithvi_embed.py."
        ) from e

    model = BACKBONE_REGISTRY.build(MODEL_ID, pretrained=True)
    model.eval()
    model.to(device)
    return model


def discover_config(model) -> dict:
    """Read bands, image size, and channel/frame counts off the live model
    (all real, confirmed-by-dir(model) attributes on PrithviViT), and pair
    them with the sourced-not-guessed mean/std constants above.
    """
    required_attrs = ["img_size", "in_chans", "embed_dim", "num_frames", "pretrained_bands"]
    missing_attrs = [a for a in required_attrs if not hasattr(model, a)]
    if missing_attrs:
        raise AttributeError(
            f"model is missing expected attribute(s) {missing_attrs} -- the backbone class may "
            "have changed. Run `print([a for a in dir(model) if not a.startswith('_')])` to see "
            "the real attribute names, then update discover_config() in src/prithvi_embed.py."
        )

    raw_bands = list(model.pretrained_bands)
    band_names = [getattr(b, "name", str(b)) for b in raw_bands]
    unmapped = [n for n in band_names if n not in HLS_BAND_TO_S2]
    if unmapped:
        raise KeyError(
            f"model.pretrained_bands includes {unmapped}, not in HLS_BAND_TO_S2 "
            f"{list(HLS_BAND_TO_S2)}. Print raw_bands directly to see the real HLSBands members, "
            "then add the missing mapping(s) to HLS_BAND_TO_S2 in src/prithvi_embed.py."
        )
    s2_bands = [HLS_BAND_TO_S2[n] for n in band_names]

    if len(PRITHVI_V2_MEAN) != len(s2_bands):
        raise ValueError(
            f"PRITHVI_V2_MEAN/STD have {len(PRITHVI_V2_MEAN)} values but the model reports "
            f"{len(s2_bands)} bands ({band_names}) -- these no longer line up. Re-check "
            "terratorch/models/backbones/prithvi_vit.py for the current PRITHVI_V2_MEAN/STD "
            "and PRETRAINED_BANDS ordering, then update this module."
        )

    cfg = {
        "bands": s2_bands,
        "hls_band_names": band_names,
        "mean": PRITHVI_V2_MEAN,
        "std": PRITHVI_V2_STD,
        "img_size": int(model.img_size),
        "in_chans": int(model.in_chans),
        "embed_dim": int(model.embed_dim),
        "num_frames": int(model.num_frames),
    }
    print(f"model.pretrained_bands (raw): {raw_bands}")
    print(f"Mapped to Sentinel-2 STAC bands: {s2_bands}")
    print(f"img_size={cfg['img_size']}, in_chans={cfg['in_chans']}, "
          f"embed_dim={cfg['embed_dim']}, num_frames={cfg['num_frames']}")
    print(f"mean={cfg['mean']}")
    print(f"std={cfg['std']}")
    return cfg


def normalize_chips(chips: np.ndarray, mean: list[float], std: list[float]) -> torch.Tensor:
    """chips: (N, C, H, W) raw reflectance -> normalized torch tensor."""
    x = torch.from_numpy(chips).float()
    mean_t = torch.tensor(mean, dtype=torch.float32)
    std_t = torch.tensor(std, dtype=torch.float32)
    return (x - mean_t[None, :, None, None]) / std_t[None, :, None, None]


@torch.no_grad()
def encode_batch(model, chips_norm: torch.Tensor, num_frames: int = 1, device: str = "cuda") -> np.ndarray:
    """Run one batch through Prithvi's encoder and return (batch, embed_dim).

    Prithvi-EO-2.0 uses 3D (time, height, width) patch embeddings, trained
    with a specific temporal window (model.num_frames, from discover_config).
    For single-date chips, the same frame is repeated num_frames times along
    the temporal axis -- the standard way to feed single-date imagery into a
    model trained on a fixed multi-temporal window. If the real forward
    signature doesn't match this, the exception below shows the actual shape
    mismatch -- fix by adjusting this call, not by guessing again.
    """
    x = chips_norm.to(device)
    x = x.unsqueeze(2).repeat(1, 1, num_frames, 1, 1)  # (B,C,H,W) -> (B,C,T,H,W)

    try:
        output = model(x)
    except Exception as e:
        raise RuntimeError(
            f"model(x) failed with input shape {tuple(x.shape)}: {e}. "
            "Prithvi's forward signature may not match the (B, C, T, H, W) "
            "assumption here -- try model(chips_norm.to(device)) without the "
            "temporal dim, or check the model's own forward() docstring via "
            "`import inspect; print(inspect.signature(model.forward))`, "
            "then update encode_batch() in src/prithvi_embed.py."
        ) from e

    if isinstance(output, (tuple, list)):
        if len(output) > 1:
            print(
                f"Note: model(x) returned {len(output)} tensors (a multi-scale feature list is "
                "common for terratorch backbones) -- using the LAST one (deepest/most semantic "
                "stage) as the embedding. If results look wrong, try output[0] or another index "
                "instead in encode_batch()."
            )
        tokens = output[-1]
    else:
        tokens = output
    if tokens.ndim == 3:
        # (B, n_tokens, dim) -- drop a leading class/cls token if present by
        # checking whether excluding it changes the token count expected
        # from img_size/patch_size; safest default is to mean-pool everything.
        pooled = tokens.mean(dim=1)
    elif tokens.ndim == 2:
        pooled = tokens  # already pooled by the backbone
    else:
        raise ValueError(
            f"Unexpected output shape {tuple(tokens.shape)} from Prithvi encoder -- "
            "inspect it directly and update encode_batch()'s pooling logic."
        )
    return pooled.cpu().numpy()
