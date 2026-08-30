"""Wrapper around IBM/NASA's Prithvi-EO-2.0 foundation model, for the
cross-model replication check (does the Front Range "terrain axis" finding
hold on a second, independently-trained foundation model, or is it specific
to Clay?).

Unlike src/clay_embed.py, this module deliberately does NOT hardcode Prithvi's
expected bands, normalization mean/std, or input image size. Two independent
web lookups while building this returned inconsistent details for those
values (one even contradicted itself between a model card's prose and its own
config), which isn't trustworthy enough to bake into 725 chips' worth of
embeddings without a mistake going unnoticed. Instead, every one of those
values is read directly off the loaded model object at runtime -- the
checkpoint itself is ground truth, docs can drift or be wrong. If an
attribute name below has changed since this was written, the error messages
tell you exactly what to inspect and where to fix it, same philosophy as
clay_embed.py.

Reference docs (check these first if something below has drifted):
  https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M
  https://github.com/NASA-IMPACT/Prithvi-EO-2.0
  https://github.com/IBM/terratorch
"""

from __future__ import annotations

import numpy as np
import torch

MODEL_ID = "prithvi_eo_v2_300"


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
    """Read bands, normalization stats, and image size off the live model,
    rather than trusting hardcoded values from documentation.

    Tries the common timm-style `pretrained_cfg` attribute first (terratorch
    backbones are frequently timm-wrapped), then a couple of other likely
    spots. Prints whatever it finds so a human can sanity-check it before
    anything downstream trusts it.
    """
    import dataclasses

    def to_dict(obj):
        if isinstance(obj, dict):
            return obj
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        if hasattr(obj, "items"):
            return dict(obj)
        if hasattr(obj, "__dict__"):
            return vars(obj)
        return None

    candidates = ["pretrained_cfg", "cfg", "model_args", "config"]
    cfg = None
    for name in candidates:
        obj = getattr(model, name, None)
        if obj is None:
            continue
        cfg = to_dict(obj)
        if cfg is not None:
            print(f"Found config at model.{name}: {cfg}")
            break
        print(f"model.{name} exists ({type(obj)}) but couldn't be converted to a dict -- skipping.")

    if cfg is None:
        raise AttributeError(
            f"Could not find a config dict on the loaded model under any of "
            f"{candidates}. Run `print([a for a in dir(model) if not "
            "a.startswith('_')])` to see the real attribute names, then "
            "update discover_config() in src/prithvi_embed.py accordingly."
        )

    required = ["bands", "mean", "std", "img_size"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(
            f"Config found at model config, but missing keys {missing}. "
            f"Full config: {cfg}. Some backbones nest these under a "
            "different key (e.g. cfg['pretrained_cfg']) -- inspect the "
            "printed dict above and update discover_config()."
        )

    print(f"\nUsing bands={cfg['bands']}, img_size={cfg['img_size']}")
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
def encode_batch(model, chips_norm: torch.Tensor, device: str = "cuda") -> np.ndarray:
    """Run one batch through Prithvi's encoder and return (batch, embed_dim).

    Prithvi-EO-2.0 uses 3D (time, height, width) patch embeddings even for
    single-date input, so this adds a size-1 temporal dimension:
    (B, C, H, W) -> (B, C, 1, H, W). If the real forward signature doesn't
    match this, the exception below will show the actual shape mismatch --
    fix by adjusting the unsqueeze/forward call here, not by guessing again.
    """
    x = chips_norm.to(device)
    x = x.unsqueeze(2)  # (B, C, H, W) -> (B, C, T=1, H, W)

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
