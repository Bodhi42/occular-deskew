"""occular-deskew v0.5.0 — document deskew.

Pipeline (default: fht_first — FHT levels to the axes before OriNet votes):
  1. Fine-angle skew (FHT / OpenCV Hough) → ±45°, levels the doc to the nearest axis
  2. rotate + adaptive crop
  3. Orientation (OriNet, ONNX/GPU, C4-TTA) → 0° / 90° / 180° / 270° on an axis-aligned doc
  4. total = (skew + orientation) % 360

Measured on 2000 real rotated scans, fht_first vs legacy orient_first:
  accuracy ≤5°: 93.65% vs 85.75%;  errors >5°: 127 vs 285 (180° flips 14 vs 105).
Set OCCULAR_PIPELINE_ORDER=orient_first for the legacy order.

Default backends need neither torch nor downloaded transformer weights:
  - orientation: OriNet   (occular_deskew/weights/orientation_orinet_fp32.onnx, ~4.8 MB)
  - fine angle:  FHT       (classical OpenCV, no weights)

Legacy backends (opt-in, require torch + transformers + peft):
  - OCCULAR_ORIENT_BACKEND=siglip  → SigLIP-Base + LoRA
  - OCCULAR_SKEW_BACKEND=dinov2    → DINOv2-Small + LoRA
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

_WEIGHTS_DIR = Path(__file__).parent / "weights"
_SKEW_CKPT = _WEIGHTS_DIR / "dinov2_skew_v0.4.0.pth"
_ORIENT_CKPT = _WEIGHTS_DIR / "siglip_orient_v0.4.0.pth"

_SKEW_BACKBONE = "facebook/dinov2-small"
_ORIENT_BACKBONE = "google/siglip-base-patch16-224"
_ORIENT_ANGLES = [0, 90, 180, 270]
_IMG_SIZE = 224

# Backends. Defaults are weight-light and torch-free.
ORIENT_BACKEND = os.environ.get("OCCULAR_ORIENT_BACKEND", "orinet")   # "orinet" | "siglip" | "paddle"
# Fine-angle detector: "fht" (default, OpenCV Hough+weighted median), "projection"
# (projection-profile, sharpest on text), "deskew" (pip `deskew` library, needs scikit-image).
# "dinov2" is a retired legacy backend (needs torch).
SKEW_BACKEND = os.environ.get("OCCULAR_SKEW_BACKEND", "fht")          # fht | projection | deskew | dinov2
# Pipeline order:
#   "orient_first" — orinet on the raw scan, then fine-angle (legacy order)
#   "fht_first"    — FHT levels the doc to the axes first, then orinet votes on an
#                    axis-aligned doc (its training distribution), then a fine-angle re-check
PIPELINE_ORDER = os.environ.get("OCCULAR_PIPELINE_ORDER", "fht_first")
# Optional final fine-angle refinement on the already-levelled page (safe small range).
# "projection" adds a ±6° projection-profile pass after orientation; "" disables.
SKEW_REFINE = os.environ.get("OCCULAR_SKEW_REFINE", "")

_orinet = None
_paddle = None
_skew_model = None
_skew_tfm = None
_orient_model = None
_orient_tfm = None
_device = None
_ready = False


# ============== torch-based model builders (legacy, lazy) ==============


def _build_skew():
    import torch
    import torch.nn as nn
    from transformers import AutoModel
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

    backbone = AutoModel.from_pretrained(_SKEW_BACKBONE)
    backbone = get_peft_model(backbone, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
        target_modules=["query", "key", "value", "dense"],
    ))
    head = nn.Sequential(
        nn.LayerNorm(384),
        nn.Linear(384, 256), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(256, 1),
    )
    ck = torch.load(str(_SKEW_CKPT), map_location="cpu", weights_only=True)
    set_peft_model_state_dict(backbone, ck["lora"])
    head.load_state_dict(ck["head"])

    class _SkewModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, x):
            return self.head(self.backbone(x).last_hidden_state[:, 0])

    return _SkewModel().eval()


def _build_orient():
    import torch
    import torch.nn as nn
    from transformers import SiglipVisionModel
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

    backbone = SiglipVisionModel.from_pretrained(_ORIENT_BACKBONE)
    backbone = get_peft_model(backbone, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.15, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    ))
    head = nn.Sequential(
        nn.LayerNorm(768),
        nn.Linear(768, 512), nn.GELU(), nn.Dropout(0.2),
        nn.Linear(512, 4),
    )
    ck = torch.load(str(_ORIENT_CKPT), map_location="cpu", weights_only=True)
    set_peft_model_state_dict(backbone, ck["lora"])
    head.load_state_dict(ck["head"])

    class _OrientModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, x):
            return self.head(self.backbone(x).pooler_output)

    return _OrientModel().eval()


def _build_torch_tfm(normalize):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(int(_IMG_SIZE * 1.14),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(_IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(*normalize),
    ])


def _ensure_models():
    global _orinet, _skew_model, _skew_tfm, _orient_model, _orient_tfm, _device, _ready
    if _ready:
        return

    _needs_torch = (ORIENT_BACKEND == "siglip") or (SKEW_BACKEND == "dinov2")
    if _needs_torch:
        import torch
        _device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- orientation ---
    if ORIENT_BACKEND == "siglip":
        _orient_model = _build_orient().to(_device)
        _orient_tfm = _build_torch_tfm(([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
    elif ORIENT_BACKEND == "paddle":
        global _paddle
        from .orient_paddle import PaddleOrientation
        _paddle = PaddleOrientation()
    else:  # OriNet (ONNX, GPU) — default
        from .orient_onnx import OriNetOrientation
        _orinet = OriNetOrientation(prefer_gpu=True)

    # --- fine angle ---
    if SKEW_BACKEND == "dinov2":
        _skew_model = _build_skew().to(_device)
        _skew_tfm = _build_torch_tfm(([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    # "fht" needs nothing to preload

    _ready = True


# ============== rotate / crop helpers ==============


def _rotate(img_pil, angle):
    if abs(angle) < 0.01:
        return img_pil
    return img_pil.rotate(
        angle, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255)
    )


def _crop_white(img_pil, threshold=250, margin_pct=0.02):
    arr = np.array(img_pil.convert("L"))
    mask = arr < threshold
    if not mask.any():
        return img_pil
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    h, w = arr.shape
    mh, mw = int(h * margin_pct), int(w * margin_pct)
    return img_pil.crop((
        max(0, cmin - mw),
        max(0, rmin - mh),
        min(w, cmax + 1 + mw),
        min(h, rmax + 1 + mh),
    ))


# ============== orientation / skew steps ==============


def _predict_orientation(image_pil):
    if ORIENT_BACKEND == "siglip":
        import torch
        x = _orient_tfm(image_pil).unsqueeze(0).to(_device)
        with torch.no_grad():
            cls = int(_orient_model(x).argmax(1).item())
        return _ORIENT_ANGLES[cls]
    if ORIENT_BACKEND == "paddle":
        return _paddle.orientation_deg(np.array(image_pil))[0]  # PP-LCNet: C4-TTA
    return _orinet.orientation_deg(np.array(image_pil))[0]  # OriNet: C4-TTA, same convention


def _predict_skew(cropped_pil):
    if SKEW_BACKEND == "dinov2":
        import torch
        x = _skew_tfm(cropped_pil).unsqueeze(0).to(_device)
        with torch.no_grad():
            return float(_skew_model(x).item())
    if SKEW_BACKEND == "projection":
        from .skew_fht import fine_angle_projection
        return fine_angle_projection(np.array(cropped_pil))
    if SKEW_BACKEND == "deskew":
        from .skew_fht import fine_angle_library
        return fine_angle_library(np.array(cropped_pil))
    from .skew_fht import fine_angle          # default: FHT
    return fine_angle(np.array(cropped_pil))


# ============== public API ==============


def detect_angle(image):
    """Определяет угол наклона документа.

    Args:
        image: PIL.Image или путь к файлу.

    Returns:
        float: угол в градусах (0–359). Поверни изображение на этот угол → выпрямится.
    """
    _ensure_models()

    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")
    image = ImageOps.exif_transpose(image)

    if PIPELINE_ORDER == "fht_first":
        # 1. FHT levels the doc to the nearest axis (removes diagonal tilt)
        skew = _predict_skew(image)
        leveled = _crop_white(_rotate(image, skew))
        # 2. orinet votes on an axis-aligned doc — its training distribution
        orientation = _predict_orientation(leveled)
        total = (float(orientation) + skew) % 360
        # 3. optional fine refinement on the now almost-upright page (small, safe range)
        if SKEW_REFINE == "projection":
            from .skew_fht import fine_angle_projection
            upright = _crop_white(_rotate(image, total))
            delta = fine_angle_projection(np.array(upright), rng=6.0)
            if abs(delta) <= 6.0:
                total = (total + delta) % 360
        return round(total, 2)

    # default "orient_first" (legacy order)
    # 1. Coarse orientation 0/90/180/270
    orientation = _predict_orientation(image)

    # 2. Rotate upright + adaptive crop (fine-angle model sees an almost-upright doc)
    corrected = _rotate(image, orientation)
    cropped = _crop_white(corrected)

    # 3. Fine-angle skew ±45°
    skew = _predict_skew(cropped)

    total = (float(orientation) + skew) % 360
    return round(total, 2)


def deskew(image):
    """Выпрямляет документ.

    Args:
        image: PIL.Image или путь к файлу.

    Returns:
        PIL.Image: выровненное изображение.
    """
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")

    angle = detect_angle(image)
    if abs(angle) < 0.01:
        return image
    return _rotate(image, angle)
