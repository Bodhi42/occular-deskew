"""occular-deskew v0.4.0 — transformer-based document deskew.

Pipeline:
  1. Skew regressor (DINOv2-Small + LoRA r=8)  → fine angle ±45°
  2. rotate + adaptive crop
  3. Orientation classifier (SigLIP-Base + LoRA r=16) → 0° / 90° / 180° / 270°
  4. total = (skew + orientation) % 360

Weights:
  - dinov2_skew_lora_v0.4.0.pth   (89 MB, DINOv2-Small backbone + LoRA + reg head)
  - siglip_orient_lora_v0.4.0.pth (378 MB, SigLIP-Base backbone + LoRA + class head)

On first call, transformers will lazily download the base model identifiers
from HuggingFace if not already cached locally.

Validation metrics (val_list × 4 rotations / clean_0deg val 32k):
  - Orientation classifier (SigLIP+LoRA v2): val_acc 97.43% (best @ ep 12)
  - Fine-angle regressor (DINOv2+LoRA, continued): AED 0.33°, p95 0.92°,
    acc≤2°=98.7%, acc≤5°=99.8% on ±45° range
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms

_WEIGHTS_DIR = Path(__file__).parent / "weights"
_SKEW_CKPT = _WEIGHTS_DIR / "dinov2_skew_v0.4.0.pth"
_ORIENT_CKPT = _WEIGHTS_DIR / "siglip_orient_v0.4.0.pth"

_SKEW_BACKBONE = "facebook/dinov2-small"
_ORIENT_BACKBONE = "google/siglip-base-patch16-224"
_ORIENT_ANGLES = [0, 90, 180, 270]
_IMG_SIZE = 224

_skew_model = None
_skew_tfm = None
_orient_model = None
_orient_tfm = None
_device = None


# ============== Model builders ==============


def _build_skew():
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


def _ensure_models():
    global _skew_model, _skew_tfm, _orient_model, _orient_tfm, _device
    if _skew_model is not None and _orient_model is not None:
        return

    _device = "cuda" if torch.cuda.is_available() else "cpu"

    _skew_model = _build_skew().to(_device)
    _skew_tfm = transforms.Compose([
        transforms.Resize(int(_IMG_SIZE * 1.14),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(_IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    _orient_model = _build_orient().to(_device)
    _orient_tfm = transforms.Compose([
        transforms.Resize(int(_IMG_SIZE * 1.14),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(_IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


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

    # 1. Fine-angle regressor ±45°
    x = _skew_tfm(image).unsqueeze(0).to(_device)
    with torch.no_grad():
        skew = float(_skew_model(x).item())

    # 2. Поворот по skew + adaptive crop
    corrected = _rotate(image, skew)
    cropped = _crop_white(corrected)

    # 3. Orientation classifier 0/90/180/270
    x = _orient_tfm(cropped).unsqueeze(0).to(_device)
    with torch.no_grad():
        cls = int(_orient_model(x).argmax(1).item())
    orientation = _ORIENT_ANGLES[cls]

    total = (skew + float(orientation)) % 360
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
