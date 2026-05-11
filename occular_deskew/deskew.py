"""
occular-deskew: document rotation detection and correction.

v0.3.0 pipeline:
  Fine-angle regressor (MobileNetV3-Small + Linear) → small angle (±30°)
  → rotate + adaptive crop
  → orientation classifier (MobileNetV3-Large + AttentionPool + Cosine τ) → 0/90/180/270

Weights:
  - orientation_classifier.pth — coarse 0°/90°/180°/270° (112k docs, 139 countries, 24 types)
  - fine_angle_regressor.pth — small-angle regressor (160k auto-labeled docs, ±30° range)

Quality (1000 in-distribution images, GT=0°):
  acc≤2° = 91.3%, acc≤5° = 95.9%, p95 = 3.76°
  Main remaining ≥5° error source: orientation classifier 90°/180°-flip (~88%);
  fine regressor errs by ≤8° in about 5% of cases.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from pathlib import Path

_ORIENT_WEIGHTS = Path(__file__).parent / "weights" / "orientation_classifier.pth"
_REGRESSOR_WEIGHTS = Path(__file__).parent / "weights" / "fine_angle_regressor.pth"
_ORIENT_ANGLES = [0, 90, 180, 270]
_ORIENT_FEAT_DIM = 960
_ORIENT_IMG_SIZE = 320
_REGRESSOR_IMG_SIZE = 224
_REGRESSOR_RANGE = 30.0

_orient_model = None
_orient_tfm = None
_regressor = None
_regressor_tfm = None
_device = None


# ============== Orientation: MobileNetV3-Large + AttentionPool + Cosine τ ==============

class _CosineLearnableTemp(nn.Module):
    def __init__(self, in_features, num_classes=4):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        self.log_temperature = nn.Parameter(torch.tensor(np.log(16.0)))

    def forward(self, x):
        temp = self.log_temperature.exp()
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.weight, dim=1)
        return temp * x_norm @ w_norm.t()


class _AttentionPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(_ORIENT_FEAT_DIM, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1),
        )

    def forward(self, x):
        w = self.attn(x)
        w = w.view(w.size(0), -1)
        w = F.softmax(w, dim=1)
        w = w.view(w.size(0), 1, x.size(2), x.size(3))
        return (x * w).sum(dim=[2, 3])


class _OrientationModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.mobilenet_v3_large(weights=None)
        self.features = backbone.features
        self.pool = _AttentionPool()
        self.proj = nn.Sequential(
            nn.Linear(_ORIENT_FEAT_DIM, 1280),
            nn.Hardswish(),
            nn.Dropout(0.2),
        )
        self.head = _CosineLearnableTemp(1280)

    def forward(self, x):
        return self.head(self.proj(self.pool(self.features(x))))


# ============== Skew regressor: MobileNetV3-Small + Linear head ==============

class _SkewRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        m = models.mobilenet_v3_small(weights=None)
        in_features = m.classifier[0].in_features  # 576
        self.features = m.features
        self.avgpool = m.avgpool
        self.head = nn.Sequential(
            nn.Linear(in_features, 256), nn.Hardswish(),
            nn.Dropout(0.1), nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x).flatten(1)
        return self.head(x)


# ============== model loaders ==============

def _ensure_models():
    global _orient_model, _orient_tfm, _regressor, _regressor_tfm, _device
    if _orient_model is not None and _regressor is not None:
        return

    _device = "cuda" if torch.cuda.is_available() else "cpu"

    _orient_model = _OrientationModel()
    _orient_model.load_state_dict(
        torch.load(str(_ORIENT_WEIGHTS), map_location="cpu", weights_only=True)
    )
    _orient_model.eval().to(_device)
    _orient_tfm = transforms.Compose([
        transforms.Resize(365),
        transforms.CenterCrop(_ORIENT_IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Адаптация: best.pth student'а сохранён как state_dict mobilenet_v3_small
    # с custom classifier. Мы строим обёртку чтобы воспроизвести структуру.
    student_state = torch.load(str(_REGRESSOR_WEIGHTS), map_location="cpu", weights_only=True)
    _regressor = _build_regressor_from_state(student_state)
    _regressor.eval().to(_device)
    _regressor_tfm = transforms.Compose([
        transforms.Resize(int(_REGRESSOR_IMG_SIZE * 1.14)),
        transforms.CenterCrop(_REGRESSOR_IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def _build_regressor_from_state(state_dict):
    """state_dict student'а сохранён как mobilenet_v3_small с custom classifier.
    Загружаем как родную torchvision сетку с переопределённым classifier."""
    m = models.mobilenet_v3_small(weights=None)
    in_features = m.classifier[0].in_features
    m.classifier = nn.Sequential(
        nn.Linear(in_features, 256), nn.Hardswish(),
        nn.Dropout(0.1), nn.Linear(256, 1),
    )
    m.load_state_dict(state_dict)
    return m


def _orient_probs(img_pil):
    _ensure_models()
    t = _orient_tfm(img_pil.convert("RGB")).unsqueeze(0).to(_device)
    with torch.no_grad():
        return F.softmax(_orient_model(t), dim=1)[0].cpu().numpy()


def _orient_predict(img_pil):
    return _ORIENT_ANGLES[int(np.argmax(_orient_probs(img_pil)))]


def _regressor_predict(img_pil):
    _ensure_models()
    t = _regressor_tfm(img_pil.convert("RGB")).unsqueeze(0).to(_device)
    with torch.no_grad():
        return float(_regressor(t).item())


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
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")

    # 1. Light student → мелкий угол (±30°)
    skew = _regressor_predict(image)
    corrected = _rotate(image, skew)

    # 2. Adaptive crop — убираем белые поля после поворота
    cropped = _crop_white(corrected)

    # 3. Phase C orientation → 0/90/180/270
    orientation = _orient_predict(cropped)

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
