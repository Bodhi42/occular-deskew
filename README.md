# occular-deskew

Document deskew: detect and correct the rotation angle of a document of any kind — from passports and diplomas to receipts and forms. A single `deskew(image)` call returns the upright image.

## Installation

```bash
pip install git+https://github.com/Bodhi42/occular-deskew.git
```

## Usage

```python
from occular_deskew import detect_angle, deskew
from PIL import Image

img = Image.open("doc.jpg")

# 1. Get the angle
angle = detect_angle(img)          # float, 0-359°
print(f"Angle: {angle}°")

# 2. Straighten the image
upright = deskew(img)              # PIL.Image
upright.save("doc_upright.jpg")
```

Accepts a `PIL.Image` or a path to a file.

## v0.4.0 — architecture

Two-stage pipeline based on **transformer backbones with LoRA adapters**:

1. **Fine-angle skew regressor** — DINOv2-Small (22M params, frozen) + LoRA r=8 + regression head. Predicts a fine rotation angle in the range ±45°. Trained on 160k auto-labeled documents. Val: AED 0.33°, p95 0.92°, acc≤2°=98.7%, acc≤5°=99.8%.
2. **Orientation classifier** — SigLIP-Base (93M params, frozen) + LoRA r=16 + classification head. Classifies coarse orientation among 0° / 90° / 180° / 270°. Trained on 89k documents from 139 countries with EMA + CutMix. Val: 97.43% on val × 4 rotations.

Order: skew regressor → rotate → adaptive crop → orientation classifier → total angle.

LoRA-trainable parameters: **0.39M (skew) + 1.60M (orient) = ~2M** total. Both base models are downloaded from HuggingFace Hub on first call (~370 MB SigLIP + ~85 MB DINOv2 cached locally).

---

# occular-deskew (русский)

Document deskew: определение и коррекция угла наклона документа любого типа — от паспортов и дипломов до чеков и форм. Один вызов `deskew(image)` → выпрямленная картинка.

## Установка

```bash
pip install git+https://github.com/Bodhi42/occular-deskew.git
```

## Использование

```python
from occular_deskew import detect_angle, deskew
from PIL import Image

img = Image.open("doc.jpg")

# 1. Просто получить угол
angle = detect_angle(img)          # float, 0-359°
print(f"Angle: {angle}°")

# 2. Выпрямить картинку
upright = deskew(img)              # PIL.Image
upright.save("doc_upright.jpg")
```

Принимает `PIL.Image` или путь к файлу.

## v0.4.0 — архитектура

Двухступенчатый пайплайн на **трансформер-бэкбонах с LoRA-адаптерами**:

1. **Регрессор мелкого угла** — DINOv2-Small (22M params, заморожен) + LoRA r=8 + regression-голова. Предсказывает мелкий угол поворота в диапазоне ±45°. Обучен на 160k автоматически размеченных документов. Val: AED 0.33°, p95 0.92°, acc≤2°=98.7%, acc≤5°=99.8%.
2. **Классификатор ориентации** — SigLIP-Base (93M params, заморожен) + LoRA r=16 + classification-голова. Определяет крупную ориентацию среди 0° / 90° / 180° / 270°. Обучен на 89k документов из 139 стран с EMA + CutMix. Val: 97.43% на val × 4 ротациях.

Порядок: регрессор → поворот → adaptive crop → классификатор → итоговый угол.

Trainable LoRA-параметров: **0.39M (skew) + 1.60M (orient) = ~2M** всего. Базовые модели подтягиваются с HuggingFace Hub при первом вызове (~370 МБ SigLIP + ~85 МБ DINOv2 кэшируются локально).
