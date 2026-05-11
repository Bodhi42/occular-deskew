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

## v0.3.0 — architecture

Two-stage pipeline:

1. **Light Student** (MobileNetV3-Small + Linear head) — fine angle ±30°. Trained on 160k oracle-cleaned documents.
2. **Phase C orientation** (MobileNetV3-Large + AttentionPool + Cosine τ) — coarse 0°/90°/180°/270°. Train pool: 112k (47k Phase A + 24 domains + 139 countries).

Order: student → rotate → adaptive crop → Phase C → total angle.

## Quality (1000 in-distribution, GT = 0°)

| acc≤2° | acc≤5° | AED | p95 |
|---|---|---|---|
| 91.3% | 95.9% | 5.95° | 3.76° |

On clean English forms (FUNSD): **99%** acc≤2°.

The main source of remaining ≥5° errors is Phase C 90°/180°-flip (~88% of all "large" errors); the fine model errs by small angles in 5% of cases (≤8°).

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

## v0.3.0 — архитектура

Двухступенчатый пайплайн:

1. **Light Student** (MobileNetV3-Small + Linear head) — fine angle ±30°. Обучен на 160k oracle-cleaned документов.
2. **Phase C orientation** (MobileNetV3-Large + AttentionPool + Cosine τ) — coarse 0°/90°/180°/270°. Train pool: 112k (47k Phase A + 24 domains + 139 countries).

Порядок: student → rotate → adaptive crop → Phase C → total angle.

## Качество (1000 in-distribution, GT=0°)

| acc≤2° | acc≤5° | AED | p95 |
|---|---|---|---|
| 91.3% | 95.9% | 5.95° | 3.76° |

На англоязычных формах (FUNSD): **99%** acc≤2°.

Главный источник остаточных ошибок ≥5° — Phase C 90°/180°-flip (~88% от всех «больших» ошибок); fine-модель ошибается мелко в 5% случаев (≤8°).
