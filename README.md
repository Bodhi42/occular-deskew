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

1. **Fine-angle regressor** (MobileNetV3-Small + Linear head) — predicts a small rotation angle in the range ±30°. Trained on 160k machine-labeled documents.
2. **Orientation classifier** (MobileNetV3-Large + AttentionPool + cosine head) — classifies coarse orientation among 0° / 90° / 180° / 270°. Trained on 112k documents covering 139 countries and 24 document types.

Order: regressor → rotate → adaptive crop → classifier → total angle.

## Quality (1000 in-distribution images, GT = 0°)

| acc≤2° | acc≤5° | AED | p95 |
|---|---|---|---|
| 91.3% | 95.9% | 5.95° | 3.76° |

On clean English forms (FUNSD): **99%** acc≤2°.

The main source of remaining ≥5° errors is the orientation classifier producing a 90°/180° flip (~88% of all "large" errors); the fine-angle regressor errs by small angles in about 5% of cases (≤8°).

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

1. **Регрессор малых углов** (MobileNetV3-Small + Linear-голова) — предсказывает мелкий угол поворота в диапазоне ±30°. Обучен на 160k автоматически размеченных документов.
2. **Классификатор ориентации** (MobileNetV3-Large + AttentionPool + cosine-голова) — определяет крупную ориентацию среди 0° / 90° / 180° / 270°. Обучен на 112k документов из 139 стран по 24 типам.

Порядок: регрессор → поворот → adaptive crop → классификатор → итоговый угол.

## Качество (1000 in-distribution картинок, GT = 0°)

| acc≤2° | acc≤5° | AED | p95 |
|---|---|---|---|
| 91.3% | 95.9% | 5.95° | 3.76° |

На чистых англоязычных формах (FUNSD): **99%** acc≤2°.

Главный источник остаточных ошибок ≥5° — классификатор ориентации даёт 90°/180°-flip (~88% от всех «больших» ошибок); регрессор малых углов ошибается мелко примерно в 5% случаев (≤8°).
