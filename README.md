# occular-deskew

Document deskew: detect and correct the rotation angle of a document of any kind — from passports and diplomas to receipts and forms. A single `deskew(image)` call returns the upright image.

## Installation

```bash
pip install occular-deskew
```

For GPU orientation add `onnxruntime-gpu`. Optional backends:
`pip install occular-deskew[deskew]` (pip deskew fine-angle), `[paddle]` (PP-LCNet orientation),
`[legacy]` (retired SigLIP/DINOv2). Or install from source:

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

## v0.5.0 — architecture

Lightweight pipeline, **torch-free by default** (ONNX + OpenCV):

1. **Fine-angle** — FHT (OpenCV Hough + weighted median), ±45°, levels the page to the nearest axis.
2. **Orientation** — OriNet (ONNX, ~4.8 MB, C4-TTA) → 0° / 90° / 180° / 270°, runs on GPU.

Order (`fht_first`, default): FHT levels the page → OriNet votes on the axis-aligned page (its training distribution) → total angle. This replaced the previous DINOv2 (skew) + SigLIP (orientation) LoRA models — both were retired.

Benchmarks:
- **Orientation: 100%** on opendocs (113 clean docs), 93.65% on a mixed 2000-scan set. Tesseract OSD: 78.8% on opendocs (with ~12% outright failures) and 20.4% on the noisy set.
- `fht_first` vs legacy orient-first order: errors >5° **127 vs 285** (180° flips 14 vs 105).
- Fine-angle (FHT) median residual **0.01°**.

### Backends (env vars)
- `OCCULAR_ORIENT_BACKEND` = `orinet` (default) | `siglip` (legacy, torch) | `paddle` (PP-LCNet)
- `OCCULAR_SKEW_BACKEND` = `fht` (default) | `projection` | `deskew` (pip `deskew` lib) | `dinov2` (legacy, torch)
- `OCCULAR_PIPELINE_ORDER` = `fht_first` (default) | `orient_first`

The default backends need neither torch nor any downloaded transformer weights.

### Pre-release check

```bash
python scripts/pre_release_check.py            # uses a clean doc set if found
OCCULAR_TEST_DOCS=/path/to/docs python scripts/pre_release_check.py
```

Smoke + regression gate: torch-free import, all backends run, OriNet weights load, orientation ≥ 95%, FHT fine median ≤ 1°. Exit 0 = safe to release, 1 = do not release (CI-friendly).

---

# occular-deskew (русский)

Document deskew: определение и коррекция угла наклона документа любого типа — от паспортов и дипломов до чеков и форм. Один вызов `deskew(image)` → выпрямленная картинка.

## Установка

```bash
pip install occular-deskew
```

Для GPU-ориентации доставьте `onnxruntime-gpu`. Опциональные бэкенды:
`pip install occular-deskew[deskew]` (мелкий угол через pip deskew), `[paddle]` (ориентация PP-LCNet),
`[legacy]` (выведенные SigLIP/DINOv2). Либо из исходников:

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

## v0.5.0 — архитектура

Лёгкий пайплайн, **по умолчанию без torch** (ONNX + OpenCV):

1. **Мелкий угол** — FHT (OpenCV Hough + взвешенная медиана), ±45°, выравнивает страницу по ближайшей оси.
2. **Ориентация** — OriNet (ONNX, ~4.8 МБ, C4-TTA) → 0° / 90° / 180° / 270°, работает на GPU.

Порядок (`fht_first`, по умолчанию): FHT выравнивает по осям → OriNet голосует на осе-выровненной странице (в своём распределении обучения) → итоговый угол. Это заменило прежние LoRA-модели DINOv2 (наклон) + SigLIP (ориентация) — обе выведены из эксплуатации.

Бенчмарки:
- **Ориентация: 100%** на opendocs (113 чистых док.), 93.65% на смешанном наборе из 2000 сканов. Tesseract OSD: 78.8% на opendocs (с ~12% отказов) и 20.4% на грязном наборе.
- Порядок `fht_first` vs старый orient-first: ошибок >5° **127 vs 285** (180°-перевороты 14 vs 105).
- Мелкий угол (FHT), median остаточной ошибки **0.01°**.

### Бэкенды (переменные окружения)
- `OCCULAR_ORIENT_BACKEND` = `orinet` (по умолч.) | `siglip` (legacy, torch) | `paddle` (PP-LCNet)
- `OCCULAR_SKEW_BACKEND` = `fht` (по умолч.) | `projection` | `deskew` (pip-библиотека `deskew`) | `dinov2` (legacy, torch)
- `OCCULAR_PIPELINE_ORDER` = `fht_first` (по умолч.) | `orient_first`

Бэкенды по умолчанию не требуют ни torch, ни скачивания трансформер-весов.

### Проверка перед релизом

```bash
python scripts/pre_release_check.py            # использует чистый набор, если найден
OCCULAR_TEST_DOCS=/путь/к/докам python scripts/pre_release_check.py
```

Smoke + регрессия: torch-free импорт, все бэкенды работают, веса OriNet грузятся, ориентация ≥ 95%, median FHT ≤ 1°. Exit 0 = можно релизить, 1 = нельзя (для CI).
