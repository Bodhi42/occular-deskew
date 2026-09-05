"""Fine-angle deskew (±45°) via FHT — classical OpenCV (HoughLinesP + weighted median).

Replaces the DINOv2-Small skew regressor. No weights, no torch, ~0.1-0.2 s/page.
Ported from occular-ocr (occular/deskew.py).

Convention: returns an angle in degrees such that rotating the image by +angle
(counter-clockwise, as PIL `Image.rotate(+angle)` does) uprights it — same sign
convention deskew.py already sums into `total`.

Guard "do no harm": rotates ONLY when detection is confident (lines agree) and the
angle is in a sane range [lo, hi]. Otherwise returns 0 → leaves the page untouched.
"""
import cv2
import numpy as np


def detect_skew_angle(gray, min_conf: float = 0.6, lo: float = 0.7, hi: float = 45.0):
    """FHT (HoughLinesP + weighted median of angles). Returns (angle°, apply?)."""
    if max(gray.shape) > 1600:
        s = 1600 / max(gray.shape)
        gray = cv2.resize(gray, None, fx=s, fy=s)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    if lines is None:
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=20)
    if lines is None:
        return 0.0, False
    angs, ws = [], []
    for l in lines:
        x1, y1, x2, y2 = [float(v) for v in np.asarray(l).ravel()[:4]]
        L = np.hypot(x2 - x1, y2 - y1)
        if L < 20:
            continue
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        while a > 45:                                # skew, not orientation -> fold to ±45°
            a -= 90
        while a < -45:
            a += 90
        angs.append(a); ws.append(L)
    if not angs:
        return 0.0, False
    angs = np.array(angs); ws = np.array(ws); tot = ws.sum()
    order = np.argsort(angs); med = float(np.median(angs)); c = 0.0
    for i in order:                                  # weighted median (longer lines matter more)
        c += ws[i]
        if c >= tot / 2:
            med = float(angs[i]); break
    conf = ws[np.abs(angs - med) <= 2.0].sum() / tot
    apply = (conf >= min_conf) and (lo <= abs(med) <= hi)
    return round(med, 2), apply


def fine_angle(image_rgb: np.ndarray) -> float:
    """RGB numpy -> fine skew angle in degrees (0.0 if not confident). Sign matches PIL rotate(+a)."""
    gray = image_rgb if image_rgb.ndim == 2 else cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    angle, apply = detect_skew_angle(gray)
    return angle if apply else 0.0


# ---------------------------------------------------------------------------
# Projection-profile skew detector (coarse→fine over ±45°).
# Sharper than Hough on text pages: rotating the edge map so text rows line up
# maximises the row-sum profile's sharpness. Sign matches PIL rotate(+a).
# ---------------------------------------------------------------------------

def _profile_score(edges, ang):
    h, w = edges.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    r = cv2.warpAffine(edges, M, (w, h), flags=cv2.INTER_NEAREST)
    proj = r.sum(axis=1).astype(np.float64)
    d = np.diff(proj)
    return float((d * d).sum())


def fine_angle_projection(image_rgb: np.ndarray, rng: float = 45.0) -> float:
    """RGB numpy -> fine skew angle (°) via projection profile, coarse→fine over ±rng.
    Returns the angle to rotate by (PIL rotate(+a)) to level the page; 0.0 if no signal."""
    gray = image_rgb if image_rgb.ndim == 2 else cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    if max(gray.shape) > 700:                         # projection is cheap on a downscaled map
        s = 700 / max(gray.shape)
        gray = cv2.resize(gray, None, fx=s, fy=s)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    if edges.sum() < 255 * 50:                        # almost no edges -> nothing to align
        return 0.0
    # coarse sweep
    coarse = np.arange(-rng, rng + 0.001, 2.0)
    scores = [_profile_score(edges, a) for a in coarse]
    ba = float(coarse[int(np.argmax(scores))])
    # fine sweep around the coarse best
    fine = np.arange(ba - 2.0, ba + 2.0 + 0.001, 0.25)
    scores = [_profile_score(edges, a) for a in fine]
    best = float(fine[int(np.argmax(scores))])
    # projection maximised when text rows are horizontal -> rotate by -best to level
    return round(-best, 2)


def fine_angle_library(image_rgb: np.ndarray) -> float:
    """RGB numpy -> fine skew angle (°) via the pip `deskew` library (Hough, scikit-image).
    Returns the angle to rotate by (PIL rotate(+a)) to level the page; 0.0 if no signal.
    Requires `pip install deskew` (pulls scikit-image)."""
    try:
        from deskew import determine_skew
    except ImportError as e:
        raise ImportError(
            "SKEW_BACKEND='deskew' needs the 'deskew' package: pip install deskew"
        ) from e
    gray = image_rgb if image_rgb.ndim == 2 else cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    angle = determine_skew(gray, min_angle=-45.0, max_angle=45.0)
    if angle is None:
        return 0.0
    angle = float(angle)
    return round(angle, 2) if abs(angle) <= 45.0 else 0.0
