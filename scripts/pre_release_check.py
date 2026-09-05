#!/usr/bin/env python3
"""Pre-release checker for occular-deskew.

Runs a battery of checks before tagging a release and prints a PASS/FAIL report.
Exit code 0 = all required checks passed, 1 = at least one required check failed.

Usage:
    python scripts/pre_release_check.py
    OCCULAR_TEST_DOCS=/path/to/clean/docs python scripts/pre_release_check.py

Checks
  smoke (required):
    - package imports with NO torch present (default backends are torch-free)
    - detect_angle / deskew are callable, return a valid angle in [0, 360)
    - OriNet weights file is present and loads (GPU if available, else CPU)
    - fine-angle backends 'fht' and 'projection' run; 'deskew' runs if installed
  regression (required if a clean doc set is available, else SKIPPED):
    - orientation accuracy on 90-multiple rotations >= ORIENT_MIN
    - fine-angle (FHT) median residual on small tilts <= FINE_MED_MAX
"""
import os
import sys
import glob
import random
import importlib
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

# thresholds (set with margin below measured numbers)
ORIENT_MIN = 0.95      # measured 1.00 on opendocs
FINE_MED_MAX = 1.0     # measured ~0.01-0.21 deg (FHT)
N_DOCS = 40            # sample size for the regression checks

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

_results = []  # (name, required, status, detail)


def record(name, required, ok, detail=""):
    _results.append((name, required, "PASS" if ok else "FAIL", detail))
    return ok


def find_docs():
    env = os.environ.get("OCCULAR_TEST_DOCS")
    cands = [env] if env else []
    cands += [
        os.path.expanduser("~/opendoc_v2/opendoc_dataset_v2/images"),
    ]
    for d in cands:
        if d and os.path.isdir(d):
            fs = []
            for e in ("jpg", "jpeg", "png"):
                fs += glob.glob(os.path.join(d, f"*.{e}"))
            if fs:
                return sorted(fs)
    return []


# ---------------- smoke checks ----------------

def check_torch_free_import():
    # ensure defaults don't require torch: import package with torch hidden
    import builtins
    real_import = builtins.__import__

    def guard(name, *a, **k):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch is hidden by pre_release_check (defaults must be torch-free)")
        return real_import(name, *a, **k)

    builtins.__import__ = guard
    try:
        for m in list(sys.modules):
            if m.startswith("occular_deskew"):
                del sys.modules[m]
        import occular_deskew  # noqa
        importlib.import_module("occular_deskew.deskew")
        return record("import: torch-free package import", True, True)
    except Exception as e:
        return record("import: torch-free package import", True, False, repr(e)[:120])
    finally:
        builtins.__import__ = real_import


def check_api_smoke():
    try:
        import numpy as np
        from PIL import Image
        from occular_deskew import detect_angle, deskew
        # a synthetic page with clear horizontal text-like bars
        arr = np.full((600, 450, 3), 255, np.uint8)
        for y in range(60, 560, 40):
            arr[y:y + 12, 40:410] = 20
        img = Image.fromarray(arr)
        ang = detect_angle(img)
        ok = isinstance(ang, float) and 0.0 <= ang < 360.0
        out = deskew(img)
        ok = ok and out is not None
        return record("smoke: detect_angle/deskew callable", True, ok, f"angle={ang}")
    except Exception as e:
        return record("smoke: detect_angle/deskew callable", True, False, repr(e)[:120])


def check_weights_and_orinet():
    try:
        from occular_deskew.orient_onnx import OriNetOrientation, _WEIGHTS
        if not os.path.isfile(_WEIGHTS):
            return record("weights: OriNet onnx present", True, False, f"missing {_WEIGHTS}")
        det = OriNetOrientation(prefer_gpu=True)
        return record("weights: OriNet loads", True, True, f"provider={det.provider}")
    except Exception as e:
        return record("weights: OriNet loads", True, False, repr(e)[:120])


def check_skew_backends():
    import numpy as np
    from occular_deskew import skew_fht
    arr = np.full((500, 400, 3), 255, np.uint8)
    for y in range(50, 460, 30):
        arr[y:y + 8, 30:370] = 15
    ok_all = True
    # required: fht, projection
    for name, fn in (("fht", skew_fht.fine_angle), ("projection", skew_fht.fine_angle_projection)):
        try:
            v = fn(arr); ok = isinstance(v, float)
        except Exception as e:
            ok = False; v = repr(e)[:60]
        ok_all &= record(f"skew backend '{name}' runs", True, ok, f"->{v}")
    # optional: deskew library
    try:
        v = skew_fht.fine_angle_library(arr)
        record("skew backend 'deskew' (optional)", False, isinstance(v, float), f"->{v}")
    except ImportError:
        record("skew backend 'deskew' (optional)", False, True, "not installed (ok)")
    except Exception as e:
        record("skew backend 'deskew' (optional)", False, False, repr(e)[:80])
    return ok_all


# ---------------- regression checks ----------------

def check_orientation_regression(files):
    try:
        import numpy as np, cv2
        from occular_deskew.orient_onnx import OriNetOrientation
        det = OriNetOrientation(prefer_gpu=True)
        rng = random.Random(0)
        cwmap = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
        ok = tot = 0
        for f in files[:N_DOCS]:
            im = cv2.imread(f)
            if im is None:
                continue
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            a = rng.choice([0, 90, 180, 270])
            rot = im if a == 0 else cv2.rotate(im, cwmap[a])
            deg, _ = det.orientation_deg(rot)
            ok += (deg == a); tot += 1
        acc = ok / tot if tot else 0
        return record(f"regression: orientation acc >= {ORIENT_MIN:.0%}", True,
                      acc >= ORIENT_MIN, f"{ok}/{tot} = {acc:.1%}")
    except Exception as e:
        return record("regression: orientation accuracy", True, False, repr(e)[:120])


def check_fine_regression(files):
    try:
        import numpy as np, cv2
        from occular_deskew.skew_fht import fine_angle
        from PIL import Image
        rng = random.Random(1)
        resid = []
        for f in files[:N_DOCS]:
            im = Image.open(f).convert("RGB")
            th = round(rng.uniform(-15, 15), 2)
            rot = np.array(im.rotate(-th, resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255)))
            pred = fine_angle(rot)
            d = ((th - pred + 45) % 90) - 45
            resid.append(abs(d))
        med = float(np.median(resid)) if resid else 99
        return record(f"regression: FHT fine median <= {FINE_MED_MAX}deg", True,
                      med <= FINE_MED_MAX, f"median={med:.2f}deg n={len(resid)}")
    except Exception as e:
        return record("regression: FHT fine median", True, False, repr(e)[:120])


# ---------------- runner ----------------

def main():
    print("=" * 66)
    print(" occular-deskew — pre-release check")
    print("=" * 66)

    check_torch_free_import()
    check_api_smoke()
    check_weights_and_orinet()
    check_skew_backends()

    files = find_docs()
    if files:
        print(f"\n[regression on {min(len(files), N_DOCS)} docs from a clean set]")
        check_orientation_regression(files)
        check_fine_regression(files)
    else:
        _results.append(("regression: clean doc set", True, "SKIP",
                         "set OCCULAR_TEST_DOCS=/path to enable"))

    print()
    width = max(len(n) for n, *_ in _results)
    failed = 0
    for name, required, status, detail in _results:
        req = "" if required else " (optional)"
        print(f"  [{status:4s}]  {name:<{width}}{req}  {detail}")
        if status == "FAIL" and required:
            failed += 1

    print("=" * 66)
    if failed:
        print(f" RESULT: FAIL — {failed} required check(s) failed. Do NOT release.")
        return 1
    print(" RESULT: PASS — safe to release.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
