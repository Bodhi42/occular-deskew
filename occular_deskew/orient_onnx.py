"""OriNet-based coarse orientation (0/90/180/270), ONNX + GPU.

Drop-in replacement for the SigLIP+LoRA orientation classifier used in deskew.py.
Ported from occular-ocr's orientation.py, but runs on CUDAExecutionProvider and
returns the same [0,90,180,270] convention deskew already expects.

Model: orientation_orinet_fp32.onnx (~4.8 MB). Input 320x320 letterbox (white pad),
ImageNet norm. Inference runs the C4 orbit (4 rotations) and averages the shifted
answers — this aggregation is part of the model and removes ~40% of single-pass errors.
"""
import os
import sys
import threading
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

SIZE = 320
ANGLES = (0, 90, 180, 270)          # class k -> image is rotated ANGLES[k] clockwise
MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)
_IDX = (np.arange(4).reshape(4, 1) + np.arange(4).reshape(1, 4)) % 4

_WEIGHTS = Path(__file__).parent / "weights" / "orientation_orinet_fp32.onnx"


def _ensure_cuda_libs_on_path():
    """onnxruntime-gpu 1.28 needs CUDA 13 / cuDNN 9 .so's on the loader path.
    They ship as pip 'nvidia-*' wheels under site-packages; add them if present."""
    cands = []
    for base in (Path.home() / ".local/lib/python3.12/site-packages/nvidia",):
        cands += [base / "cu13/lib", base / "cudnn/lib"]
    extra = os.pathsep.join(str(p) for p in cands if p.is_dir())
    if extra:
        os.environ["LD_LIBRARY_PATH"] = extra + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")


class OriNetOrientation:
    def __init__(self, prefer_gpu: bool = True):
        import onnxruntime as ort
        if prefer_gpu:
            _ensure_cuda_libs_on_path()
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if prefer_gpu else ["CPUExecutionProvider"])
        self.session = ort.InferenceSession(str(_WEIGHTS), sess_options=so, providers=providers)
        self.provider = self.session.get_providers()[0]
        self.input_name = self.session.get_inputs()[0].name
        self._lock = threading.Lock()
        print(f"OriNet orientation loaded ({self.provider})", file=sys.stderr)

    @staticmethod
    def _letterbox(image: np.ndarray) -> np.ndarray:
        im = Image.fromarray(image)
        w, h = im.size
        if w >= h:
            nw, nh = SIZE, max(1, round(h * SIZE / w))
        else:
            nw, nh = max(1, round(w * SIZE / h)), SIZE
        canvas = Image.new("RGB", (SIZE, SIZE), (255, 255, 255))
        canvas.paste(im.resize((nw, nh), Image.BILINEAR), ((SIZE - nw) // 2, (SIZE - nh) // 2))
        return np.asarray(canvas)

    @classmethod
    def _prep(cls, image: np.ndarray) -> np.ndarray:
        x = cls._letterbox(image).astype(np.float32).transpose(2, 0, 1) / 255.0
        return (x - MEAN) / STD

    @staticmethod
    def _rotate(image: np.ndarray, k: int) -> np.ndarray:
        if k == 0:
            return image
        return cv2.rotate(image, {1: cv2.ROTATE_90_CLOCKWISE,
                                  2: cv2.ROTATE_180,
                                  3: cv2.ROTATE_90_COUNTERCLOCKWISE}[k])

    def predict(self, image: np.ndarray) -> Tuple[int, float]:
        """-> (class 0..3, confidence). class k = image is rotated ANGLES[k] clockwise."""
        batch = np.stack([self._prep(self._rotate(image, k)) for k in range(4)])
        with self._lock:
            logits = self.session.run(None, {self.input_name: batch})[0]
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        agg = np.take_along_axis(probs, _IDX, axis=1).mean(axis=0)
        k = int(agg.argmax())
        return k, float(agg[k])

    def orientation_deg(self, image: np.ndarray) -> Tuple[int, float]:
        """-> (angle in {0,90,180,270} to rotate the image upright, confidence).
        Matches deskew's _ORIENT_ANGLES convention (PIL rotate(+angle) uprights it)."""
        k, conf = self.predict(image)
        return ANGLES[k], conf
