"""Orientation backend via PaddleX PP-LCNet_x1_0_doc_ori, with C4-TTA aggregation.

Same interface as orient_onnx.OriNetOrientation: predict(rgb)->(class0..3, conf),
orientation_deg(rgb)->(deg in {0,90,180,270}, conf). Used to A/B against OriNet
inside the same fht_first pipeline.

PP-LCNet returns only top-1 (label + score), so C4 aggregation votes over the four
90° rotations weighted by score (instead of averaging full class distributions).
"""
import os
import sys
import numpy as np
import cv2

ANGLES = (0, 90, 180, 270)


class PaddleOrientation:
    def __init__(self, prefer_gpu: bool = True):
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        from paddlex import create_model
        self.model = create_model("PP-LCNet_x1_0_doc_ori")
        # sign of the aggregation shift; calibrated once on first batch if needed
        self._shift = -1
        print("Paddle PP-LCNet_x1_0_doc_ori orientation loaded (CPU)", file=sys.stderr)

    @staticmethod
    def _rotate(image, k):
        if k == 0:
            return image
        return cv2.rotate(image, {1: cv2.ROTATE_90_CLOCKWISE,
                                  2: cv2.ROTATE_180,
                                  3: cv2.ROTATE_90_COUNTERCLOCKWISE}[k])

    def _predict_batch(self, imgs):
        """-> list of (class_idx 0..3, score) for each image."""
        out = []
        for r in self.model.predict(imgs, batch_size=len(imgs)):
            lbl = int(r["label_names"][0])      # "0"/"90"/"180"/"270"
            sc = float(r["scores"][0])
            out.append((lbl // 90, sc))
        return out

    def predict(self, image: np.ndarray):
        """C4-TTA vote -> (class 0..3, confidence). class k = image rotated ANGLES[k] CW."""
        rots = [self._rotate(image, k) for k in range(4)]
        preds = self._predict_batch(rots)
        scores4 = np.zeros(4)
        for k, (c, sc) in enumerate(preds):
            scores4[(c + self._shift * k) % 4] += sc
        tot = scores4.sum()
        k = int(scores4.argmax())
        return k, float(scores4[k] / tot if tot else 0.0)

    def orientation_deg(self, image: np.ndarray):
        k, conf = self.predict(image)
        return ANGLES[k], conf
