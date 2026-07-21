import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from utils import compute_iou, resolve_conflicts, draw_obb
from config import PREDICT_CONF, CLASS_CONF, CLASS_PRIORITY, CLASS_COLOR, IOU_THRESH, CONF_GAP


class Predictor:
    def __init__(self, model_path: Path, class_names: list[str]) -> None:
        self.model       = YOLO(str(model_path))
        self.class_names = class_names
        self.model_name  = model_path.name

    def predict(self, image_bytes: bytes) -> tuple[list[dict], np.ndarray]:
        """Inference trên raw image bytes.

        Returns:
            detections : list[dict] với keys class_id, class_name, confidence, polygon
            annotated  : BGR numpy array đã vẽ OBB
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Không decode được ảnh — kiểm tra định dạng file.")

        results = self.model.predict(source=img, conf=PREDICT_CONF, save=False, verbose=False)
        result  = results[0]
        obb     = result.obb

        detections = []
        annotated  = img.copy()

        if obb is None or len(obb) == 0:
            return detections, annotated

        keep_mask = [
            float(c) >= CLASS_CONF.get(int(cls), 0.25)
            for cls, c in zip(obb.cls, obb.conf)
        ]
        mask = torch.tensor(keep_mask)
        if not mask.any():
            return detections, annotated

        f_cls  = obb.cls[mask]
        f_conf = obb.conf[mask]
        f_xyxy = obb.xyxy[mask]
        f_poly = obb.xyxyxyxy[mask]

        keep_idx = resolve_conflicts(f_cls, f_conf, f_xyxy, IOU_THRESH, CONF_GAP, CLASS_PRIORITY)
        if not keep_idx:
            return detections, annotated

        for idx in keep_idx:
            cls_id   = int(f_cls[idx])
            conf_val = float(f_conf[idx])
            poly     = f_poly[idx].cpu().numpy()          # shape (4, 2)
            label    = f"{self.class_names[cls_id]} {conf_val:.2f}"
            color    = CLASS_COLOR.get(cls_id, (0, 255, 0))

            draw_obb(annotated, poly, label, color)
            detections.append({
                "class_id":   cls_id,
                "class_name": self.class_names[cls_id],
                "confidence": round(conf_val, 4),
                "polygon":    poly.tolist(),
            })

        return detections, annotated
