from collections import Counter
from pathlib import Path
import cv2
import numpy as np
import torch
import yaml


def find_latest_tune_hyper(tune_dir: Path) -> dict:
    runs = sorted(
        [d for d in tune_dir.iterdir() if d.is_dir() and d.name.startswith("tune_")],
        key=lambda d: d.stat().st_mtime,
    )
    if not runs:
        raise FileNotFoundError(f"Không tìm thấy tune run nào trong {tune_dir}")
    hyper_file = runs[-1] / "best_hyperparameters.yaml"
    if not hyper_file.exists():
        raise FileNotFoundError(f"Không tìm thấy best_hyperparameters.yaml trong {runs[-1]}")
    with open(hyper_file) as f:
        data = yaml.safe_load(f)
    print(f"  [HYPER] Loaded từ: {hyper_file}")
    return data


def find_latest_model(model_dir: Path) -> Path:
    pts = sorted(model_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    if not pts:
        raise FileNotFoundError(f"Không tìm thấy file .pt trong {model_dir}")
    return pts[-1]


def make_cls_weight_callback(data_yaml: str):
    """Trả về callback inject cls_weights vào BCE loss khi train bắt đầu.

    Dùng chung cho train.py và tune.py.
    cls_weights được đọc từ data.yaml để train và tune luôn nhất quán.
    """
    with open(data_yaml) as f:
        cls_weights = yaml.safe_load(f).get("cls_weights")

    def on_train_start(trainer):
        if not cls_weights:
            return
        try:
            import torch
            device = next(trainer.model.parameters()).device
            w = torch.tensor(cls_weights, dtype=torch.float32, device=device)
            trainer.criterion.bce = torch.nn.BCEWithLogitsLoss(
                pos_weight=w, reduction="none"
            )
            print(f"  cls_weights injected: {cls_weights}")
        except Exception as e:
            print(f"  Warning: cls_weights not applied — {e}")

    return on_train_start


# ── Visual prediction helpers (dùng chung cho predict / train / evaluate) ──────

def compute_iou(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    ax1, ay1, ax2, ay2 = box_a.tolist()
    bx1, by1, bx2, by2 = box_b.tolist()
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.0


def resolve_conflicts(
    cls_ids: torch.Tensor,
    confs: torch.Tensor,
    boxes: torch.Tensor,
    iou_thresh: float,
    conf_gap: float,
    class_priority: dict,
) -> list[int]:
    n    = len(cls_ids)
    keep = set(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if i not in keep or j not in keep:
                continue
            if compute_iou(boxes[i], boxes[j]) <= iou_thresh:
                continue
            conf_i, conf_j = confs[i].item(), confs[j].item()
            if abs(conf_i - conf_j) >= conf_gap:
                loser = j if conf_i > conf_j else i
            else:
                pri_i = class_priority.get(int(cls_ids[i]), 0)
                pri_j = class_priority.get(int(cls_ids[j]), 0)
                if pri_i != pri_j:
                    loser = j if pri_i > pri_j else i
                else:
                    loser = j if conf_i >= conf_j else i
            keep.discard(loser)
    return sorted(keep)


def draw_obb(img: np.ndarray, poly: np.ndarray, label: str, color: tuple) -> None:
    pts = poly.astype(int)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
    x_min = int(pts[:, 0].min())
    y_min = int(pts[:, 1].min())
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x_min, y_min - th - 6), (x_min + tw + 4, y_min), color, -1)
    cv2.putText(img, label, (x_min + 2, y_min - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def run_visual_predict(
    model,
    source_dir: Path,
    out_dir: Path,
    class_names: list,
    conf: float,
    class_conf: dict,
    class_priority: dict,
    class_color: dict,
    iou_thresh: float,
    conf_gap: float,
) -> None:
    """Chạy inference trên source_dir, vẽ OBB lên ảnh rồi lưu vào out_dir.

    Dùng chung cho train.py (visual sau train), evaluate.py, và predict.py.
    """
    if not source_dir.exists():
        print(f"  [SKIP] Không tìm thấy thư mục: {source_dir}")
        return

    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = [p for p in source_dir.iterdir() if p.suffix.lower() in image_exts]
    if not images:
        print(f"  [SKIP] Không có ảnh trong: {source_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    results = model.predict(source=str(source_dir), conf=conf, save=False, verbose=False)

    all_detections: list[str] = []
    for result in results:
        fname = Path(result.path).name
        obb   = result.obb
        img   = cv2.imread(result.path)

        if obb is None or len(obb) == 0:
            cv2.imwrite(str(out_dir / fname), img)
            continue

        keep_mask = [
            float(conf_val) >= class_conf.get(int(cls_id), 0.25)
            for cls_id, conf_val in zip(obb.cls, obb.conf)
        ]
        mask = torch.tensor(keep_mask)

        if not mask.any():
            cv2.imwrite(str(out_dir / fname), img)
            continue

        filtered_cls  = obb.cls[mask]
        filtered_conf = obb.conf[mask]
        filtered_xyxy = obb.xyxy[mask]
        filtered_poly = obb.xyxyxyxy[mask]

        keep_idx = resolve_conflicts(
            filtered_cls, filtered_conf, filtered_xyxy,
            iou_thresh, conf_gap, class_priority,
        )
        if not keep_idx:
            cv2.imwrite(str(out_dir / fname), img)
            continue

        for idx in keep_idx:
            cls_id   = int(filtered_cls[idx])
            box_conf = float(filtered_conf[idx])
            label    = f"{class_names[cls_id]} {box_conf:.2f}"
            color    = class_color.get(cls_id, (0, 255, 0))
            draw_obb(img, filtered_poly[idx].cpu().numpy(), label, color)
            all_detections.append(class_names[cls_id])

        cv2.imwrite(str(out_dir / fname), img)

    print(f"  [{source_dir.name}] {len(results)} ảnh  →  {out_dir}")
    if all_detections:
        for cls_name, count in Counter(all_detections).most_common():
            print(f"    {cls_name:<22} {count:>4} detection(s)")
    else:
        print(f"    (không có detection nào)")
