# test_external.py — đặt ngang hàng với models/ và data/
from pathlib import Path
from collections import Counter
from ultralytics import YOLO
import sys
import os
import torch
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

BEST_PT = ROOT / "models" / "final"

# ── Biến môi trường (environment variable) ──────────────────────────
EXTERNAL_TEST = Path(
    os.getenv("EXTERNAL_TEST_DIR", str(ROOT / "data" / "test" / "images"))
)

TEST_OUT = Path(
    os.getenv("TEST_OUT_DIR", str(ROOT / "outputs" / "test_external"))
)

# ── Cấu hình ngưỡng confidence theo từng class ──────────────────────
# bare_head threshold cao hơn để giảm false alarm (cảnh báo sai)
CLASS_CONF = {
    0: 0.2,  # bare_head       — đầu trần
    1: 0.62,  # constructor_hat — mũ bảo hộ xây dựng
    2: 0.56,  # non_safety_hat  — mũ không đạt chuẩn
}

# ── Thứ tự ưu tiên khi 2 box chồng nhau (overlap) ───────────────────
# constructor_hat > non_safety_hat > bare_head
# Lý do: ưu tiên nhận diện mũ hơn là báo lỗi "đầu trần" sai
CLASS_PRIORITY = {
    0: 1,  # bare_head
    1: 3,  # constructor_hat
    2: 2,  # non_safety_hat
}

# ── Màu vẽ box theo class (BGR) ──────────────────────────────────────
CLASS_COLOR = {
    0: (0, 0, 255),    # bare_head       — đỏ
    1: (0, 255, 0),    # constructor_hat — xanh lá
    2: (0, 165, 255),  # non_safety_hat  — cam
}


# ────────────────────────────────────────────────────────────────────
def find_latest_model(model_dir: Path) -> Path:
    """Tìm file .pt mới nhất trong thư mục model_dir dựa theo thời gian sửa đổi"""
    pts = sorted(model_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    if not pts:
        raise FileNotFoundError(f"Không tìm thấy file .pt trong {model_dir}")
    return pts[-1]


def compute_iou(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    """
    Tính IoU (Intersection over Union — tỉ lệ vùng giao / vùng hợp)
    giữa 2 bounding box dạng xyxy: (x1, y1, x2, y2)

    Input : 2 tensor shape (4,)
    Output: float trong [0, 1]
    """
    ax1, ay1, ax2, ay2 = box_a.tolist()
    bx1, by1, bx2, by2 = box_b.tolist()

    # Tọa độ vùng giao (intersection)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter   = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union   = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def resolve_conflicts(
    cls_ids: torch.Tensor,
    confs:   torch.Tensor,
    boxes:   torch.Tensor,
    iou_thresh: float = 0.45,
    conf_gap:   float = 0.15,
) -> list[int]:
    """
    Loại bỏ detection xung đột: 2 box chồng nhau (IoU > iou_thresh)
    thuộc các class đối lập (vd: bare_head vs constructor_hat).

    Quy tắc giữ lại:
      1. Nếu confidence chênh lệch >= conf_gap → tin box có conf cao hơn
      2. Nếu không → giữ box có CLASS_PRIORITY cao hơn
      3. Nếu priority bằng nhau → giữ conf cao hơn

    Input : tensor cls_ids, confs, boxes — đã filtered qua CLASS_CONF
    Output: list index cần giữ lại
    """
    n    = len(cls_ids)
    keep = set(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            if i not in keep or j not in keep:
                continue

            iou = compute_iou(boxes[i], boxes[j])
            if iou <= iou_thresh:
                continue

            conf_i = confs[i].item()
            conf_j = confs[j].item()
            gap    = abs(conf_i - conf_j)

            if gap >= conf_gap:
                # Tin vào box model chắc hơn
                loser = j if conf_i > conf_j else i
            else:
                pri_i = CLASS_PRIORITY.get(int(cls_ids[i]), 0)
                pri_j = CLASS_PRIORITY.get(int(cls_ids[j]), 0)
                if pri_i != pri_j:
                    loser = j if pri_i > pri_j else i
                else:
                    loser = j if conf_i >= conf_j else i

            keep.discard(loser)

    return sorted(keep)


def draw_obb(
    img:     np.ndarray,
    poly:    np.ndarray,
    label:   str,
    color:   tuple,
) -> None:
    """
    Vẽ OBB (Oriented Bounding Box — bounding box có góc xoay) lên ảnh.

    poly : shape (4, 2) — 4 điểm góc (x, y) của box thực
    Dùng cv2.polylines thay rectangle để giữ đúng góc xoay
    """
    pts = poly.astype(int)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)

    # Vẽ label tại góc trên-trái của polygon
    x_min = int(pts[:, 0].min())
    y_min = int(pts[:, 1].min())
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x_min, y_min - th - 6), (x_min + tw + 4, y_min), color, -1)
    cv2.putText(
        img, label,
        (x_min + 2, y_min - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
    )


# ────────────────────────────────────────────────────────────────────
def test_external(conf: float = 0.05) -> None:
    model_path = find_latest_model(BEST_PT)

    if not EXTERNAL_TEST.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục test: {EXTERNAL_TEST}")

    print(f"\n{'─'*55}")
    print(f"  Model : {model_path.name}")
    print(f"  Source: {EXTERNAL_TEST}")
    print(f"  Output: {TEST_OUT}")
    print(f"{'─'*55}\n")

    model   = YOLO(str(model_path))
    results = model.predict(
        source  = str(EXTERNAL_TEST),
        conf    = conf,        # conf thấp để lọc thủ công qua CLASS_CONF
        save    = False,
        verbose = False,
    )

    save_dir = TEST_OUT / model_path.stem
    save_dir.mkdir(parents=True, exist_ok=True)

    all_detections: list[str] = []  # Dùng để in summary cuối

    for result in results:
        fname = Path(result.path).name
        obb   = result.obb  # OBB — Oriented Bounding Box

        if obb is None or len(obb) == 0:
            print(f"{fname} — không detect được object nào")
            continue

        # ── Bước 1: Lọc theo CLASS_CONF per-class ───────────────────
        keep_mask = [
            conf_val >= CLASS_CONF.get(int(cls_id), 0.25)
            for cls_id, conf_val in zip(obb.cls, obb.conf)
        ]
        mask = torch.tensor(keep_mask)

        if not mask.any():
            print(f"{fname} — tất cả box dưới ngưỡng CLASS_CONF")
            continue

        filtered_cls  = obb.cls[mask]
        filtered_conf = obb.conf[mask]
        filtered_xyxy = obb.xyxy[mask]       # dùng để tính IoU (axis-aligned)

        # obb.xyxyxyxy: shape (N, 4, 2) — 4 điểm góc thực của OBB
        filtered_poly = obb.xyxyxyxy[mask]

        # ── Bước 2: Resolve conflict — loại box chồng xung đột ──────
        keep_idx = resolve_conflicts(filtered_cls, filtered_conf, filtered_xyxy)

        if not keep_idx:
            print(f"{fname} — tất cả box bị loại sau conflict resolution")
            continue

        final_cls  = filtered_cls[keep_idx]
        final_conf = filtered_conf[keep_idx]
        final_poly = filtered_poly[keep_idx]

        # ── Bước 3: Vẽ OBB lên ảnh ──────────────────────────────────
        img = result.plot(boxes=False)  # ảnh gốc + label YOLO, không có box

        for poly, cls_id, box_conf in zip(final_poly, final_cls, final_conf):
            cls_int = int(cls_id)
            name    = result.names[cls_int]
            label   = f"{name} {box_conf:.2f}"
            color   = CLASS_COLOR.get(cls_int, (0, 255, 0))

            # poly.cpu().numpy() — shape (4, 2)
            draw_obb(img, poly.cpu().numpy(), label, color)

        out_path = save_dir / fname
        cv2.imwrite(str(out_path), img)

        # ── In kết quả từng ảnh ─────────────────────────────────────
        print(f"\n{fname}")
        for cls_id, box_conf in zip(final_cls, final_conf):
            name = result.names[int(cls_id)]
            print(f"  {name:<20} conf={box_conf:.2f}")
            all_detections.append(name)

    # ── Summary toàn bộ test set ─────────────────────────────────────
    print(f"\n{'─'*55}")
    print("📊 Summary toàn bộ test set:")
    if all_detections:
        for cls_name, count in Counter(all_detections).most_common():
            print(f"  {cls_name:<20} {count:>4} detection(s)")
    else:
        print("  Không có detection nào được giữ lại.")
    print(f"{'─'*55}")
    print(f"\n✅ Results saved: {save_dir}")


if __name__ == "__main__":
    test_external()