"""
Augment ảnh cho các class thiếu instances (so sánh theo instances, không phải ảnh).
Cấu trúc thư mục:
  pj/data/split/images/train/
  pj/data/split/labels/train/

Chạy: python scripts/augment_split.py
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import albumentations as A

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
SPLIT_DIR     = PROJECT_ROOT / "data" / "split"
TRAIN_IMG_DIR = SPLIT_DIR / "images" / "train"
TRAIN_LBL_DIR = SPLIT_DIR / "labels" / "train"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from config import AUGMENT

TARGET_COUNT = AUGMENT["target_count"]
COPY_PASTE_N = AUGMENT["copy_paste_n"]
AUG_CLASSES  = AUGMENT["classes"]


# ── I/O helpers ───────────────────────────────────────────────────────────────

def read_yolo_label(label_path: Path) -> list:
    boxes = []
    if label_path.exists():
        for line in label_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) == 9:
                boxes.append([int(parts[0])] + [float(x) for x in parts[1:]])
    return boxes


def write_yolo_label(label_path: Path, boxes: list):
    lines = []
    for b in boxes:
        vals = " ".join(f"{v:.6f}" for v in b[1:])
        lines.append(f"{int(b[0])} {vals}")
    label_path.write_text("\n".join(lines))


# ── Counting ──────────────────────────────────────────────────────────────────

def count_instances(class_id: int) -> int:
    count = 0
    for f in TRAIN_LBL_DIR.glob("*.txt"):
        for line in f.read_text().strip().splitlines():
            parts = line.strip().split()
            if parts and int(parts[0]) == class_id:
                count += 1
    return count


def find_images_with_class(class_id: int) -> list:
    result = []
    for img_path in TRAIN_IMG_DIR.glob("*.*"):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        lbl_path = TRAIN_LBL_DIR / (img_path.stem + ".txt")
        boxes = read_yolo_label(lbl_path)
        if any(b[0] == class_id for b in boxes):
            result.append((img_path, lbl_path, boxes))
    return result


# ── Geometry helpers ──────────────────────────────────────────────────────────

def obb_to_aabb(points):
    xs = points[0::2]
    ys = points[1::2]
    x1, x2 = max(0.0, min(xs)), min(1.0, max(xs))
    y1, y2 = max(0.0, min(ys)), min(1.0, max(ys))
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w  = min(x2 - x1, 2 * cx, 2 * (1.0 - cx))
    h  = min(y2 - y1, 2 * cy, 2 * (1.0 - cy))
    return [cx, cy, w, h]


def yolo_to_pascal(box, img_w, img_h):
    cx, cy, w, h = box
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return [max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)]


def pascal_to_yolo(box, img_w, img_h):
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return [cx, cy, w, h]


# ── Augmentation pipeline ─────────────────────────────────────────────────────

direct_aug = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.8),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=25, val_shift_limit=15, p=0.5),
        A.GaussNoise(std_range=(0.012, 0.028), p=0.3),
        A.MotionBlur(blur_limit=3, p=0.2),
        A.Affine(
            translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
            scale=(0.9, 1.1),
            rotate=(-10, 10),
            p=0.4,
        ),
        A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=0.3),
    ],
    keypoint_params=A.KeypointParams(
        format="xy",
        label_fields=["box_idx"],
        remove_invisible=True,
    ),
)


def augment_direct(class_id: int, source_images: list, target: int):
    current = count_instances(class_id)
    if current >= target:
        print(f"  [class {class_id}] {current} instances >= {target}, bỏ qua.")
        return

    total_needed = target - current
    added = 0
    idx = 0
    MAX_ATTEMPTS = total_needed * 15
    print(f"\n[DIRECT AUG] class {class_id}: {current} → {target} (cần thêm ~{total_needed} instances)")

    while added < total_needed and idx < MAX_ATTEMPTS:
        img_path, _, boxes = source_images[idx % len(source_images)]
        img = cv2.imread(str(img_path))
        if img is None:
            idx += 1
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        keypoints  = []
        kp_box_idx = []
        box_meta   = []
        for bi, b in enumerate(boxes):
            coords = b[1:]
            for i in range(4):
                kp_x = max(0.0, min(w * 0.999, coords[i*2]     * w))
                kp_y = max(0.0, min(h * 0.999, coords[i*2 + 1] * h))
                keypoints.append((kp_x, kp_y))
                kp_box_idx.append(float(bi))
            box_meta.append(b[0])

        try:
            augmented   = direct_aug(image=img, keypoints=keypoints, box_idx=kp_box_idx)
            aug_img     = cv2.cvtColor(augmented["image"], cv2.COLOR_RGB2BGR)
            aug_kps     = augmented["keypoints"]
            aug_box_idx = augmented["box_idx"]
            aug_h, aug_w = aug_img.shape[:2]

            kp_by_box = defaultdict(list)
            for (kx, ky), bi in zip(aug_kps, aug_box_idx):
                kp_by_box[int(bi)].append((kx, ky))

            aug_boxes = []
            for bi, cid in enumerate(box_meta):
                corners = kp_by_box[bi]
                if len(corners) < 4:
                    continue
                norm_coords = []
                for (kx, ky) in corners:
                    norm_coords.extend([
                        max(0.0, min(0.9999, kx / aug_w)),
                        max(0.0, min(0.9999, ky / aug_h)),
                    ])
                xs = norm_coords[0::2]
                ys = norm_coords[1::2]
                if (max(xs) - min(xs)) * (max(ys) - min(ys)) < 0.001:
                    continue
                aug_boxes.append([cid] + norm_coords)

            if not aug_boxes:
                idx += 1
                continue

            # đếm instances của class_id trong ảnh augment này
            new_instances = sum(1 for b in aug_boxes if b[0] == class_id)
            if new_instances == 0:
                idx += 1
                continue

            stem      = f"{img_path.stem}_daug_{class_id}_{added:04d}"
            new_img_p = TRAIN_IMG_DIR / f"{stem}.jpg"
            new_lbl_p = TRAIN_LBL_DIR / f"{stem}.txt"

            cv2.imwrite(str(new_img_p), aug_img)
            write_yolo_label(new_lbl_p, aug_boxes)
            added += new_instances
            print(f"  ✓ {new_img_p.name}  (+{new_instances} instances, total added: {added}/{total_needed})")

        except Exception as e:
            print(f"  ✗ Lỗi: {e}")

        idx += 1

    print(f"[DIRECT AUG] class {class_id}: hoàn thành +{added} instances")


# ── Copy-paste (chỉ cho class 0 - bare_head) ──────────────────────────────────

def get_class_crops(source_images: list, class_id: int) -> list:
    crops = []
    for img_path, _, boxes in source_images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        for b in boxes:
            if b[0] != class_id:
                continue
            aabb = obb_to_aabb(b[1:])
            x1, y1, x2, y2 = yolo_to_pascal(aabb, w, h)
            crop = img[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)
    return crops


def find_background_images(class_id: int, exclude_stems: set) -> list:
    backgrounds = []
    for img_path in TRAIN_IMG_DIR.glob("*.*"):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        if img_path.stem in exclude_stems:
            continue
        lbl_path = TRAIN_LBL_DIR / (img_path.stem + ".txt")
        boxes = read_yolo_label(lbl_path)
        if not any(b[0] == class_id for b in boxes):
            backgrounds.append((img_path, lbl_path, boxes))
    return backgrounds


def paste_crop_on_background(bg_img, crop, bg_boxes, class_id, scale_range=(0.08, 0.18)):
    bg_h, bg_w = bg_img.shape[:2]
    target_h = int(bg_h * np.random.uniform(*scale_range))
    ratio    = target_h / max(crop.shape[0], 1)
    target_w = int(crop.shape[1] * ratio)

    if target_w < 10 or target_h < 10:
        return bg_img, bg_boxes, False

    resized = cv2.resize(crop, (target_w, target_h))
    max_x = bg_w - target_w
    max_y = int(bg_h * 0.6) - target_h
    if max_x <= 0 or max_y <= 0:
        return bg_img, bg_boxes, False

    px = np.random.randint(0, max_x)
    py = np.random.randint(0, max(1, max_y))

    result = bg_img.copy()
    result[py:py+target_h, px:px+target_w] = resized

    cx, cy, w, h = pascal_to_yolo([px, py, px+target_w, py+target_h], bg_w, bg_h)
    x1, y1 = cx - w/2, cy - h/2
    x2, y2 = cx + w/2, cy - h/2
    x3, y3 = cx + w/2, cy + h/2
    x4, y4 = cx - w/2, cy + h/2
    new_box = [class_id, x1, y1, x2, y2, x3, y3, x4, y4]

    return result, bg_boxes + [new_box], True


def augment_copy_paste(class_id: int, source_images: list, n: int):
    crops = get_class_crops(source_images, class_id)
    if not crops:
        print(f"[COPY-PASTE] Không tìm thấy crop nào cho class {class_id}!")
        return

    exclude_stems = {p.stem for p, _, _ in source_images}
    backgrounds   = find_background_images(class_id, exclude_stems)
    if not backgrounds:
        print(f"[COPY-PASTE] Không có ảnh background cho class {class_id}!")
        return

    print(f"\n[COPY-PASTE] class {class_id}: tạo {n} ảnh từ {len(crops)} crops × {len(backgrounds)} backgrounds")

    for i in range(n):
        crop     = crops[i % len(crops)]
        bg_path, _, bg_boxes = backgrounds[i % len(backgrounds)]
        bg_img = cv2.imread(str(bg_path))
        if bg_img is None:
            continue

        result_img, result_boxes, ok = paste_crop_on_background(bg_img, crop, bg_boxes, class_id)
        if not ok:
            print(f"  ✗ Skip {i}")
            continue

        stem      = f"{bg_path.stem}_cp_{class_id}_{i:04d}"
        new_img_p = TRAIN_IMG_DIR / f"{stem}.jpg"
        new_lbl_p = TRAIN_LBL_DIR / f"{stem}.txt"

        cv2.imwrite(str(new_img_p), result_img)
        write_yolo_label(new_lbl_p, result_boxes)
        print(f"  ✓ {new_img_p.name}")

    print(f"[COPY-PASTE] class {class_id}: hoàn thành +{n} ảnh")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(f"AUGMENT classes {AUG_CLASSES} → target {TARGET_COUNT} instances mỗi class")
    print("=" * 60)

    for class_id in AUG_CLASSES:
        current = count_instances(class_id)
        print(f"\nClass {class_id}: {current} instances hiện tại")

        if current >= TARGET_COUNT:
            print(f"  Đã đủ {TARGET_COUNT}, bỏ qua.")
            continue

        source_images = find_images_with_class(class_id)
        print(f"  Tìm thấy {len(source_images)} ảnh nguồn chứa class {class_id}")

        if not source_images:
            print(f"  ❌ Không tìm thấy ảnh nguồn! Kiểm tra lại class ID.")
            continue

        augment_direct(class_id, source_images, TARGET_COUNT)

        if class_id == 0:
            augment_copy_paste(class_id, source_images, COPY_PASTE_N)

    print("\n✅ Xong! Chạy check_class_distribution.py để kiểm tra kết quả.")


if __name__ == "__main__":
    main()
