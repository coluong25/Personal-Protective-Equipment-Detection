"""
Augment ảnh cho class casual_hat (class index cần khớp với helmet.yaml của bạn)
Cấu trúc thư mục:
  pj/split/images/train/
  pj/split/labels/train/

Chạy: python augment_casual_hat.py
"""

import cv2
import numpy as np
import shutil
from pathlib import Path
import albumentations as A  # pip install albumentations

# ─── CẤU HÌNH ────────────────────────────────────────────────────────────────
PROJECT_ROOT   = Path(r"C:\0\Project\01\Personal Protective Equipment Detection")
SPLIT_DIR      = PROJECT_ROOT / "split"
TRAIN_IMG_DIR  = SPLIT_DIR / "images" / "train"
TRAIN_LBL_DIR  = SPLIT_DIR / "labels" / "train"

CASUAL_HAT_CLASS_ID = 0        # ← Kiểm tra lại trong helmet.yaml của bạn
BACKGROUND_CLASS_ID = None     # class nào là background nếu có, None nếu không

TARGET_COUNT   = 60            # số ảnh casual_hat muốn có sau augment
COPY_PASTE_N   = 20            # số ảnh copy-paste tạo thêm
# ─────────────────────────────────────────────────────────────────────────────

def read_yolo_label(label_path: Path):
    """Đọc OBB label → list of [class_id, x1,y1,x2,y2,x3,y3,x4,y4]"""
    boxes = []
    if label_path.exists():
        for line in label_path.read_text().strip().splitlines():
            parts = line.strip().split()
            # OBB có 9 phần tử: class + 8 tọa độ
            if len(parts) == 9:
                boxes.append([int(parts[0])] + [float(x) for x in parts[1:]])
    return boxes


def obb_to_aabb(points):
    xs = points[0::2]
    ys = points[1::2]
    x1, x2 = max(0.0, min(xs)), min(1.0, max(xs))
    y1, y2 = max(0.0, min(ys)), min(1.0, max(ys))
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w  = x2 - x1
    h  = y2 - y1

    # Clip thêm lần 2: đảm bảo cx+w/2 và cy+h/2 không vượt 1.0
    w = min(w, 2 * cx, 2 * (1.0 - cx))   # w <= 2*cx và w <= 2*(1-cx)
    h = min(h, 2 * cy, 2 * (1.0 - cy))   # tương tự cho h
    return [cx, cy, w, h]

def obb_to_pixel(points, img_w, img_h):
    """
    OBB (8 số normalized) → pixel [x1,y1,x2,y2] thẳng luôn
    Bỏ qua bước trung gian AABB normalized
    Input : [x1,y1,x2,y2,x3,y3,x4,y4] normalized
    Output: [x1,y1,x2,y2] pixel
    """
    xs = [points[i] * img_w for i in range(0, 8, 2)]  # index 0,2,4,6 → nhân với width
    ys = [points[i] * img_h for i in range(1, 8, 2)]  # index 1,3,5,7 → nhân với height
    return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]


def write_yolo_label(label_path: Path, boxes: list):
    """Ghi OBB format: class x1y1 x2y2 x3y3 x4y4 (9 phần tử)"""
    lines = []
    for b in boxes:
        cid, cx, cy, w, h = b[0], b[1], b[2], b[3], b[4]
        x1, y1 = cx - w/2, cy - h/2
        x2, y2 = cx + w/2, cy - h/2
        x3, y3 = cx + w/2, cy + h/2
        x4, y4 = cx - w/2, cy + h/2
        lines.append(
            f"{cid} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} "
            f"{x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}"
        )
    label_path.write_text("\n".join(lines))
    
def find_casual_hat_images():
    result = []
    for img_path in TRAIN_IMG_DIR.glob("*.*"):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        lbl_path = TRAIN_LBL_DIR / (img_path.stem + ".txt")
        boxes = read_yolo_label(lbl_path)
        casual_boxes = [b for b in boxes if b[0] == CASUAL_HAT_CLASS_ID]
        if casual_boxes:
            result.append((img_path, lbl_path, boxes))
    return result

def yolo_to_pascal(box, img_w, img_h):
    """
    Chuyển YOLO format (cx,cy,w,h normalized) → Pascal VOC (x1,y1,x2,y2 pixel)
    Input : [cx, cy, w, h] mỗi giá trị trong [0,1]
    Output: [x1, y1, x2, y2] pixel
    """
    cx, cy, w, h = box
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return [max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)]


def pascal_to_yolo(box, img_w, img_h):
    """
    Chuyển Pascal VOC (x1,y1,x2,y2 pixel) → YOLO format (cx,cy,w,h normalized)
    """
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return [cx, cy, w, h]


# ─── PHẦN 1: AUGMENT TRỰC TIẾP ảnh gốc casual_hat ───────────────────────────

# Pipeline augmentation — giữ bối cảnh thực tế công trường
direct_aug = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.8),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.GaussNoise(p=0.3),
        A.MotionBlur(blur_limit=5, p=0.2),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.2, rotate_limit=10, p=0.7),
        A.RandomShadow(p=0.3),          # giả lập bóng đổ công trường
        A.CLAHE(p=0.2),                 # CLAHE (Contrast Limited Adaptive Histogram Equalization) — tăng độ rõ vùng tối
    ],
    bbox_params=A.BboxParams(
        format="yolo",                  # bbox đầu vào/ra đều là YOLO format
        label_fields=["class_ids"],
        min_visibility=0.3,             # loại bbox bị che >70%
    ),
)



def augment_direct(casual_images: list, target: int):
    
    """Tạo ảnh augment trực tiếp từ ảnh gốc casual_hat"""
    generated = 0
    idx = 0
    total_needed = target - len(casual_images)
    print(f"\n[DIRECT AUG] Cần tạo thêm {total_needed} ảnh từ {len(casual_images)} ảnh gốc")

    while generated < total_needed:
        img_path, lbl_path, boxes = casual_images[idx % len(casual_images)]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        bboxes    = [obb_to_aabb(b[1:]) for b in boxes]
        class_ids = [b[0] for b in boxes]
        valid = [
        (bb, cid) for bb, cid in zip(bboxes, class_ids)
        # cx+w/2 <= 1 và cy+h/2 <= 1
        if bb[0] + bb[2]/2 <= 1.0 and bb[1] + bb[3]/2 <= 1.0
        and bb[2] > 0 and bb[3] > 0   # ← thêm dòng này

        ]
        bboxes, class_ids = zip(*valid) if valid else ([], [])
        if valid:
            bboxes, class_ids = zip(*valid)
            bboxes = list(bboxes)
            class_ids = list(class_ids)
        else:
            bboxes, class_ids = [], []
    # Thêm dòng này vào đầu augment_direct(), trước khi gọi direct_aug
        try:
            augmented  = direct_aug(image=img, bboxes=bboxes, class_ids=class_ids)
            aug_img    = cv2.cvtColor(augmented["image"], cv2.COLOR_RGB2BGR)
            aug_boxes  = [
                [cid] + list(bb)
                for cid, bb in zip(augmented["class_ids"], augmented["bboxes"])
            ]

            # Đặt tên file mới tránh trùng
            new_stem  = f"{img_path.stem}_daug_{generated:04d}"
            new_img_p = TRAIN_IMG_DIR / f"{new_stem}.jpg"
            new_lbl_p = TRAIN_LBL_DIR / f"{new_stem}.txt"

            cv2.imwrite(str(new_img_p), aug_img)
            write_yolo_label(new_lbl_p, aug_boxes)
            generated += 1
            print(f"  ✓ {new_img_p.name}")
        except Exception as e:
            print(f"  ✗ Lỗi: {e}")

        idx += 1

    print(f"[DIRECT AUG] Hoàn thành: +{generated} ảnh")


def get_casual_hat_crops(casual_images: list):
    crops = []
    for img_path, _, boxes in casual_images:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        for b in boxes:
            if b[0] != CASUAL_HAT_CLASS_ID:
                continue
            aabb = obb_to_aabb(b[1:])          # OBB → AABB normalized
            x1, y1, x2, y2 = yolo_to_pascal(aabb, w, h)  # → pixel
            crop = img[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append((crop, aabb))     # lưu (ảnh crop, bbox AABB)
    return crops

def find_background_images(casual_stems: set):
    """
    Tìm ảnh công trường không có casual_hat để làm background copy-paste
    casual_stems: tên file ảnh gốc casual_hat — tránh paste vào chính nó
    """
    backgrounds = []
    for img_path in TRAIN_IMG_DIR.glob("*.*"):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        if img_path.stem in casual_stems:
            continue
        lbl_path = TRAIN_LBL_DIR / (img_path.stem + ".txt")
        boxes    = read_yolo_label(lbl_path)
        # Chỉ lấy ảnh KHÔNG có casual_hat
        has_casual = any(b[0] == CASUAL_HAT_CLASS_ID for b in boxes)
        if not has_casual:
            backgrounds.append((img_path, lbl_path, boxes))
    return backgrounds


def paste_crop_on_background(bg_img, crop, bg_boxes, scale_range=(0.08, 0.18)):
    """
    Dán crop lên vị trí ngẫu nhiên trong bg_img
    scale_range: crop sẽ chiếm bao nhiêu % chiều cao ảnh background
    """
    bg_h, bg_w = bg_img.shape[:2]
    target_h   = int(bg_h * np.random.uniform(*scale_range))
    ratio      = target_h / max(crop.shape[0], 1)
    target_w   = int(crop.shape[1] * ratio)

    if target_w < 10 or target_h < 10:
        return bg_img, bg_boxes, False

    resized = cv2.resize(crop, (target_w, target_h))

    # Chọn vị trí paste ngẫu nhiên — ưu tiên nửa trên (đầu người thường ở trên)
    max_x = bg_w - target_w
    max_y = int(bg_h * 0.6) - target_h   # giới hạn ở 60% chiều cao
    if max_x <= 0 or max_y <= 0:
        return bg_img, bg_boxes, False

    px = np.random.randint(0, max_x)
    py = np.random.randint(0, max(1, max_y))

    result = bg_img.copy()
    result[py:py+target_h, px:px+target_w] = resized   # ghi đè vùng background

    new_box_yolo = pascal_to_yolo([px, py, px+target_w, py+target_h], bg_w, bg_h)
    new_box      = [CASUAL_HAT_CLASS_ID] + new_box_yolo

    return result, bg_boxes + [new_box], True


def augment_copy_paste(casual_images: list, backgrounds: list, n: int):
    """Tạo n ảnh copy-paste"""
    crops = get_casual_hat_crops(casual_images)
    if not crops:
        print("[COPY-PASTE] Không tìm thấy crop nào!")
        return
    if not backgrounds:
        print("[COPY-PASTE] Không có ảnh background!")
        return

    print(f"\n[COPY-PASTE] Tạo {n} ảnh từ {len(crops)} crops × {len(backgrounds)} backgrounds")

    for i in range(n):
        crop, _    = crops[i % len(crops)]
        bg_path, bg_lbl, bg_boxes = backgrounds[i % len(backgrounds)]
        bg_img     = cv2.imread(str(bg_path))

        result_img, result_boxes, ok = paste_crop_on_background(bg_img, crop, bg_boxes)
        if not ok:
            print(f"  ✗ Skip {i}: không paste được")
            continue

        new_stem  = f"{bg_path.stem}_cp_{i:04d}"
        new_img_p = TRAIN_IMG_DIR / f"{new_stem}.jpg"
        new_lbl_p = TRAIN_LBL_DIR / f"{new_stem}.txt"

        cv2.imwrite(str(new_img_p), result_img)
        write_yolo_label(new_lbl_p, result_boxes)
        print(f"  ✓ {new_img_p.name}")

    print(f"[COPY-PASTE] Hoàn thành: +{n} ảnh")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("AUGMENT casual_hat — Direct + Copy-Paste")
    print("=" * 60)

    casual_images = find_casual_hat_images()
    print(f"Tìm thấy {len(casual_images)} ảnh gốc có casual_hat")

    if not casual_images:
        print("❌ Không tìm thấy ảnh casual_hat! Kiểm tra lại CASUAL_HAT_CLASS_ID")
        return

    # Phần 1: Direct augmentation
    augment_direct(casual_images, target=TARGET_COUNT)

    # Phần 2: Copy-paste
    casual_stems = {p.stem for p, _, _ in casual_images}
    backgrounds  = find_background_images(casual_stems)
    augment_copy_paste(casual_images, backgrounds, n=COPY_PASTE_N)

    print("\n✅ Xong! Train lại model và so sánh AP50 casual_hat.")


if __name__ == "__main__":
    main()
