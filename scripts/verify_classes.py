# Kiểm tra lại annotated xem đúng path với split không?
# scripts/verify_classes.py
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
label_dir = ROOT / "data" / "annotated" / "labels"

CLASS_NAMES = {
    0: "constructor_hat", 
    1: "no_helmet",
    2: "non_safety_hat",
}

# Đọc 5 file đầu tiên, in class ID và tên
for label_path in sorted(label_dir.iterdir())[:5]:
    lines = [l.strip() for l in label_path.read_text().splitlines() if l.strip()]
    class_ids = [int(l.split()[0]) for l in lines]
    named = [CLASS_NAMES[c] for c in class_ids]
    print(f"{label_path.name}: {class_ids} {named}")