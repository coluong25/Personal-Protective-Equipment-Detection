"""
fix_labels.py
Đọc tất cả file .txt YOLO label, sửa class_id (cột đầu) thành int, ghi đè tại chỗ.
Sau đó in phân phối class theo từng split.
"""

from pathlib import Path
from collections import Counter


# ─── CONFIG ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
SPLIT_ROOT = ROOT /"split"        
CLASS_NAMES = ["bare_head", "constructor_hat", "non_safety_hat"] 
SPLITS      = ["train", "val", "test"]
# ───────────────────────────────────────────────────────────


def fix_label_file(label_path: Path) -> int:
    """
    Đọc file, ép cột 0 về int, ghi đè nếu có dòng bị sai.
    Trả về số dòng đã sửa.
    """
    lines     = label_path.read_text().splitlines()
    fixed     = []
    n_changed = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts    = line.split()
        # int(float(...)) xử lý cả '0' lẫn '0.0'
        class_id = int(float(parts[0]))
        new_line = " ".join([str(class_id)] + parts[1:])

        if new_line != line:
            n_changed += 1

        fixed.append(new_line)

    label_path.write_text("\n".join(fixed) + "\n")
    return n_changed


def fix_all_labels(split_root: Path) -> dict[str, int]:
    """Duyệt toàn bộ split, trả về dict {split: số_dòng_đã_sửa}."""
    summary = {}

    for split in SPLITS:
        label_dir = split_root / "labels" / split
        if not label_dir.exists():
            print(f"[SKIP] Không tìm thấy: {label_dir}")
            continue

        total_changed = 0
        label_files   = list(label_dir.glob("*.txt"))

        for lf in label_files:
            total_changed += fix_label_file(lf)

        summary[split] = total_changed
        print(f"[{split:5}] {len(label_files):4} files | {total_changed} dòng được sửa")

    return summary


def check_class_distribution(split_root: Path, class_names: list[str]):
    """In bảng phân phối class sau khi đã fix."""
    split_root = Path(split_root)

    if not split_root.exists():
        raise FileNotFoundError(f"Không tìm thấy: {split_root}")

    results = {}

    for split in SPLITS:
        label_dir = split_root / "labels" / split
        if not label_dir.exists():
            continue

        counter = Counter()
        for label_file in label_dir.glob("*.txt"):
            with open(label_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    class_id = int(float(line.split()[0]))
                    counter[class_id] += 1

        results[split] = counter

    # ── In bảng ──
    max_id = max(
        (max(c.keys()) for c in results.values() if c),
        default=len(class_names) - 1,
    )
    col_w = 10

    header = f"{'Class':<20}" + "".join(f"{s:>{col_w}}" for s in results)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for cid in range(max_id + 1):
        name = class_names[cid] if cid < len(class_names) else f"class_{cid}"
        row  = f"{name:<20}" + "".join(
            f"{results[s].get(cid, 0):>{col_w}}" for s in results
        )
        print(row)

    print("=" * len(header))
    total_row = f"{'TOTAL':<20}" + "".join(
        f"{sum(results[s].values()):>{col_w}}" for s in results
    )
    print(total_row)


# ─── MAIN ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("── Bắt đầu fix label files ──")
    fix_all_labels(SPLIT_ROOT)

    print("\n── Phân phối class sau khi fix ──")
    check_class_distribution(SPLIT_ROOT, CLASS_NAMES)
