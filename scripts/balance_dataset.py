#  đếm tổng instances constructor_hat, random chọn những dòng cần xóa, ghi lại file label — ảnh giữ nguyên, chỉ label thay đổi.
import random
from pathlib import Path
from collections import Counter

RANDOM_SEED = 42
TARGET_COUNTS = {
    "train": {"bare_head": 57, "constructor_hat": 200, "non_safety_hat": 162},
    "val":   {"bare_head": 25, "constructor_hat": 80,  "non_safety_hat": 45},
    "test":  {"bare_head": 17, "constructor_hat": 80,  "non_safety_hat": 64},
}
CLASS_NAMES = {0: "bare_head", 1: "constructor_hat", 2: "non_safety_hat"}
DATASET_ROOT = Path(__file__).resolve().parent.parent / "split"


def collect_instances_by_class(split: str) -> dict[int, list[tuple[Path, int]]]:
    # Y như bên check class dis
    """
    Quét tất cả file label, thu thập từng instance (dòng) theo class.
    Trả về: {class_id: [(label_path, line_index), ...]}
    
    Ví dụ:
        input : split/labels/train/*.txt
        output: {1: [(Path("a.txt"), 0), (Path("a.txt"), 3), (Path("b.txt"), 1)]}
    """
    label_dir = DATASET_ROOT / "labels" / split
    # defaultdict(list) tự tạo list rỗng khi key chưa tồn tại
    instances: dict[int, list[tuple[Path, int]]] = {}

    for label_file in sorted(label_dir.glob("*.txt")):
        with open(label_file) as f:
            lines = f.readlines()

        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            class_id = int(line.split()[0])
            if class_id not in instances:
                instances[class_id] = []
            instances[class_id].append((label_file, line_idx))

    return instances


def undersample_class(split: str, class_id: int, target_n: int) -> None:
    """
    Xóa bớt dòng của class_id trong các file label cho đến khi còn target_n instances.
    File ảnh giữ nguyên.
    """
    instances = collect_instances_by_class(split)
    all_instances = instances.get(class_id, [])
    current_n = len(all_instances)

    if current_n <= target_n:
        print(f"  [{CLASS_NAMES[class_id]}] {current_n} <= {target_n}, bỏ qua")
        return

    random.seed(RANDOM_SEED)
    # Chọn ngẫu nhiên những dòng SẼ BỊ XÓA
    to_remove: set[tuple[Path, int]] = set(
        random.sample(all_instances, current_n - target_n)
    )
    print(f"  [{CLASS_NAMES[class_id]}] {current_n} → {target_n} (xóa {len(to_remove)} dòng)")

    # Nhóm các dòng cần xóa theo từng file
    # vì cần đọc-ghi từng file 1 lần thay vì nhiều lần
    remove_by_file: dict[Path, set[int]] = {}
    for label_path, line_idx in to_remove:
        if label_path not in remove_by_file:
            remove_by_file[label_path] = set()
        remove_by_file[label_path].add(line_idx)

    # Ghi lại từng file, bỏ các dòng bị xóa
    for label_path, remove_indices in remove_by_file.items():
        with open(label_path) as f:
            lines = f.readlines()

        # Giữ lại dòng không nằm trong remove_indices
        new_lines = [
            line for idx, line in enumerate(lines)
            if idx not in remove_indices
        ]

        with open(label_path, "w") as f:
            f.writelines(new_lines)


def main():
    for split, class_targets in TARGET_COUNTS.items():
        print(f"\n{'='*40}\nProcessing split: {split}")
        for cls_name, target_n in class_targets.items():
            # Tìm class_id từ tên
            class_id = next(k for k, v in CLASS_NAMES.items() if v == cls_name)
            undersample_class(split, class_id, target_n)

    print("\nDone!")


if __name__ == "__main__":
    main()