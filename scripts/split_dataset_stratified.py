import shutil
import numpy as np
from pathlib import Path
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
# pj/scripts/strat_split.py → .parent.parent = pj/


def get_class_vector(label_path: Path, num_classes: int = 3) -> list[int]:
    """
    Input : pj/data/annotated/labels/anh1.txt
    Output: [1, 1, 0]  — ảnh có class 0 và 1, không có class 2
            [0, 0, 0]  — label trống = background sample (mũ không có người)
    """
    lines = [l.strip() for l in label_path.read_text().splitlines() if l.strip()]
    # [OBB label1, label2]
    if not lines:
        return [0] * num_classes  # background: giữ ảnh, label trống

    present = {int(l.split()[0]) for l in lines}
    return [1 if i in present else 0 for i in range(num_classes)]
# Strat split cần vị trí cố định => vector(thứ tự vào)

def split_dataset_multilabel(
    img_dir: Path = ROOT / "data" / "annotated" / "images",
    lbl_dir: Path = ROOT / "data" / "annotated" / "labels",
    out_dir: Path = ROOT /"data"/ "split",
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    num_classes: int = 3,
    random_state: int = 42,
) -> None:
    assert train_ratio + val_ratio < 1.0, "train + val phải < 1.0"

    # ── 1. Build pairs + label matrix ───────────────────────────────
    paired, label_matrix = [], []

    for img_path in sorted(img_dir.iterdir()): # iterdir duyệt file bên trong thư mục
        lbl_path = lbl_dir / (img_path.stem + ".txt") #=> path.steam lấy tên không đuôi
        if not lbl_path.exists():
            print(f"[SKIP] Không có label: {lbl_path.name}") # => lấy tên có đuôi
            continue

        vec = get_class_vector(lbl_path, num_classes) # => list class theo thứ tự
        if all(v == 0 for v in vec):
            print(f"[BG]   Label trống (background): {lbl_path.name}")

        paired.append(img_path.name)
        label_matrix.append(vec)

    X = np.array(paired) # => tên ảnh 
    y = np.array(label_matrix) # => [[0,0,1], [1,1,1]]
    print(f"✅ Tổng ảnh hợp lệ: {len(X)}")

    # ── 2. Tách train / temp ─────────────────────────────────────────
    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=1 - train_ratio,
        random_state=random_state
    )
    train_idx, temp_idx = next(msss.split(X, y))

    # ── 3. Tách val / test từ temp ───────────────────────────────────
    test_ratio_in_temp = (1 - train_ratio - val_ratio) / (1 - train_ratio)
    msss2 = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=test_ratio_in_temp,
        random_state=random_state
    )
    val_idx, test_idx = next(msss2.split(X[temp_idx], y[temp_idx]))

    # index tương đối của temp → index tuyệt đối của X
    val_idx  = temp_idx[val_idx]
    test_idx = temp_idx[test_idx]

    # ── 4. Copy file vào pj/split/ ───────────────────────────────────
    splits = {"train": train_idx, "val": val_idx, "test": test_idx}

    for split_name, indices in splits.items():
        img_out = out_dir / "images" / split_name
        lbl_out = out_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for name in X[indices]:
            src_lbl = lbl_dir / (Path(name).stem + ".txt")
            shutil.copy(img_dir / name, img_out / name)
            shutil.copy(src_lbl, lbl_out / src_lbl.name)

        print(f"  {split_name:5s}: {len(indices)} ảnh → {img_out}")

    print("\n✅ Split hoàn tất!")


if __name__ == "__main__":
    split_dataset_multilabel()