from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt

def check_class_distribution(split_root: str | Path, class_names: list[str]):
    # ✅ Ép về Path trước mọi thao tác
    split_root = Path(split_root)
    
    if not split_root.exists():
        raise FileNotFoundError(f"Không tìm thấy: {split_root}")

    splits = ["train", "val", "test"]
    results = {}

    for split in splits:
        label_dir = split_root / "labels" / split
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

        print(f"\n📂 {split.upper()}")
        total = sum(counter.values())
        for class_id, count in sorted(counter.items()):
            name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            ratio = count / total * 100
            print(f"  {name:>15}: {count:>5} instances  ({ratio:.1f}%)")

    fig, axes = plt.subplots(1, len(splits), figsize=(14, 4))
    for ax, split in zip(axes, splits):
        counter = results[split]
        labels = [class_names[i] for i in sorted(counter)]
        values = [counter[i] for i in sorted(counter)]
        ax.bar(labels, values)  # bỏ hardcode màu
        ax.set_title(split)
        ax.set_ylabel("Số instances")
        for i, v in enumerate(values):
            ax.text(i, v + 1, str(v), ha="center", fontsize=9)

    plt.suptitle("Class Distribution across Splits", fontweight="bold")
    plt.tight_layout()

    out_dir = split_root.parent / "split_distribution"
    out_dir.mkdir(parents=True, exist_ok=True)

    png_path = out_dir / "class_distribution.png"
    plt.savefig(png_path, dpi=150)
    plt.show()
    print(f"\n✅ Đã lưu biểu đồ : {png_path}")

    lines = ["Class Distribution across Splits", "=" * 50]
    for split in splits:
        counter = results[split]
        total = sum(counter.values())
        n_images = sum(1 for _ in (split_root / "labels" / split).glob("*.txt"))
        lines.append(f"\n{split.upper()} ({n_images} images, {total} instances)")
        lines.append("-" * 40)
        for class_id in sorted(counter):
            name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            ratio = counter[class_id] / total * 100
            lines.append(f"  {name:<20}: {counter[class_id]:>5}  ({ratio:.1f}%)")

    txt_path = out_dir / "class_distribution.txt"
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ Đã lưu số liệu  : {txt_path}")


check_class_distribution(
    split_root=Path(__file__).resolve().parent.parent / "data" / "split",  
    class_names=["bare_head", "constructor_hat", "wrong_headgear_type"]
)