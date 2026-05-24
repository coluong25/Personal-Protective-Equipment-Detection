from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
label_dir = ROOT / "split" / "labels"/"train"
image_dir = ROOT / "split" / "images"/"train"

removed = 0
for p in list(label_dir.iterdir()) + list(image_dir.iterdir()):
    # _daug_ và _cp_ là suffix Roboflow thêm vào khi augment
    if "_daug_" in p.name or "_cp_" in p.name:
        p.unlink()   # xóa file
        removed += 1
        print(f"  🗑  {p.name}")

print(f"\n✅ Đã xóa {removed} file augmented")
print(f"   Chạy lại split_dataset_stratified() sau khi xóa")