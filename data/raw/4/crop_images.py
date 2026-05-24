from PIL import Image
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent.parent.parent
input_folder = Path(__file__).resolve().parent
output_folder = ROOT / "data" / "raw"/ "ai_gen"
output_folder.mkdir(exist_ok=True)

# Lấy danh sách tên gốc đã crop rồi
done = {p.stem.rsplit("_", 2)[0] for p in output_folder.glob("*.png")}

for img_path in input_folder.glob("*.[pj][np][gg]"):  # khớp .png và .jpg
    if img_path.stem in done:
        print(f"Bỏ qua: {img_path.name}")
        continue

    img = Image.open(img_path)
    w, h = img.size

    crops = {
        "top_left":     img.crop((0,    0,    w//2, h//2)),
        "top_right":    img.crop((w//2, 0,    w,    h//2)),
        "bottom_left":  img.crop((0,    h//2, w//2, h)),
        "bottom_right": img.crop((w//2, h//2, w,    h)),
    }

    for name, crop in crops.items():
        crop.save(output_folder / f"{img_path.stem}_{name}.png")