import os
from pathlib import Path
# Kiểm tra số ảnh và label phải khớp nhau
def verify_split(data_root: str):
    splits = ["train", "val", "test"]
    for split in splits:
        img_dir = Path(data_root) / "images" / split
        lbl_dir = Path(data_root) / "labels" / split
        
        imgs = list(img_dir.glob("*"))
        lbls = list(lbl_dir.glob("*.txt"))
        
        print(f"{split}: {len(imgs)} images, {len(lbls)} labels")
        assert len(imgs) == len(lbls), f"Mismatch in {split}!"

verify_split("../split")