import sys
from pathlib import Path
from PIL import Image
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# Map class ID → tên class để log dễ đọc
CLASS_NAMES = {
    0: "casual_hat",
    1: "constructor_hat",
    2: "no_helmet"
}

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def get_classes(label_path: Path) -> list[int]:
    """
    Đọc file .txt YOLO, trả về list tất cả class ID trong ảnh.
    Mỗi dòng: class_id x_center y_center width height

    Input : Path("split/labels/train/anh1.txt")
    Output: [0, 1, 1]
    """
    lines = [l.strip() for l in label_path.read_text().splitlines() if l.strip()]
    if not lines:
        return []
    return [int(line.split()[0]) for line in lines]


class CustomHatDataset(Dataset):
    """
    Đọc ảnh + label từ split/images/{split} và split/labels/{split}.
    Mỗi ảnh trả về:
        image : tensor [3, 64, 64]
        labels: tensor [N] — tất cả class ID của N box trong ảnh
    """

    def __init__(self, root_dir: Path, split: str = "train", transform=None):
        self.transform = transform
        self.split = split
        self.samples = []

        image_dir = Path(root_dir) / "images" / split
        label_dir = Path(root_dir) / "labels" / split

        skipped = 0
        for img_path in sorted(image_dir.iterdir()):
            label_path = label_dir / f"{img_path.stem}.txt"

            if not label_path.exists():
                skipped += 1
                continue

            class_ids = get_classes(label_path)
            if not class_ids:
                skipped += 1
                continue

            self.samples.append((img_path, class_ids))

        # Log sau khi load xong
        all_class_ids = [cid for _, ids in self.samples for cid in ids]
        dist = Counter(all_class_ids)
        named_dist = {CLASS_NAMES[k]: v for k, v in sorted(dist.items())}

        print(f"\n{'─'*45}")
        print(f"  Split     : {split}")
        print(f"  Ảnh hợp lệ: {len(self.samples)}")
        print(f"  Bỏ qua    : {skipped} ảnh (không có label)")
        print(f"  Class dist : {named_dist}")
        print(f"{'─'*45}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_ids = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(class_ids)


def collate_fn(batch):
    images, labels = zip(*batch)
    return torch.stack(images), list(labels)


def verify_batch(images: torch.Tensor, labels: list, class_names: dict) -> None:
    """
    Log thông tin 1 batch để verify pipeline hoạt động đúng.
    Kiểm tra shape, pixel range sau normalize, và sample đầu tiên.
    """
    print(f"\n{'─'*45}")
    print(f"  Batch shape : {images.shape}")
    print(f"  Pixel range : [{images.min():.2f}, {images.max():.2f}]")  # phải ~ [-1, 1]
    print(f"  Số sample   : {len(labels)}")

    # Log 3 sample đầu để kiểm tra
    for i in range(min(3, len(labels))):
        class_ids = labels[i].tolist()
        named = [class_names[c] for c in class_ids]
        print(f"  Sample {i}    : {len(class_ids)} box | classes = {named}")
    print(f"{'─'*45}\n")


if __name__ == "__main__":
    split_dir = ROOT / "split"

    print("\n📦 Loading datasets...")
    train_dataset = CustomHatDataset(split_dir, split="train", transform=transform)
    val_dataset   = CustomHatDataset(split_dir, split="val",   transform=transform)
    test_dataset  = CustomHatDataset(split_dir, split="test",  transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              num_workers=0, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False,
                              num_workers=0, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False,
                              num_workers=0, collate_fn=collate_fn)

    print("\n🔍 Verifying train batch...")
    images, labels = next(iter(train_loader))
    verify_batch(images, labels, CLASS_NAMES)