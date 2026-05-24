# Chạy thẳng trong terminal
from pathlib import Path

ROOT = Path(r"C:\0\Project\01\Personal Protective Equipment Detection")

for split in ["train", "val", "test"]:
    label_dir = ROOT / "split" / "labels" / split
    bad = []
    for p in label_dir.iterdir():
        for line in p.read_text().splitlines():
            if line.strip() and len(line.split()) != 9:
                bad.append((p.name, len(line.split()), line))
    
    status = "✅" if not bad else f"❌ {len(bad)} dòng sai"
    print(f"{split:5s}: {status}")
    for name, count, line in bad[:3]:
        print(f"       {name}: {count} số → '{line[:60]}'")