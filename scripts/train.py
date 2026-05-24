# scripts/train.py
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

ROOT    = Path(__file__).resolve().parent.parent
DATA    = ROOT / "data.yaml"
MODELS  = ROOT / "models"

# ── Config ───────────────────────────────────────────────────────────
CFG = {
    "model"      : "yolov8s-obb.pt",      # pretrained weights tự download
    "data"       : str(DATA),
    "epochs"     : 50,
    "imgsz"      : 640,               # chuẩn YOLO, không dùng 64x64 nữa
    "batch"      : 16,                # giảm xuống 8 nếu GPU yếu
    "patience"   : 10,                # early stopping — dừng nếu 10 epoch không cải thiện
    "device"     : 0,                 # 0 = GPU đầu tiên | "cpu" nếu không có GPU
    "workers"    : 0,                 # Windows cần 0
    "seed"       : 42,
}

# ── Run name theo timestamp để không ghi đè checkpoint cũ ───────────
RUN_NAME = f"yolov8s_{datetime.now().strftime('%Y%m%d_%H%M')}"


def train() -> None:
    print(f"\n{'─'*50}")
    print(f"  Model   : {CFG['model']}")
    print(f"  Data    : {CFG['data']}")
    print(f"  Epochs  : {CFG['epochs']}  |  Batch: {CFG['batch']}")
    print(f"  Run     : {RUN_NAME}")
    print(f"{'─'*50}\n")

    model = YOLO(CFG["model"])

    # train() trả về Results object chứa metrics sau khi xong
    results = model.train(
        data      = CFG["data"],
        epochs    = CFG["epochs"],
        imgsz     = CFG["imgsz"],
        batch     = CFG["batch"],
        patience  = CFG["patience"],
        device    = CFG["device"],
        workers   = CFG["workers"],
        seed      = CFG["seed"],
        project   = str(MODELS / "checkpoints"),  # folder lưu checkpoint
        name      = RUN_NAME,                      # subfolder theo run
        save      = True,                          # save best.pt + last.pt
        val       = True,                          # chạy val sau mỗi epoch
    )

    # Copy best.pt sang models/final/
    best_src = MODELS / "checkpoints" / RUN_NAME / "weights" / "best.pt"
    final_dir = MODELS / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    best_dst = final_dir / f"{RUN_NAME}_best.pt"
    best_dst.write_bytes(best_src.read_bytes())

    print(f"\n✅ Training xong!")
    print(f"   Checkpoint : {best_src}")
    print(f"   Final model: {best_dst}")
    print(f"   mAP50      : {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")


if __name__ == "__main__":
    train()