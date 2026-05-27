# predict.py — đặt ngang hàng với evaluate.py
from pathlib import Path
from ultralytics import YOLO
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

BEST_PT   = ROOT / "models" / "final"
PRED_OUT  = ROOT / "outputs" / "predictions"
TEST_IMGS = ROOT / "data" / "test" / "images"   # ← ảnh test có sẵn, dùng lại luôn


def find_latest_model(model_dir: Path) -> Path:
    pts = sorted(model_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    if not pts:
        raise FileNotFoundError(f"Không tìm thấy file .pt trong {model_dir}")
    return pts[-1]


def predict(source: Path = TEST_IMGS, conf: float = 0.25) -> None:
    model_path = find_latest_model(BEST_PT)

    print(f"\n{'─'*50}")
    print(f"  Model : {model_path.name}")
    print(f"  Source: {source}")
    print(f"  Output: {PRED_OUT}")
    print(f"{'─'*50}\n")

    model = YOLO(str(model_path))

    model.predict(
        source  = str(source),
        conf    = conf,
        save    = True,       # lưu ảnh có vẽ bbox
        project = str(PRED_OUT),
        name    = model_path.stem,
    )

    print(f"\n✅ Results saved: {PRED_OUT / model_path.stem}")


if __name__ == "__main__":
    predict()