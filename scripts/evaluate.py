from pathlib import Path
from ultralytics import YOLO
import json
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
import yaml
DATA     = ROOT / "data.yaml"
BEST_PT  = ROOT / "models" / "final"   # tìm file .pt mới nhất trong đây
EVAL_OUT = ROOT / "outputs" / "eval"


def find_latest_model(model_dir: Path) -> Path:
    """
    Tìm file .pt mới nhất trong models/final/.
    Input : Path("models/final/")
    Output: Path("models/final/yolov8s_20240513_1430_best.pt")
    """
    pts = sorted(model_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    if not pts:
        raise FileNotFoundError(f"Không tìm thấy file .pt trong {model_dir}")
    return pts[-1]   # mới nhất theo thời gian


def evaluate() -> None:
    model_path = find_latest_model(BEST_PT)

    print(f"\n{'─'*50}")
    print(f"  Model : {model_path.name}")
    print(f"  Data  : {DATA}")
    print(f"  Output: {EVAL_OUT}")
    print(f"{'─'*50}\n")

    model = YOLO(str(model_path))

    # ── Chạy evaluation trên test set ───────────────────────────────
    # val() của Ultralytics trả về metrics + tự save confusion matrix, PR curve
    metrics = model.val(
        data    = str(DATA),
        split   = "test",          # dùng test set, không phải val
        save_json = True,          # save kết quả ra file JSON
        plots   = True,            # vẽ confusion matrix + PR curve tự động
        project = str(EVAL_OUT),
        name    = model_path.stem, # tên folder output = tên model
    )

    # ── In metrics ra terminal ───────────────────────────────────────
    with open(DATA) as f:
        CLASS_NAMES = yaml.safe_load(f)["names"]

    print(f"\n{'─'*50}")
    print(f"  {'Metric':<25} {'Value':>10}")
    print(f"{'─'*50}")
    print(f"  {'mAP50 (toàn bộ)':<25} {metrics.box.map50:>10.4f}")
    print(f"  {'mAP50-95 (toàn bộ)':<25} {metrics.box.map:>10.4f}")
    print(f"{'─'*50}")

    # Metrics từng class
    for i, name in enumerate(CLASS_NAMES):
        ap50 = metrics.box.ap50[i] if i < len(metrics.box.ap50) else float("nan")
        print(f"  AP50 {name:<20} {ap50:>10.4f}")

    print(f"{'─'*50}")
    print(f"\n✅ Plots saved: {EVAL_OUT / model_path.stem}")


if __name__ == "__main__":
    evaluate()