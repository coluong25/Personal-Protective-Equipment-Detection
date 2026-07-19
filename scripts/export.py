from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))

from utils import find_latest_model
from config import EXPORT as ECFG

BEST_PT    = ROOT / "models" / "final"
EXPORT_OUT = ROOT / "outputs" / "export"


def export() -> None:
    model_path = find_latest_model(BEST_PT)
    fmt        = ECFG["format"]

    print(f"\n{'─'*50}")
    print(f"  Model  : {model_path.name}")
    print(f"  Format : {fmt}")
    print(f"  Output : {EXPORT_OUT}")
    print(f"{'─'*50}\n")

    model = YOLO(str(model_path))

    export_kwargs = {k: v for k, v in ECFG.items() if k != "format"}
    if fmt != "onnx":
        export_kwargs.pop("opset", None)

    exported_path = model.export(format=fmt, **export_kwargs)

    EXPORT_OUT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    dest_name = f"{model_path.stem}_{timestamp}{Path(str(exported_path)).suffix}"
    dest      = EXPORT_OUT / dest_name
    Path(str(exported_path)).replace(dest)

    print(f"\n✅ Exported: {dest}")


if __name__ == "__main__":
    export()