from pathlib import Path
import os
from ultralytics import YOLO
from config import (
    ROOT,
    PREDICT_CONF,
    CLASS_CONF, CLASS_PRIORITY, CLASS_COLOR,
    IOU_THRESH, CONF_GAP,
)
from utils import find_latest_model, run_visual_predict
import yaml

BEST_PT = ROOT / "models" / "final"

EXTERNAL_TEST = Path(
    os.getenv("EXTERNAL_TEST_DIR", str(ROOT / "data" / "test" / "images"))
)

TEST_OUT = Path(
    os.getenv("TEST_OUT_DIR", str(ROOT / "outputs" / "predict"))
)


def predict(conf: float = PREDICT_CONF) -> None:
    model_path = find_latest_model(BEST_PT)

    if not EXTERNAL_TEST.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục test: {EXTERNAL_TEST}")

    print(f"\n{'─'*55}")
    print(f"  Model : {model_path.name}")
    print(f"  Source: {EXTERNAL_TEST}")
    print(f"  Output: {TEST_OUT}")
    print(f"{'─'*55}\n")

    with open(ROOT / "data.yaml") as f:
        class_names = yaml.safe_load(f)["names"]

    model   = YOLO(str(model_path))
    out_dir = TEST_OUT / model_path.stem

    run_visual_predict(
        model          = model,
        source_dir     = EXTERNAL_TEST,
        out_dir        = out_dir,
        class_names    = class_names,
        conf           = conf,
        class_conf     = CLASS_CONF,
        class_priority = CLASS_PRIORITY,
        class_color    = CLASS_COLOR,
        iou_thresh     = IOU_THRESH,
        conf_gap       = CONF_GAP,
    )

    print(f"\n✅ Results saved: {out_dir}")


if __name__ == "__main__":
    predict()
