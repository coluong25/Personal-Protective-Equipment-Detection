from datetime import datetime
from pathlib import Path
import yaml
from ultralytics import YOLO
from config import (
    ROOT, TRAIN as CFG, HYPER, MODELS,
    PREDICT_CONF, CLASS_CONF, CLASS_PRIORITY, CLASS_COLOR,
    IOU_THRESH, CONF_GAP,
)
from utils import make_cls_weight_callback, run_visual_predict

RUN_NAME = f"{Path(CFG['model']).stem}_{datetime.now().strftime('%Y%m%d_%H%M')}"


def _visual_after_train(model_path: Path, run_name: str) -> None:
    """Chạy visual predict trên toàn bộ data/split sau khi train xong."""
    with open(CFG["data"]) as f:
        data_cfg = yaml.safe_load(f)

    class_names = data_cfg["names"]
    # data.yaml: path: ./data/split  →  ROOT/data/split
    split_root  = (Path(CFG["data"]).parent / data_cfg["path"]).resolve()

    model   = YOLO(str(model_path))
    vis_out = ROOT / "outputs" / "train_visual" / run_name

    print(f"\n{'─'*55}")
    print(f"  Visual predict sau train  →  {vis_out}")
    print(f"{'─'*55}")

    for split in ("train", "val"):
        src = split_root / "images" / split
        run_visual_predict(
            model          = model,
            source_dir     = src,
            out_dir        = vis_out / split,
            class_names    = class_names,
            conf           = PREDICT_CONF,
            class_conf     = CLASS_CONF,
            class_priority = CLASS_PRIORITY,
            class_color    = CLASS_COLOR,
            iou_thresh     = IOU_THRESH,
            conf_gap       = CONF_GAP,
        )

    print(f"\n✅ Visual results saved: {vis_out}")


def train() -> None:
    print(f"\n{'─'*50}")
    print(f"  Model   : {CFG['model']}")
    print(f"  Data    : {CFG['data']}")
    print(f"  Epochs  : {CFG['epochs']}  |  Batch: {CFG['batch']}")
    print(f"  Run     : {RUN_NAME}")
    print(f"{'─'*50}\n")

    model = YOLO(CFG["model"])
    model.add_callback("on_train_start", make_cls_weight_callback(CFG["data"]))

    results = model.train(
        data      = CFG["data"],
        epochs    = CFG["epochs"],
        imgsz     = CFG["imgsz"],
        batch     = CFG["batch"],
        patience  = CFG["patience"],
        device    = CFG["device"],
        workers   = CFG["workers"],
        seed      = CFG["seed"],
        project   = str(MODELS / "checkpoints"),
        name      = RUN_NAME,
        save      = True,
        val       = True,
        **HYPER,
    )

    best_src  = MODELS / "checkpoints" / RUN_NAME / "weights" / "best.pt"
    final_dir = MODELS / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    best_dst = final_dir / f"{RUN_NAME}_best.pt"
    best_dst.write_bytes(best_src.read_bytes())

    print(f"\n✅ Training xong!")
    print(f"   Checkpoint : {best_src}")
    print(f"   Final model: {best_dst}")
    print(f"   mAP50      : {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")

    _visual_after_train(best_dst, RUN_NAME)


if __name__ == "__main__":
    train()
