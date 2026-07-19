from pathlib import Path
import yaml
from ultralytics import YOLO
from config import (
    DATA, MODELS, EVAL_OUT, EVAL,
    PREDICT_CONF, CLASS_CONF, CLASS_PRIORITY, CLASS_COLOR,
    IOU_THRESH, CONF_GAP,
)
from utils import find_latest_model, run_visual_predict


BEST_PT = MODELS / "final"


def evaluate() -> None:
    model_path = find_latest_model(BEST_PT)

    print(f"\n{'─'*50}")
    print(f"  Model : {model_path.name}")
    print(f"  Data  : {DATA}")
    print(f"  Output: {EVAL_OUT}")
    print(f"{'─'*50}\n")

    model = YOLO(str(model_path))

    metrics = model.val(
        data    = str(DATA),
        project = str(EVAL_OUT),
        name    = model_path.stem,
        **EVAL,
    )

    with open(DATA) as f:
        data_cfg    = yaml.safe_load(f)
        CLASS_NAMES = data_cfg["names"]

    box = metrics.box

    # ── Overall metrics ──────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  {'Metric':<25} {'Value':>10}")
    print(f"{'─'*50}")
    print(f"  {'Precision (mean)':<25} {box.mp:>10.4f}")
    print(f"  {'Recall (mean)':<25} {box.mr:>10.4f}")
    print(f"  {'mAP50':<25} {box.map50:>10.4f}")
    print(f"  {'mAP50-95':<25} {box.map:>10.4f}")

    # ── Per-class metrics ────────────────────────────────────────────
    print(f"{'─'*50}")
    print(f"  {'Class':<20} {'P':>7} {'R':>7} {'AP50':>7} {'AP50-95':>9}")
    print(f"{'─'*50}")
    for i, name in enumerate(CLASS_NAMES):
        p    = box.p[i]    if i < len(box.p)    else float("nan")
        r    = box.r[i]    if i < len(box.r)    else float("nan")
        ap50 = box.ap50[i] if i < len(box.ap50) else float("nan")
        ap   = box.ap[i]   if i < len(box.ap)   else float("nan")
        print(f"  {name:<20} {p:>7.4f} {r:>7.4f} {ap50:>7.4f} {ap:>9.4f}")

    print(f"{'─'*50}")
    print(f"\n✅ Plots saved: {EVAL_OUT / model_path.stem}")

    # ── Lưu kết quả ra file txt ──────────────────────────────────────
    out_dir = EVAL_OUT / model_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"Model : {model_path.name}",
        f"Data  : {DATA}",
        "─" * 50,
        f"  {'Metric':<25} {'Value':>10}",
        "─" * 50,
        f"  {'Precision (mean)':<25} {box.mp:>10.4f}",
        f"  {'Recall (mean)':<25} {box.mr:>10.4f}",
        f"  {'mAP50':<25} {box.map50:>10.4f}",
        f"  {'mAP50-95':<25} {box.map:>10.4f}",
        "─" * 50,
        f"  {'Class':<20} {'P':>7} {'R':>7} {'AP50':>7} {'AP50-95':>9}",
        "─" * 50,
    ]
    for i, name in enumerate(CLASS_NAMES):
        p    = box.p[i]    if i < len(box.p)    else float("nan")
        r    = box.r[i]    if i < len(box.r)    else float("nan")
        ap50 = box.ap50[i] if i < len(box.ap50) else float("nan")
        ap   = box.ap[i]   if i < len(box.ap)   else float("nan")
        lines.append(f"  {name:<20} {p:>7.4f} {r:>7.4f} {ap50:>7.4f} {ap:>9.4f}")
    lines.append("─" * 50)

    txt_path = out_dir / "results.txt"
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ Results saved: {txt_path}")

    # ── Visual predict trên split đang eval ─────────────────────────
    split_name  = EVAL.get("split", "test")
    split_root  = (DATA.parent / data_cfg["path"]).resolve()
    src_dir     = split_root / "images" / split_name
    visual_out  = out_dir / "visual"

    print(f"\n{'─'*50}")
    print(f"  Visual predict ({split_name})  →  {visual_out}")
    print(f"{'─'*50}")

    run_visual_predict(
        model          = model,
        source_dir     = src_dir,
        out_dir        = visual_out,
        class_names    = CLASS_NAMES,
        conf           = PREDICT_CONF,
        class_conf     = CLASS_CONF,
        class_priority = CLASS_PRIORITY,
        class_color    = CLASS_COLOR,
        iou_thresh     = IOU_THRESH,
        conf_gap       = CONF_GAP,
    )

    print(f"\n✅ Visual results saved: {visual_out}")


if __name__ == "__main__":
    evaluate()
