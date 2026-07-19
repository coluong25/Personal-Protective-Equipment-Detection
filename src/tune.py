from datetime import datetime
from ultralytics import YOLO
from config import TRAIN as CFG, TUNE as TCFG, MODELS
from utils import make_cls_weight_callback

RUN_NAME = f"tune_{datetime.now().strftime('%Y%m%d_%H%M')}"


def tune() -> None:
    print(f"\n{'─'*50}")
    print(f"  Model      : {CFG['model']}")
    print(f"  Data       : {CFG['data']}")
    print(f"  Epochs/trial: {TCFG['epochs']}  |  Iterations: {TCFG['iterations']}")
    print(f"  Run        : {RUN_NAME}")
    print(f"{'─'*50}\n")

    model = YOLO(CFG["model"])
    model.add_callback("on_train_start", make_cls_weight_callback(CFG["data"]))

    result = model.tune(
        data       = CFG["data"],
        epochs     = TCFG["epochs"],
        iterations = TCFG["iterations"],
        optimizer  = TCFG["optimizer"],
        imgsz      = CFG["imgsz"],
        batch      = CFG["batch"],
        device     = CFG["device"],
        workers    = CFG["workers"],
        seed       = CFG["seed"],
        plots      = True,
        save       = False,   # không save checkpoint của từng trial
        val        = True,
        project    = str(MODELS / "tune"),
        name       = RUN_NAME,
    )

    print(f"\n{'─'*50}")
    print("  Best hyperparameters:")
    if result:
        for k, v in result.items():
            print(f"    {k}: {v}")
    else:
        hyper_file = MODELS / "tune" / RUN_NAME / "best_hyperparameters.yaml"
        if hyper_file.exists():
            print(f"  (result=None, đọc từ file: {hyper_file})")
            print(hyper_file.read_text())
        else:
            print("  Không lấy được kết quả từ model.tune()")
    print(f"\n  Kết quả lưu tại: {MODELS / 'tune' / RUN_NAME}")
    print(f"  Cập nhật TRAIN config rồi chạy train.py với các params tốt nhất.")
    print(f"{'─'*50}\n")


if __name__ == "__main__":
    tune()
