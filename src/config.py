from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
DATA     = ROOT / "data.yaml"
MODELS   = ROOT / "models"
EVAL_OUT = ROOT / "outputs" / "eval"
TEST_OUT = ROOT / "outputs" / "test_external"

# Thư mục ảnh test ngoài — override bằng env var EXTERNAL_TEST_DIR nếu cần
import os
EXTERNAL_TEST = Path(os.getenv("EXTERNAL_TEST_DIR", str(ROOT / "data" / "external_test" / "images")))


# ── Augmentation ─────────────────────────────────────────────────────────────
# target_count = số instances mỗi class muốn đạt (so sánh instances, không phải ảnh)
# classes = danh sách class ID cần augment (bỏ class đã đủ instances)
# copy_paste_n = số ảnh synthetic tạo thêm (chỉ áp dụng cho class 0 - bare_head)
AUGMENT = {
    "target_count" : 250,    # target instances mỗi class sau direct aug
    "classes"      : [0, 2], # 0=bare_head, 2=wrong_headgear_type (class 1 đã có 504)
    "copy_paste_n" : 20,     # số ảnh copy-paste tạo thêm (áp dụng cả class 0 và 2)
}


# ── Tuning ───────────────────────────────────────────────────────────────────
TUNE = {
    "epochs"     : 30,    # epochs mỗi trial — ngắn hơn TRAIN để chạy được nhiều trial
    "iterations" : 10,    # số HPO trial; tăng lên 100 nếu có thời gian
    "optimizer"  : "AdamW",
}

# ── Best Hyperparameters — tự động load từ tune run mới nhất ─────────────────
# Nếu chưa có tune run nào, HYPER = {} và train.py vẫn chạy với default
try:
    from utils import find_latest_tune_hyper
    HYPER: dict = find_latest_tune_hyper(MODELS / "tune")
except FileNotFoundError:
    HYPER = {}


# ── Training ─────────────────────────────────────────────────────────────────
# Các thông số hay thay đổi khi thử nghiệm model mới hoặc điều chỉnh training
TRAIN = {
    "model"   : "yolov8m-obb.pt",  # pretrained weights — đổi sang yolov8m/l để tăng accuracy
    "data"    : str(DATA),
    "epochs"  : 100,
    "imgsz"   : 640,
    "batch"   : 8,                  # giảm xuống 8 nếu GPU VRAM < 6GB
    "patience": 25,                 # early stopping — dừng nếu 25 epoch không cải thiện
    "device"  : 0,                  # 0 = GPU đầu tiên | "cpu" nếu không có GPU
    "workers" : 0,                  # Windows bắt buộc để 0
    "seed"    : 42,
}


# ── Inference / Test External ─────────────────────────────────────────────────
# Ngưỡng confidence ban đầu — thấp để lọc thủ công qua CLASS_CONF bên dưới
PREDICT_CONF = 0.05

# Ngưỡng confidence riêng từng class (fine-tune để giảm false positive)
# Key = class_id theo data.yaml: 0=bare_head, 1=constructor_hat, 2=wrong_headgear_type
CLASS_CONF = {
    0: 0.20,   # bare_head       — đầu trần (ngưỡng thấp, ưu tiên không bỏ sót)
    1: 0.62,   # constructor_hat — mũ bảo hộ
    2: 0.56,   # wrong_headgear_type — mũ không đạt chuẩn
}

# Độ ưu tiên khi 2 box chồng nhau: số cao hơn = ưu tiên hơn
# constructor_hat (3) > wrong_headgear_type (2) > bare_head (1)
CLASS_PRIORITY = {
    0: 1,  # bare_head
    1: 3,  # constructor_hat
    2: 2,  # wrong_headgear_type
}

# Màu vẽ bounding box theo class (BGR)
CLASS_COLOR = {
    0: (0,   0,   255),  # bare_head       — đỏ
    1: (0,   255,   0),  # constructor_hat — xanh lá
    2: (0,   165, 255),  # wrong_headgear_type — cam
}

# Conflict resolution: loại box chồng nhau (IoU > iou_thresh)
IOU_THRESH = 0.45   # box IoU vượt ngưỡng này mới xử lý conflict
CONF_GAP   = 0.15   # chênh lệch confidence đủ lớn → tin box có conf cao hơn


# ── Evaluation ────────────────────────────────────────────────────────────────
EVAL = {
    "split"    : "test",   # "val" hoặc "test"
    "save_json": True,
    "plots"    : True,
    "conf"     : 0.001,    # thấp để tính đủ PR curve (không dùng để filter kết quả)
    "iou"      : 0.6,      # IoU threshold cho NMS trong lúc eval
    "max_det"  : 300,      # max detections per image
    "device"   : 0,        # 0 = GPU | "cpu"
    "workers"  : 0,        # Windows cần 0
}


# ── Export ────────────────────────────────────────────────────────────────────
EXPORT = {
    "format"  : "onnx",   # "onnx" | "engine" (TensorRT) | "torchscript"
    "imgsz"   : 640,
    "simplify": True,
    "dynamic" : False,
    "opset"   : 17,       # opset: phiên bản tập lệnh ONNX — chỉ áp dụng cho format="onnx"
}