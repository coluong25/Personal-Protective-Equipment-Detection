"""
test_api.py — Test FastAPI endpoints bằng requests.

Chạy server trước:  uvicorn app.main:app --reload
Sau đó chạy:        python tests/test_api.py
"""
import base64
from pathlib import Path

import requests

BASE_URL  = "http://localhost:8000"
TEST_DIR  = Path("data/test/images")
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def _first_image() -> Path:
    images = [p for p in TEST_DIR.iterdir() if p.suffix.lower() in IMAGE_EXT]
    if not images:
        raise FileNotFoundError(f"Không tìm thấy ảnh trong {TEST_DIR}")
    return images[0]


def test_health() -> None:
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200, f"health failed: {r.text}"
    data = r.json()
    assert data["status"] == "ok"
    print(f"[PASS] /health  →  model={data['model_name']}")


def test_predict_json(image_path: Path) -> None:
    with open(image_path, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/predict/json",
            files={"file": (image_path.name, f, "image/jpeg")},
        )
    assert r.status_code == 200, f"predict/json failed: {r.text}"
    data = r.json()
    print(f"[PASS] /predict/json  →  {data['count']} detection(s)")
    for d in data["detections"]:
        print(f"       {d['class_name']:<22} conf={d['confidence']:.4f}")


def test_predict_image(image_path: Path) -> None:
    with open(image_path, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/predict/image",
            files={"file": (image_path.name, f, "image/jpeg")},
        )
    assert r.status_code == 200, f"predict/image failed: {r.text}"
    data   = r.json()
    b64    = data["image_base64"]
    img_bytes = base64.b64decode(b64)
    out    = Path("tests/output_annotated.jpg")
    out.write_bytes(img_bytes)
    print(f"[PASS] /predict/image  →  {data['count']} detection(s), saved {out}")


def test_invalid_file() -> None:
    r = requests.post(
        f"{BASE_URL}/predict/json",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert r.status_code == 400, f"Expected 400, got {r.status_code}"
    print("[PASS] /predict/json với file không hợp lệ → 400")


if __name__ == "__main__":
    img = _first_image()
    print(f"\nTest image: {img.name}\n{'─'*40}")
    test_health()
    test_predict_json(img)
    test_predict_image(img)
    test_invalid_file()
    print(f"{'─'*40}\nAll tests passed.")
