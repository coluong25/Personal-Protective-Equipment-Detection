import base64
import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))                  # app/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))   # src/

import cv2
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile

from predictor import Predictor
from schemas import DetectionItem, HealthResponse, PredictImageResponse, PredictJsonResponse
from config import ROOT
from utils import find_latest_model

_predictor: Predictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor
    model_path  = find_latest_model(ROOT / "models" / "final")
    with open(ROOT / "data.yaml") as f:
        class_names = yaml.safe_load(f)["names"]
    _predictor = Predictor(model_path, class_names)
    print(f"\n  Model loaded: {model_path.name}")
    yield
    _predictor = None


app = FastAPI(
    title="PPE Detection API",
    description="Personal Protective Equipment Detection — YOLOv8m-OBB",
    version="1.0.0",
    lifespan=lifespan,
)


def _get_predictor() -> Predictor:
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model chưa được load.")
    return _predictor


def _parse_detections(raw: list[dict]) -> list[DetectionItem]:
    return [DetectionItem(**d) for d in raw]


@app.get("/health", response_model=HealthResponse)
def health():
    p = _get_predictor()
    return HealthResponse(status="ok", model_name=p.model_name)


@app.post("/predict/json", response_model=PredictJsonResponse)
async def predict_json(file: UploadFile = File(..., description="Ảnh JPG / PNG")):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File phải là ảnh (image/*).")
    image_bytes        = await file.read()
    detections_raw, _ = _get_predictor().predict(image_bytes)
    detections         = _parse_detections(detections_raw)
    return PredictJsonResponse(
        detections = detections,
        count      = len(detections),
        model_name = _predictor.model_name,
    )


@app.post("/predict/image", response_model=PredictImageResponse)
async def predict_image(file: UploadFile = File(..., description="Ảnh JPG / PNG")):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File phải là ảnh (image/*).")
    image_bytes             = await file.read()
    detections_raw, annotated = _get_predictor().predict(image_bytes)
    detections               = _parse_detections(detections_raw)
    _, buf   = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img_b64  = base64.b64encode(buf).decode()
    return PredictImageResponse(
        image_base64 = img_b64,
        detections   = detections,
        count        = len(detections),
        model_name   = _predictor.model_name,
    )
