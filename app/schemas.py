from pydantic import BaseModel


class DetectionItem(BaseModel):
    class_id:   int
    class_name: str
    confidence: float
    polygon:    list[list[float]]  # 4 điểm OBB: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]


class PredictJsonResponse(BaseModel):
    detections: list[DetectionItem]
    count:      int
    model_name: str


class PredictImageResponse(BaseModel):
    image_base64: str               # JPEG đã vẽ OBB, encode base64
    detections:   list[DetectionItem]
    count:        int
    model_name:   str


class HealthResponse(BaseModel):
    status:     str
    model_name: str
