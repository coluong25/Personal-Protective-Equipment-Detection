import torch
import cv2
import ultralytics
import numpy as np

print("=== Kiểm tra môi trường ===")
print(f"PyTorch version   : {torch.__version__}")
print(f"CUDA available    : {torch.cuda.is_available()}")  # True nếu có GPU
print(f"OpenCV version    : {cv2.__version__}")
print(f"Ultralytics ver   : {ultralytics.__version__}")
print(f"NumPy version     : {np.__version__}")

# Test YOLOv8 load nhanh
from ultralytics import YOLO
model = YOLO("yolov8n.pt")  # tự tải model nhỏ nhất (~6MB)
print("YOLOv8 load OK!")