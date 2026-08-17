import os
from app.core.config import MODEL_PATH
from ultralytics import YOLO


# Singleton simples
model = None


def get_model():
    global model

    if model is None:
        print(f"Loading model from {MODEL_PATH}")
        model = (
            YOLO(MODEL_PATH)
            if os.path.isfile(MODEL_PATH)
            else YOLO("yolov8n.pt")
        )

    return model
