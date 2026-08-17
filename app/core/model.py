import os
import numpy as np
from app.core.config import MODEL_PATH
from ultralytics import YOLO


# Singleton simples
_model = None


def get_model() -> YOLO:
    global _model

    if _model is None:
        print(f"Loading model from {MODEL_PATH}")
        _model = (
            YOLO(MODEL_PATH)
            if os.path.isfile(MODEL_PATH)
            else YOLO("yolov8n.pt")
        )
        # Warm-up: compila kernels do PyTorch na inicialização
        # para que a 1ª requisição real tenha latência normal.
        _model.predict(source=np.zeros((640, 640, 3), dtype=np.uint8), conf=0.5, verbose=False)

    return _model
