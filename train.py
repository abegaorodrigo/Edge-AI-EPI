import os
from ultralytics import YOLO

MODEL= os.getenv("TRAIN_MODEL_PATH", "yolov8n.pt")
DATA = os.getenv("TRAIN_DATA_PATH", ".\\PPE-Detection-1\\data.yaml")

if __name__ == "__main__":
    # Carrega o modelo
    model = YOLO(MODEL)

    # Inicia o treinamento
    model.train(
        data=DATA,
        epochs=60,
        imgsz=640,
        batch=16,
        patience=15,
        workers=4,  # Agora você pode usar workers à vontade!
    )