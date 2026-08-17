import os

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    r".\runs\detect\train-8\weights\best.pt",
)

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))

CLASS_NAMES = {
    0: "boots",
    1: "gloves",
    2: "helmet",
    3: "human",
    4: "vest",
}

PPE_CLASSES = ("helmet", "vest", "gloves", "boots")

CLASS_COLORS = {
    "helmet": (255, 200, 0),
    "vest": (0, 215, 255),
    "gloves": (0, 140, 255),
    "boots": (200, 0, 200),
    "human": (255, 255, 255),
}
