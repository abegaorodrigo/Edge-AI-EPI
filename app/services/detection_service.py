from dataclasses import dataclass
from typing import Any

from ultralytics import YOLO

from app.core.config import CLASS_NAMES, CONFIDENCE_THRESHOLD, PPE_CLASSES


@dataclass
class DetectionResult:
    result: Any
    persons: list[list[float]]
    items_by_class: dict[str, list[list[float]]]
    total_time_ms: float
    duration_ms: dict[str, float]


class DetectionService:
    def __init__(self, model: YOLO):
        self.model = model

    def predict(self, image) -> DetectionResult:
        results = self.model.predict(
            source=image,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
        )
        result = results[0]

        duration = {
            "preprocess": round(result.speed.get("preprocess", 0.0), 2),
            "inference": round(result.speed.get("inference", 0.0), 2),
            "postprocess": round(result.speed.get("postprocess", 0.0), 2),
        }
        total_time_ms = round(sum(duration.values()), 2)

        persons: list[list[float]] = []
        items_by_class: dict[str, list[list[float]]] = {}

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = CLASS_NAMES.get(class_id, str(class_id))
            bbox = box.xyxy[0].tolist()

            if class_name == "human":
                persons.append(bbox)
            else:
                items_by_class.setdefault(class_name, []).append(bbox)

        return DetectionResult(
            result=result,
            persons=persons,
            items_by_class=items_by_class,
            total_time_ms=total_time_ms,
            duration_ms=duration,
        )

    @staticmethod
    def build_detections(result) -> list[dict]:
        return [
            {
                "classe": CLASS_NAMES.get(int(box.cls[0]), str(int(box.cls[0]))),
                "confianca": round(float(box.conf[0]), 3),
                "bbox": [round(v, 1) for v in box.xyxy[0].tolist()],
            }
            for box in result.boxes
        ]

    @staticmethod
    def is_associated_to_person(
        item_bbox: list[float],
        person_bbox: list[float],
    ) -> bool:
        item_xmin, item_ymin, item_xmax, item_ymax = item_bbox
        person_xmin, person_ymin, person_xmax, person_ymax = person_bbox

        center_x = (item_xmin + item_xmax) / 2
        center_y = (item_ymin + item_ymax) / 2

        return (
            person_xmin <= center_x <= person_xmax
            and person_ymin <= center_y <= person_ymax
        )

    def check_missing_ppe(
        self,
        persons: list[list[float]],
        items_by_class: dict[str, list[list[float]]],
    ) -> list[dict]:
        alerts = []

        for person_bbox in persons:
            detected = []
            missing = []

            for ppe_class in PPE_CLASSES:
                class_items = items_by_class.get(ppe_class, [])
                has_item = any(
                    self.is_associated_to_person(item_bbox, person_bbox)
                    for item_bbox in class_items
                )

                if has_item:
                    detected.append(ppe_class)
                else:
                    missing.append(ppe_class)

            alerts.append(
                {
                    "pessoa_bbox": [round(v, 1) for v in person_bbox],
                    "epis_detectados": detected,
                    "epis_faltando": missing,
                }
            )

        return alerts
