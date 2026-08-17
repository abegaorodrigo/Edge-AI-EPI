from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response

from app.core.model import get_model
from app.services.detection_service import DetectionService
from app.services.image_service import decode_image, encode_png
from app.utils.annotations import draw_annotations

router = APIRouter()

# O modelo é carregado uma vez por processo da API.
detection_service = DetectionService(get_model())


async def _read_uploaded_image(file: UploadFile):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Envie um arquivo de imagem",
        )

    return decode_image(await file.read())


@router.post("/predict", tags=["prediction"])
async def predict(file: UploadFile = File(...)):
    image = await _read_uploaded_image(file)
    detection = detection_service.predict(image)

    alerts = detection_service.check_missing_ppe(
        detection.persons,
        detection.items_by_class,
    )
    detections = detection_service.build_detections(detection.result)

    return {
        "deteccoes": detections,
        "alertas": alerts,
        "tempo_total_ms": detection.total_time_ms,
        "duracao_ms": detection.duration_ms,
    }


@router.post("/predict/annotated", tags=["prediction"])
async def predict_annotated(file: UploadFile = File(...)):
    image = await _read_uploaded_image(file)
    detection = detection_service.predict(image)

    alerts = detection_service.check_missing_ppe(
        detection.persons,
        detection.items_by_class,
    )
    detections = detection_service.build_detections(detection.result)

    annotated = draw_annotations(
        image,
        detections,
        alerts,
        detection.total_time_ms,
        detection.duration_ms,
    )

    return Response(
        content=encode_png(annotated),
        media_type="image/png",
    )
