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
            detail="Envie um arquivo de imagem (image/jpeg, image/png, etc.)",
        )

    return decode_image(await file.read())


@router.post(
    "/predict",
    tags=["Predição"],
    summary="Detecção de EPIs — resposta JSON",
    response_description="Detecções, alertas de conformidade e métricas de tempo",
    responses={
        200: {"description": "Detecções realizadas com sucesso"},
        400: {"description": "Arquivo enviado não é uma imagem válida"},
    },
)
async def predict(file: UploadFile = File(..., description="Imagem JPG ou PNG para análise")):
    """
    Recebe uma imagem e retorna as detecções em formato **JSON estruturado**.

    O retorno inclui:
    - **deteccoes**: lista de objetos detectados com classe, confiança e bounding box `[x1, y1, x2, y2]`
    - **alertas**: por pessoa detectada — quais EPIs estão presentes e quais estão faltando
    - **tempo_total_ms**: latência total da inferência em milissegundos
    - **duracao_ms**: breakdown por etapa — `preprocess`, `inference`, `postprocess`

    Classes monitoradas: `helmet`, `vest`, `gloves`, `boots`, `human`.
    """
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


@router.post(
    "/predict/annotated",
    tags=["Predição"],
    summary="Detecção de EPIs — imagem anotada",
    response_description="Imagem PNG com bounding boxes e status de conformidade sobrepostos",
    responses={
        200: {"content": {"image/png": {}}, "description": "Imagem PNG anotada com sucesso"},
        400: {"description": "Arquivo enviado não é uma imagem válida"},
        500: {"description": "Falha interna ao codificar a imagem de saída"},
    },
)
async def predict_annotated(file: UploadFile = File(..., description="Imagem JPG ou PNG para análise")):
    """
    Recebe uma imagem e retorna o **frame visualmente anotado** como `image/png`.

    Sobreposições incluídas:
    - **Caixas coloridas por classe** de EPI (capacete, colete, luvas, botas)
    - **Caixa verde** ao redor de trabalhadores em conformidade total (`CONFORME ✔`)
    - **Caixa vermelha** com lista dos EPIs faltantes (`FALTANDO: helmet, vest…`)
    - **HUD superior** com status geral da cena e métricas de tempo detalhadas

    Ideal para exibição em painéis de monitoramento, câmeras de segurança e clientes de vídeo.
    """
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
