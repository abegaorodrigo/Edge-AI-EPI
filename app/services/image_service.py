import cv2
import numpy as np
from fastapi import HTTPException


def decode_image(content: bytes):
    image = cv2.imdecode(
        np.frombuffer(content, np.uint8),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Arquivo enviado nao e uma imagem valida",
        )

    return image


def encode_png(image) -> bytes:
    ok, png = cv2.imencode(".png", image)

    if not ok:
        raise HTTPException(
            status_code=500,
            detail="Falha ao gerar imagem anotada",
        )

    return png.tobytes()
