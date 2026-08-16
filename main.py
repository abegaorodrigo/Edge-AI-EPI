"""
API de Deteccao de EPIs
========================

Este arquivo pega o script original (que rodava o YOLO numa unica imagem
local, mostrava numa janela com cv2.imshow) e o transforma numa API web
com dois endpoints, que e o formato pedido no teste tecnico.

A logica de deteccao (rodar o modelo, separar pessoas de itens de EPI,
verificar se o item esta na posicao esperada do corpo) e a MESMA do
script original. So mudou o "empacotamento": em vez de rodar direto no
terminal e abrir uma janela, agora ela roda dentro de funcoes que a API
chama quando alguem manda uma imagem por HTTP.
"""

import os

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response, RedirectResponse
from ultralytics import YOLO

app = FastAPI(title="Deteccao de EPIs")

# --------------------------------------------------------------------
# 1. Carrega o modelo UMA VEZ, quando a API sobe (nao a cada requisicao)
# --------------------------------------------------------------------
# No seu script original essa linha era:
#   model = YOLO(r"runs\detect\train-8\weights\best.pt")
# Aqui so trocamos o caminho fixo por uma variavel de ambiente, para o
# Docker conseguir apontar para o arquivo certo. Se o arquivo treinado
# nao existir (ex: rodando so para testar a API), cai para o modelo
# generico do YOLO (yolov8n.pt) so para a API nao quebrar.


MODEL_PATH = os.getenv("MODEL_PATH", ".\\runs\\detect\\train-8\\weights\\best.pt")

model = YOLO(MODEL_PATH) if os.path.isfile(MODEL_PATH) else YOLO("yolov8n.pt")

# Classes do seu dataset, na mesma ordem do data.yaml usado no treino
CLASS_NAMES = {0: "boots", 1: "gloves", 2: "helmet", 3: "human", 4: "vest"}




def rodar_deteccao(imagem):
    results = model.predict(source=imagem, conf=0.5, verbose=False)
    result = results[0]

    tempos = result.speed
    tempo_inferencia_ms = round(sum(tempos.values()), 2)

    persons = []
    itens_por_classe = {}

    for box in result.boxes:
        cls_id = int(box.cls[0])
        class_name = CLASS_NAMES.get(cls_id, str(cls_id))
        xyxy = box.xyxy[0].tolist()

        if class_name == "human":
            persons.append(xyxy)
        else:
            itens_por_classe.setdefault(class_name, []).append(xyxy)

    return result, persons, itens_por_classe, tempo_inferencia_ms


def esta_associado_a_pessoa(item_bbox, pessoa_bbox):
    """
    Verifica se um item de EPI pertence a uma pessoa: o centro do item
    precisa estar dentro da caixa da pessoa. Sem depender de uma zona
    fixa de altura, ja que isso muda com angulo de camera, distancia
    e posicao (pessoa em pe, curvada, de lado, etc).
    """
    i_xmin, i_ymin, i_xmax, i_ymax = item_bbox
    p_xmin, p_ymin, p_xmax, p_ymax = pessoa_bbox

    centro_x = (i_xmin + i_xmax) / 2
    centro_y = (i_ymin + i_ymax) / 2

    return p_xmin <= centro_x <= p_xmax and p_ymin <= centro_y <= p_ymax


def checar_epis_faltando(persons, itens_por_classe):
    """
    Para cada pessoa detectada, verifica quais EPIs estao faltando,
    associando cada item pela posicao (centro dentro da caixa da
    pessoa), sem assumir uma zona vertical fixa por classe.
    """
    alertas = []

    for pessoa_bbox in persons:
        epis_detectados = []
        epis_faltando = []

        for classe in ("helmet", "vest", "gloves", "boots"):
            itens_da_classe = itens_por_classe.get(classe, [])
            tem_item = any(
                esta_associado_a_pessoa(item_bbox, pessoa_bbox)
                for item_bbox in itens_da_classe
            )
            if tem_item:
                epis_detectados.append(classe)
            else:
                epis_faltando.append(classe)

        alertas.append(
            {
                "pessoa_bbox": [round(v, 1) for v in pessoa_bbox],
                "epis_detectados": epis_detectados,
                "epis_faltando": epis_faltando,
            }
        )

    return alertas


def ler_imagem_do_upload(conteudo: bytes):
    """Converte os bytes recebidos por HTTP numa imagem que o OpenCV entende."""
    imagem = cv2.imdecode(np.frombuffer(conteudo, np.uint8), cv2.IMREAD_COLOR)
    if imagem is None:
        raise HTTPException(status_code=400, detail="Arquivo enviado nao e uma imagem valida")
    return imagem


# --------------------------------------------------------------------
# 3. Endpoints HTTP
# --------------------------------------------------------------------

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    """Endpoint simples para checar se a API esta no ar (usado pelo Docker)."""
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Envie um arquivo de imagem")

    conteudo = await file.read()
    imagem = ler_imagem_do_upload(conteudo)

    result, persons, itens_por_classe, tempo_inferencia_ms = rodar_deteccao(imagem)
    alertas = checar_epis_faltando(persons, itens_por_classe)

    deteccoes = [
        {
            "classe": CLASS_NAMES.get(int(box.cls[0]), str(int(box.cls[0]))),
            "confianca": round(float(box.conf[0]), 3),
            "bbox": [round(v, 1) for v in box.xyxy[0].tolist()],
        }
        for box in result.boxes
    ]

    return {
        "deteccoes": deteccoes,
        "alertas": alertas,
        "tempo_inferencia_ms": tempo_inferencia_ms,
    }

@app.post("/predict/annotated")
async def predict_annotated(file: UploadFile = File(...)):
    """
    Endpoint 2 (imagem): recebe uma imagem e devolve a mesma imagem
    com as caixas desenhadas em cima (PNG) - equivalente ao
    cv2.imshow(...) do seu script, so que devolvendo os bytes da
    imagem em vez de abrir uma janela (dentro do Docker nao tem tela
    para abrir janela).
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Envie um arquivo de imagem")

    conteudo = await file.read()
    imagem = ler_imagem_do_upload(conteudo)

    result, _, _, _ = rodar_deteccao(imagem)
    frame_anotado = result.plot()  # a mesma linha "annotated_frame = results[0].plot()" do seu script

    ok, png = cv2.imencode(".png", frame_anotado)
    if not ok:
        raise HTTPException(status_code=500, detail="Falha ao gerar imagem anotada")

    return Response(content=png.tobytes(), media_type="image/png")