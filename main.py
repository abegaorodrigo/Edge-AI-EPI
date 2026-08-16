"""
API de Deteccao de EPIs
========================

API web construida com FastAPI para deteccao de Equipamentos de Protecao
Individual (EPIs) em tempo real usando YOLOv8.

Oferece endpoints para:
1. Retorno estruturado em JSON (/predict) com caixas, confiancas, alertas de conformidade e tempos detalhados.
2. Retorno visual anotado (/predict/annotated) com caixas padronizadas, cores por classe e status de conformidade (verde/vermelho).
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
MODEL_PATH = os.getenv("MODEL_PATH", ".\\runs\\detect\\train-8\\weights\\best.pt")

model = YOLO(MODEL_PATH) if os.path.isfile(MODEL_PATH) else YOLO("yolov8n.pt")

# Classes do dataset na mesma ordem do data.yaml usado no treino
CLASS_NAMES = {0: "boots", 1: "gloves", 2: "helmet", 3: "human", 4: "vest"}

# Cores BGR padronizadas para cada classe
CLASS_COLORS = {
    "helmet": (255, 200, 0),    # Ciano / Azul Claro
    "vest": (0, 215, 255),      # Amarelo / Dourado
    "gloves": (0, 140, 255),    # Laranja
    "boots": (200, 0, 200),     # Roxo
    "human": (255, 255, 255),   # Branco
}


def rodar_deteccao(imagem):
    results = model.predict(source=imagem, conf=0.5, verbose=False)
    result = results[0]

    # result.speed contem {'preprocess': float, 'inference': float, 'postprocess': float} em ms
    # Arredonda cada etapa para 2 casas decimais
    duracao = {
        "preprocess": round(result.speed.get("preprocess", 0.0), 2),
        "inference": round(result.speed.get("inference", 0.0), 2),
        "postprocess": round(result.speed.get("postprocess", 0.0), 2),
    }
    tempo_total_ms = round(sum(duracao.values()), 2)

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

    return result, persons, itens_por_classe, tempo_total_ms, duracao


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


def desenhar_anotacoes(imagem, deteccoes, alertas, tempo_total_ms=None, duracao=None):
    """
    Desenha caixas delimitadoras padronizadas, cores por classe e status
    de conformidade (verde/vermelho) diretamente no frame.
    """
    frame = imagem.copy()
    h, w, _ = frame.shape

    # 1. Desenha os itens de EPI detectados
    for det in deteccoes:
        classe = det.get("classe", "")
        if classe == "human":
            continue  # O trabalhador e desenhado com o status de conformidade abaixo

        conf = det.get("confianca", 0.0)
        bbox = det.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = map(int, bbox)
        cor = CLASS_COLORS.get(classe, (0, 255, 0))

        # Caixa do EPI
        cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)

        # Label do EPI com fundo colorido para facilitar a leitura
        texto = f"{classe} ({conf:.2f})"
        (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, max(0, y1)), cor, -1)
        cv2.putText(frame, texto, (x1 + 2, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # 2. Desenha cada pessoa com status de conformidade (Verde se OK, Vermelho se irregular)
    pessoas_conformes = 0
    total_pessoas = len(alertas)

    for alerta in alertas:
        bbox = alerta.get("pessoa_bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = map(int, bbox)
        faltando = alerta.get("epis_faltando", [])

        if not faltando:
            pessoas_conformes += 1
            cor_pessoa = (0, 200, 0)  # Verde
            status_texto = "CONFORME [OK]"
        else:
            cor_pessoa = (0, 0, 255)  # Vermelho
            status_texto = f"FALTANDO: {', '.join(faltando)}"

        # Caixa da pessoa
        cv2.rectangle(frame, (x1, y1), (x2, y2), cor_pessoa, 3)

        # Barra de status sobre a pessoa
        (tw, th), _ = cv2.getTextSize(status_texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        bg_y1 = max(0, y1 - th - 10)
        bg_y2 = max(0, y1)
        cv2.rectangle(frame, (x1, bg_y1), (x1 + tw + 10, bg_y2), cor_pessoa, -1)
        cv2.putText(frame, status_texto, (x1 + 5, bg_y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # 3. HUD Superior (Painel de Status)
    hud_bg_color = (30, 30, 30)
    cv2.rectangle(frame, (0, 0), (w, 42), hud_bg_color, -1)

    if total_pessoas == 0:
        conformidade_str = "Nenhum trabalhador na cena"
        status_color = (200, 200, 200)
    elif pessoas_conformes == total_pessoas:
        conformidade_str = f"STATUS: 100% SEGURO ({pessoas_conformes}/{total_pessoas})"
        status_color = (0, 255, 0)
    else:
        conformidade_str = f"ALERTA: {total_pessoas - pessoas_conformes} IRREGULAR(ES) ({pessoas_conformes}/{total_pessoas})"
        status_color = (0, 0, 255)

    cv2.putText(frame, conformidade_str, (15, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA)

    # Exibe métricas de tempo (detalhado por pré/inferência/pós se disponível)
    if duracao and isinstance(duracao, dict):
        metricas_str = f"Total: {tempo_total_ms:.1f}ms (Pre: {duracao.get('preprocess', 0):.1f} | Inf: {duracao.get('inference', 0):.1f} | Pos: {duracao.get('postprocess', 0):.1f})"
        (mw, _), _ = cv2.getTextSize(metricas_str, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(frame, metricas_str, (w - mw - 15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    elif tempo_total_ms is not None:
        metricas_str = f"Tempo: {tempo_total_ms:.1f}ms"
        (mw, _), _ = cv2.getTextSize(metricas_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(frame, metricas_str, (w - mw - 15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

    return frame


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
    """
    Endpoint 1 (JSON): Recebe uma imagem e retorna detecções, caixas,
    confiança, alertas de EPIs faltantes, tempo total e duração detalhada.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Envie um arquivo de imagem")

    conteudo = await file.read()
    imagem = ler_imagem_do_upload(conteudo)

    result, persons, itens_por_classe, tempo_total_ms, duracao = rodar_deteccao(imagem)
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
        "tempo_total_ms": tempo_total_ms,
        "duracao_ms": duracao,
    }


@app.post("/predict/annotated")
async def predict_annotated(file: UploadFile = File(...)):
    """
    Endpoint 2 (Imagem Anotada): Recebe uma imagem e retorna o frame com
    caixas delimitadoras customizadas, cores padronizadas e indicadores
    de conformidade (verde/vermelho).
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Envie um arquivo de imagem")

    conteudo = await file.read()
    imagem = ler_imagem_do_upload(conteudo)

    result, persons, itens_por_classe, tempo_total_ms, duracao = rodar_deteccao(imagem)
    alertas = checar_epis_faltando(persons, itens_por_classe)

    deteccoes = [
        {
            "classe": CLASS_NAMES.get(int(box.cls[0]), str(int(box.cls[0]))),
            "confianca": round(float(box.conf[0]), 3),
            "bbox": [round(v, 1) for v in box.xyxy[0].tolist()],
        }
        for box in result.boxes
    ]

    frame_anotado = desenhar_anotacoes(imagem, deteccoes, alertas, tempo_total_ms, duracao)

    ok, png = cv2.imencode(".png", frame_anotado)
    if not ok:
        raise HTTPException(status_code=500, detail="Falha ao gerar imagem anotada")

    return Response(content=png.tobytes(), media_type="image/png")