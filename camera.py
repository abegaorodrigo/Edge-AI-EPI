"""
Cliente de Câmera / Edge AI para Detecção de EPIs
=================================================
Este script captura frames da webcam em tempo real e os envia via HTTP POST
para a API FastAPI rodando no Docker (ou localmente).

Toda a renderização de bounding boxes, cores por classe e diagnósticos
de conformidade (verde/vermelho) agora é gerada de forma centralizada e
padronizada diretamente pela API (/predict/annotated).
"""

import os
import time
import cv2
import numpy as np
import requests

# Endpoint da API configurável por variável de ambiente
API_URL = os.getenv("API_URL", "http://localhost:8000/predict/annotated")


def desenhar_banner_erro(frame, mensagem):
    """Exibe banner de aviso na tela caso a API esteja indisponível."""
    h, w, _ = frame.shape
    cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 180), -1)
    cv2.putText(
        frame,
        mensagem,
        (15, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main():
    print("=" * 60)
    print("🛡️  Cliente Câmera Edge-AI EPI")
    print("=" * 60)
    print(f"📡 Endpoint conectado: {API_URL}")
    print("ℹ️  Renderização de bounding boxes gerenciada centralmente pela API.")
    print("⌨️  Teclas: 'q' ou 'ESC' para sair.")
    print("=" * 60)

    # Inicializa sessão HTTP para reaproveitamento de conexão TCP (baixa latência)
    session = requests.Session()

    # 1. Abre a webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌ Erro ao abrir a webcam. Verifique se está conectada ou em uso por outro aplicativo.")
        return

    print("🎥 Câmera iniciada com sucesso! Transmitindo para a API...")

    while True:
        t_inicio = time.time()
        success, frame = cap.read()

        if not success:
            print("❌ Falha ao capturar frame da webcam.")
            break

        # 2. Codifica o frame capturado em JPEG na memória
        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            continue

        frame_exibicao = frame

        # 3. Envia para a API que retorna a imagem já anotada com o padrão oficial
        try:
            response = session.post(
                API_URL,
                files={"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")},
                timeout=2.5,
            )

            tempo_total_ms = (time.time() - t_inicio) * 1000.0

            if response.status_code == 200:
                # Decodifica a imagem anotada retornada pelo servidor
                imagem_anotada = cv2.imdecode(
                    np.frombuffer(response.content, np.uint8),
                    cv2.IMREAD_COLOR,
                )
                if imagem_anotada is not None:
                    frame_exibicao = imagem_anotada
            else:
                desenhar_banner_erro(frame_exibicao, f"Erro {response.status_code} da API: {response.text}")

        except requests.exceptions.ConnectionError:
            desenhar_banner_erro(frame_exibicao, "❌ API Offline. Verifique se o Docker está rodando (docker compose up -d)")
        except requests.exceptions.Timeout:
            desenhar_banner_erro(frame_exibicao, "⚠️ Timeout ao comunicar com a API")
        except Exception as e:
            desenhar_banner_erro(frame_exibicao, f"Erro: {str(e)}")

        # 4. Exibe o vídeo anotado na janela
        cv2.imshow("Edge AI EPI - Monitoramento em Tempo Real", frame_exibicao)

        # 5. Captura de teclas para encerramento
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # 'q' ou ESC
            print("\nEncerrando visualização...")
            break

    # Libera os recursos
    cap.release()
    cv2.destroyAllWindows()
    session.close()
    print("Aplicação finalizada.")


if __name__ == "__main__":
    main()
