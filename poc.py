from time import sleep

import cv2

from ultralytics import YOLO

# 1. Carrega o modelo que você treinou
# (Ajuste a pasta se o seu treino mais recente tiver outro nome, ex: train-5)
model = YOLO(r"runs\detect\train-8\weights\best.pt")

# 2. Abre a webcam (0 geralmente é a câmera padrão do notebook/computador)
# cap = cv2.VideoCapture(0)

# # Define a resolução da captura (opcional)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# if not cap.isOpened():
#     print("❌ Erro ao abrir a webcam. Verifique se ela está conectada ou em uso por outro app.")
#     exit()

# print("🎥 Webcam iniciada! Pressione a tecla 'q' na janela do vídeo para fechar.")

# while True:
#     # Lê um frame da webcam 
#     success, frame = cap.read()

#     if not success:
#         print("Erro ao capturar o frame.")
#         break

# 3. Faz a predição no frame atual (rodando na GPU RTX 5070 Ti)
frame = 'E:\\Projeto\\imagem_epi_teste.jpg'
results = model.predict(source=frame, conf=0.5, device="cpu")

for result in results:
    boxes = result.boxes
    names = result.names  # Mapeamento ID -> Nome da classe (ex: {0: 'person', 1: 'helmet'})

    persons = []
    helmets = []

    # 1. Separar detecções de pessoas e capacetes
    for box in boxes:
        cls_id = int(box.cls[0])
        class_name = names[cls_id]
        xyxy = box.xyxy[0].tolist()  # [xmin, ymin, xmax, ymax]

        if class_name.lower() in ['human', 'pessoa']:
            persons.append(xyxy)
        elif class_name.lower() in ['helmet', 'capacete', 'safety_helmet']:
            helmets.append(xyxy)
    print(persons, helmets)
    # 2. Verificar se algum capacete está associado a uma pessoa
    for p_xmin, p_ymin, p_xmax, p_ymax in persons:
        has_helmet = False
        for h_xmin, h_ymin, h_xmax, h_ymax in helmets:
            # Verifica se o centro do capacete está dentro do X da pessoa 
            # e na metade superior do corpo (Y)
            h_center_x = (h_xmin + h_xmax) / 2
            h_center_y = (h_ymin + h_ymax) / 2

            if (p_xmin <= h_center_x <= p_xmax) and (p_ymin <= h_center_y <= (p_ymin + (p_ymax - p_ymin) * 0.5)):
                has_helmet = True
                break

        if not has_helmet:
            print(f"Pessoa sem capacete detectada na posição: [{p_xmin:.1f}, {p_ymin:.1f}, {p_xmax:.1f}, {p_ymax:.1f}]")

# 4. Desenha as caixas de detecção no frame
annotated_frame = results[0].plot()

# 5. Exibe a imagem na janela popup
cv2.imshow("YOLOv8 Real-Time Detection - Webcam", annotated_frame)


# # Pressione a tecla 'q' no teclado para encerrar o loop
if cv2.waitKey(0) & 0xFF == ord('q'):
  
    exit(0)

# Libera a câmera e fecha a janela do OpenCV
# cap.release()
# cv2.destroyAllWindows()



# import cv2
# from ultralytics import YOLO

# # 1. Carrega o modelo que você treinou
# # (Ajuste a pasta se o seu treino mais recente tiver outro nome, ex: train-5)
# model = YOLO(r"runs\\detect\\train-8\\weights\\best.pt")

# # 2. Abre a webcam (0 geralmente é a câmera padrão do notebook/computador)
# cap = cv2.VideoCapture(0)

# # Define a resolução da captura (opcional)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# if not cap.isOpened():
#     print("❌ Erro ao abrir a webcam. Verifique se ela está conectada ou em uso por outro app.")
#     exit()

# print("🎥 Webcam iniciada! Pressione a tecla 'q' na janela do vídeo para fechar.")

# while True:
#     # Lê um frame da webcam
#     success, frame = cap.read()

#     if not success:
#         print("Erro ao capturar o frame.")
#         break

#     # 3. Faz a predição no frame atual (rodando na GPU RTX 5070 Ti)
#     results = model.predict(source=frame, conf=0.5, device=0)

#     # 4. Desenha as caixas de detecção no frame
#     annotated_frame = results[0].plot()
#     print(results[0])

#     # 5. Exibe a imagem na janela popup
#     cv2.imshow("YOLOv8 Real-Time Detection - Webcam", annotated_frame)

#     # Pressione a tecla 'q' no teclado para encerrar o loop
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Libera a câmera e fecha a janela do OpenCV
# cap.release()
# cv2.destroyAllWindows()
