# Imagem base oficial do Python (versão slim para manter a imagem leve)
FROM python:3.10-slim

# Evita que o Python gere arquivos .pyc e força logs sem buffer no console
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH="runs/detect/train-8/weights/best.pt"

# Diretório de trabalho dentro do container
WORKDIR /app

# Instala dependências do sistema necessárias para OpenCV, XCB/X11 e healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libxcb1 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libsm6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copia o arquivo de dependências
COPY requirements.txt .

# 1. Atualiza o pip
# 2. Instala PyTorch e Torchvision especificamente para CPU (evita baixar ~4GB de CUDA/NVIDIA desnecessários)
# 3. Instala as dependências do projeto
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copia o restante do código da aplicação e pesos do modelo
COPY . .

# Expõe a porta padrão da API FastAPI
EXPOSE 8000

# Healthcheck para monitorar o status do container
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando de inicialização do servidor Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
