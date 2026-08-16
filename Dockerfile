# Imagem base oficial do Python (versão slim para manter a imagem leve)
FROM python:3.10-slim

# Evita que o Python gere arquivos .pyc e força logs sem buffer no console
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH="runs/detect/train-8/weights/best.pt"

# Diretório de trabalho dentro do container
WORKDIR /app

# Instala dependências do sistema necessárias para o OpenCV e processamento de imagem
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copia apenas o requirements primeiro para aproveitar o cache de camadas do Docker
COPY requirements.txt .

# Instala as dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia o restante do código da aplicação e artefatos necessários
COPY . .

# Expõe a porta padrão da API FastAPI
EXPOSE 8000

# Healthcheck para monitorar o status do container
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando de inicialização do servidor Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
