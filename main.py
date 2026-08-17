from fastapi import FastAPI
# pyrefly: ignore [missing-import]
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager
from app.api.routes import router
from app.core.model import get_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    # get_model() carrega os pesos e já executa o warm-up internamente
    app.state.model = get_model()

    yield

    app.state.model = None


app = FastAPI(
    title="Edge-AI-EPI",
    version="1.0.0",
    description="""
## 🛡️ API de Detecção de EPIs em Tempo Real

Sistema de **Edge AI** para detecção automática de Equipamentos de Proteção Individual (EPIs)
usando um modelo **YOLOv8** customizado treinado com ~93.5% mAP50.

### Endpoints disponíveis

| Endpoint | Tipo de resposta | Uso |
|---|---|---|
| `POST /predict` | JSON | Bounding boxes, confiança, alertas de conformidade e métricas de tempo |
| `POST /predict/annotated` | image/png | Frame visual anotado com caixas coloridas e HUD de status |

### Classes monitoradas
`helmet` · `vest` · `gloves` · `boots` · `human`
""",
    lifespan=lifespan,
    contact={"name": "Rodrigo Abegão", "url": "https://github.com/abegaorodrigo/Edge-AI-EPI"},
)

app.include_router(router)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}

