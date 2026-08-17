from fastapi import FastAPI
# pyrefly: ignore [missing-import]
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager
from app.api.routes import router
from app.core.model import get_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = get_model()

    yield

    app.state.model = None


app = FastAPI(
    title="Detecção de EPIs",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}

