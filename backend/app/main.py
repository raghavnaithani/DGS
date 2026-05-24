from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.router import router as api_router
from .database.connection import initialize_database
from .models import Alternative, DecisionNode, KnowledgeChunk, Risk, UserIntent

logger = logging.getLogger(__name__)

PHASE1_MODELS = (Alternative, DecisionNode, KnowledgeChunk, Risk, UserIntent)


@asynccontextmanager
async def lifespan(app: FastAPI):
    database_connection = initialize_database()
    database_connection.close()
    logger.info("Models loaded and database ready")
    yield

app = FastAPI(title="DGS Backend", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:3000",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router, prefix="/v1")


@app.get("/health")
async def health():
    return {"status": "ok"}

