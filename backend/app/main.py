from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI

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


@app.get("/health")
async def health():
    return {"status": "ok"}

