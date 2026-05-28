from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .api.router import router as api_router
from .database.jobs_store import SQLiteJobStore
from .database.connection import initialize_database
from .database.vector_store import get_vector_store
from .engines.ingestion import IngestionService
from .engines.retriever import HybridRetriever
from .services.simulation_worker import SimulationJobWorker
from .models import Alternative, DecisionNode, KnowledgeChunk, Risk, UserIntent

logger = logging.getLogger(__name__)

PHASE1_MODELS = (Alternative, DecisionNode, KnowledgeChunk, Risk, UserIntent)


@asynccontextmanager
async def lifespan(app: FastAPI):
    database_connection = initialize_database()
    database_connection.close()
    job_store = SQLiteJobStore()
    vector_store = get_vector_store()
    simulation_worker = SimulationJobWorker(
        job_store=job_store,
        vector_store=vector_store,
        poll_interval_seconds=float(settings.simulation_worker_poll_interval_seconds),
    )
    app.state.job_store = job_store
    app.state.vector_store = vector_store
    app.state.ingestion_service = IngestionService(job_store=job_store, vector_store=vector_store)
    app.state.retriever = HybridRetriever(vector_store=vector_store, db_path=job_store.db_path)
    app.state.simulation_worker = simulation_worker
    app.state.background_tasks = set()
    simulation_worker.start()
    logger.info("Models loaded and database ready")
    yield
    simulation_worker.stop()

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

