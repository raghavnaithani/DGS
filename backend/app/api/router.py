from fastapi import APIRouter

from .simulation import router as simulation_router
from .jobs import router as jobs_router
from .knowledge import router as knowledge_router
from .intake import router as intake_router
from .graph import router as graph_router, share_router as graph_share_router

router = APIRouter()
router.include_router(jobs_router, prefix="/jobs", tags=["jobs"])
router.include_router(knowledge_router, prefix="/knowledge", tags=["knowledge"])
router.include_router(intake_router, prefix="/intake", tags=["intake"])
router.include_router(simulation_router, prefix="/simulate", tags=["simulate"])
router.include_router(graph_router, prefix="/graph", tags=["graph"])
router.include_router(graph_share_router, prefix="/share", tags=["share"])
