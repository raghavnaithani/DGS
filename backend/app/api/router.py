from fastapi import APIRouter

from .jobs import router as jobs_router
from .knowledge import router as knowledge_router
from .intake import router as intake_router

router = APIRouter()
router.include_router(jobs_router, prefix="/jobs", tags=["jobs"])
router.include_router(knowledge_router, prefix="/knowledge", tags=["knowledge"])
router.include_router(intake_router, prefix="/intake", tags=["intake"])
