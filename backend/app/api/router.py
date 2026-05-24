from fastapi import APIRouter

from .intake import router as intake_router

router = APIRouter()
router.include_router(intake_router, prefix="/intake", tags=["intake"])
