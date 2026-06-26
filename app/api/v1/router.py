from fastapi import APIRouter

from app.api.v1.routers import pdf_extraction

# Register all v1 routers here as new pipelines are added.
v1_router = APIRouter()
v1_router.include_router(pdf_extraction.router)
