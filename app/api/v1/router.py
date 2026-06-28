from fastapi import APIRouter, Depends

from app.api.v1.routers import auth, pdf_extraction
from app.core.auth import get_current_user

# Register all v1 routers here as new pipelines are added.
v1_router = APIRouter()
v1_router.include_router(auth.router)
v1_router.include_router(pdf_extraction.router, dependencies=[Depends(get_current_user)])
