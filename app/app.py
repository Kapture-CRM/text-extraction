import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import v1_router
from app.core.config import settings
from app.core.logger import get_logger
from prometheus_fastapi_instrumentator import Instrumentator

logger = get_logger("text-extraction")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up Text Extraction Service 🔥")
    yield
    logger.info("Shutting down Text Extraction Service")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_TITLE,
        description=settings.APP_DESCRIPTION,
        version=settings.APP_VERSION,
        docs_url=f"{settings.API_BASE_PATH}/docs",
        redoc_url=f"{settings.API_BASE_PATH}/redoc",
        openapi_url=f"{settings.API_BASE_PATH}/openapi.json",
        root_path="/text-extraction",
        root_path_in_servers=True,
        swagger_ui_parameters={
            "displayRequestDuration": True,
            "displayOperationId": True,
        },
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_origin_regex=r"(https://.*\.kapturecrm\.com|https://.*\.vitos\.ai|https://.*\.kapdesk\.com)",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        client_host = request.client.host if request.client else "unknown"
        query = f"?{request.url.query}" if request.url.query else ""
        start = time.perf_counter()
        logger.info(f"--> {request.method} {request.url.path}{query} from {client_host}")
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                f"Unhandled error for {request.method} {request.url.path} "
                f"after {time.perf_counter() - start:.3f}s"
            )
            raise
        logger.info(
            f"<-- {request.method} {request.url.path} - {response.status_code} "
            f"({time.perf_counter() - start:.3f}s)"
        )
        return response

    app.include_router(v1_router, prefix=settings.API_BASE_PATH)

    Instrumentator().instrument(app).expose(app, endpoint="/internal/metrics", include_in_schema=False)

    @app.get(f"{settings.API_BASE_PATH}/health", tags=["Health"])
    def health():
        return {"status": "ok", "version": settings.APP_VERSION}

    return app
