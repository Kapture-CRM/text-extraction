import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler

from app.core.config import settings

_FORMATTER = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _build_file_handler() -> TimedRotatingFileHandler:
    os.makedirs(settings.LOG_DIR, exist_ok=True)
    handler = TimedRotatingFileHandler(
        filename=os.path.join(settings.LOG_DIR, "app.log"),
        when="midnight",
        interval=1,
        backupCount=settings.LOG_RETENTION_DAYS,
        encoding="utf-8",
    )
    handler.suffix = "%Y-%m-%d"
    handler.setFormatter(_FORMATTER)
    return handler


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(_FORMATTER)
    logger.addHandler(stream_handler)
    logger.addHandler(_build_file_handler())

    logger.propagate = False
    return logger
