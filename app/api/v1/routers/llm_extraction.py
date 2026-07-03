import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, HttpUrl

from app.core.logger import get_logger
from app.pipelines.llm_extraction.extractor import (
    extract_from_image_async,
    extract_from_pdf_async,
    validate_records,
)

logger = get_logger("llm-extraction")

router = APIRouter(prefix="/llm", tags=["LLM Extraction"])

SUPPORTED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg"}

CONTENT_TYPE_SUFFIXES = {
    "application/pdf": ".pdf",
    "image/png": ".png",
    "image/jpeg": ".jpg",
}


class ExtractGenericRequest(BaseModel):
    url: HttpUrl


def _infer_suffix(url: str, content_type: str | None) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in SUPPORTED_SUFFIXES:
        return suffix
    if content_type:
        content_type = content_type.split(";")[0].strip().lower()
        if content_type in CONTENT_TYPE_SUFFIXES:
            return CONTENT_TYPE_SUFFIXES[content_type]
    return suffix


@router.post("/extract/generic", summary="Extract line items from a document URL using an LLM")
async def extract_generic(body: ExtractGenericRequest):
    url = str(body.url)

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as http_client:
            response = await http_client.get(url)
            response.raise_for_status()
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Failed to download file from URL: {e}")

    contents = response.content
    if not contents:
        raise HTTPException(status_code=400, detail="Downloaded file is empty.")

    suffix = _infer_suffix(url, response.headers.get("content-type"))
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported or undetected file type: {suffix or 'unknown'}. "
                   f"Supported types: {sorted(SUPPORTED_SUFFIXES)}",
        )

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        if suffix == ".pdf":
            records, mode = await extract_from_pdf_async(tmp_path)
        else:
            records, mode = await extract_from_image_async(tmp_path)
    except Exception as e:
        logger.error(f"Extraction failed for {url}: {e}")
        raise HTTPException(status_code=502, detail=f"Extraction failed: {e}")
    finally:
        os.unlink(tmp_path)

    issues = validate_records(records)

    return {
        "url": url,
        "mode": mode,
        "records": records,
        "validation_issues": issues,
    }
