import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.logger import get_logger
from app.pipelines.llm_extraction.extractor import (
    extract_from_image_async,
    extract_from_pdf_async,
    validate_records,
)

logger = get_logger("llm-extraction")

router = APIRouter(prefix="/llm", tags=["LLM Extraction"])

SUPPORTED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg"}


@router.post("/extract/generic", summary="Extract line items from a document using an LLM")
async def extract_generic(
    file: UploadFile = File(..., description="PDF or image document to extract line items from"),
):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Supported types: {sorted(SUPPORTED_SUFFIXES)}",
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        if suffix == ".pdf":
            records, mode = await extract_from_pdf_async(tmp_path)
        else:
            records, mode = await extract_from_image_async(tmp_path)
    except Exception as e:
        logger.error(f"Extraction failed for {file.filename}: {e}")
        raise HTTPException(status_code=502, detail=f"Extraction failed: {e}")
    finally:
        os.unlink(tmp_path)

    issues = validate_records(records)

    return {
        "filename": file.filename,
        "mode": mode,
        "records": records,
        "validation_issues": issues,
    }
