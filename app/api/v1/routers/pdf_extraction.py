import os
import tempfile
import time

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.core.logger import get_logger
from app.pipelines.pdf_extraction.extractor import (
    build_bm25_index,
    build_section_store,
    bm25_search,
)

logger = get_logger("pdf-extraction")

router = APIRouter(prefix="/pdf", tags=["PDF Extraction"])


@router.post("/extract", summary="Search a PDF document by keyword")
async def extract_context(
    file: UploadFile | None = File(None, description="PDF document to search"),
    url: str | None = Form(None, description="URL of a PDF document to search (alternative to file upload)"),
    query: str = Form(..., description="Keyword or question to search for"),
    top_k: int = Form(2, description="Number of top sections to return"),
    min_score: float = Form(0.1, description="Minimum BM25 score threshold"),
    max_pages: int | None = Form(
        None, description="Only process the first N pages of the PDF. Omit to process all pages."
    ),
):
    if max_pages is not None and max_pages <= 0:
        raise HTTPException(status_code=400, detail="max_pages must be a positive integer.")

    if file is not None and url:
        raise HTTPException(status_code=400, detail="Provide either 'file' or 'url', not both.")
    if file is None and not url:
        raise HTTPException(status_code=400, detail="Either 'file' or 'url' is required.")

    request_start = time.perf_counter()
    source_name = file.filename if file is not None else url
    logger.info(
        f"Received /pdf/extract request: source={source_name!r}, query={query!r}, "
        f"top_k={top_k}, min_score={min_score}, max_pages={max_pages}"
    )

    if file is not None:
        if not file.filename.lower().endswith(".pdf"):
            logger.warning(f"Rejected non-PDF upload: {file.filename!r}")
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        contents = await file.read()
        if not contents:
            logger.warning(f"Rejected empty upload: {file.filename!r}")
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    else:
        logger.info(f"Downloading PDF from url={url!r}")
        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as http_client:
                response = await http_client.get(url)
                response.raise_for_status()
        except httpx.HTTPError as e:
            logger.warning(f"Failed to download PDF from url={url!r}: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to download file from URL: {e}")

        contents = response.content
        if not contents:
            logger.warning(f"Downloaded file is empty for url={url!r}")
            raise HTTPException(status_code=400, detail="Downloaded file is empty.")

        content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
        if not url.lower().endswith(".pdf") and content_type != "application/pdf":
            logger.warning(f"Rejected non-PDF url={url!r} (content-type={content_type!r})")
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name
    logger.info(f"Step 1/4: saved {len(contents)} bytes to temp file {tmp_path}")

    try:
        logger.info("Step 2/4: building section store from PDF tables")
        section_store = build_section_store(tmp_path, max_pages=max_pages)
        if not section_store:
            logger.warning(f"No table-based sections found in {source_name!r}")
            raise HTTPException(
                status_code=422,
                detail="No table-based sections found in the PDF.",
            )

        logger.info("Step 3/4: building BM25 index and searching")
        bm25_index, heading_token_sets = build_bm25_index(section_store)
        results = bm25_search(
            query=query,
            bm25=bm25_index,
            store=section_store,
            htoksets=heading_token_sets,
            top_k=top_k,
            min_score=min_score,
        )
        logger.info("Step 4/4: formatting response")
    except Exception:
        logger.exception(f"Failed to process {source_name!r} for query {query!r}")
        raise
    finally:
        os.unlink(tmp_path)
        logger.info(f"Cleaned up temp file {tmp_path}")

    if not results:
        logger.info(f"No matches for query {query!r} in {source_name!r} ({time.perf_counter() - request_start:.2f}s)")
        return JSONResponse(
            status_code=200,
            content={
                "query": query,
                "source": source_name,
                "max_pages": max_pages,
                "total_sections_indexed": len(section_store),
                "matches": [],
                "context": "",
                "message": "No sections matched the query above the score threshold.",
            },
        )

    context = "\n\n".join(
        f"[Section: {s['heading']} | Pages: {', '.join(str(p) for p in s['pages'])}]\n{s['content']}"
        for s in results
    )

    logger.info(
        f"Returning {len(results)} match(es) for query {query!r} in {source_name!r} "
        f"({time.perf_counter() - request_start:.2f}s)"
    )

    return {
        "query": query,
        "source": source_name,
        "max_pages": max_pages,
        "total_sections_indexed": len(section_store),
        "matches": [
            {
                "id": s["id"],
                "heading": s["heading"],
                "pages": s["pages"],
                "score": round(s["_score"], 4),
                "heading_hit": s["_heading_hit"],
                "tokens": s["tokens"],
                "content": s["content"],
            }
            for s in results
        ],
        "context": context,
    }
