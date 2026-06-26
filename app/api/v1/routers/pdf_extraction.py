import os
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.pipelines.pdf_extraction.extractor import (
    build_bm25_index,
    build_section_store,
    bm25_search,
)

router = APIRouter(prefix="/pdf", tags=["PDF Extraction"])


@router.post("/extract", summary="Search a PDF document by keyword")
async def extract_context(
    file: UploadFile = File(..., description="PDF document to search"),
    query: str = Form(..., description="Keyword or question to search for"),
    top_k: int = Form(2, description="Number of top sections to return"),
    min_score: float = Form(0.1, description="Minimum BM25 score threshold"),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        section_store = build_section_store(tmp_path)
        if not section_store:
            raise HTTPException(
                status_code=422,
                detail="No table-based sections found in the PDF.",
            )

        bm25_index, heading_token_sets = build_bm25_index(section_store)
        results = bm25_search(
            query=query,
            bm25=bm25_index,
            store=section_store,
            htoksets=heading_token_sets,
            top_k=top_k,
            min_score=min_score,
        )
    finally:
        os.unlink(tmp_path)

    if not results:
        return JSONResponse(
            status_code=200,
            content={
                "query": query,
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

    return {
        "query": query,
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
