import asyncio
import json
import time
from pathlib import Path

import fitz  # pymupdf
from google import genai
from google.genai import types
from google.oauth2 import service_account

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger("llm-extraction")

# ---------------------------------------------------------------------------
# Quantity-like field names — used by validation to find the qty field
# regardless of what the header called it.
# ---------------------------------------------------------------------------

QTY_KEY_HINTS = {"quantity", "qty", "menge", "amount", "count", "anzahl", "pieces", "pcs"}

# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

_client = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        start = time.perf_counter()
        creds_info = json.loads(settings.GOOGLE_APPLICATION_CREDENTIALS_JSON)
        credentials = service_account.Credentials.from_service_account_info(
            creds_info,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        _client = genai.Client(
            vertexai=True,
            project=settings.GCP_PROJECT_ID,
            location=settings.GCP_LOCATION,
            credentials=credentials,
        )
        logger.info(f"Gemini client initialized in {time.perf_counter() - start:.2f}s")
    return _client


GEMINI_CONFIG = types.GenerateContentConfig(
    temperature=0,
    response_mime_type="application/json",
    thinking_config=types.ThinkingConfig(thinking_level="minimal"),
)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """
You are a data extraction engine. Read the document and return every line item as a JSON array.
The document may be a table, a plain list, or mixed. The language may be anything.

Output ONLY a valid JSON array. No markdown, no explanation, no extra text.
Every record must include "low_confidence".

SCHEMA DETECTION — apply these checks in order, stop at the first match.

Check 1 — Explicit headers present
If the document has a header row (column labels, not data), use those labels as JSON keys.
Normalize: lowercase, replace spaces and special characters with underscores, strip
leading/trailing underscores.

Check 2 — No headers, but column purpose is clear from the values
Scan all rows to understand what each column contains, then name it:
- Consistent short numeric-only values across all rows → "code"
- Consistent small integers across all rows → "quantity"
- Text that looks like product names or includes weights/volumes → "product_name"
- Text matching packaging notation → "pack_info"
- Short annotation words → "notes"
- A "x N" suffix alongside a quantity: merge into quantity, not a separate column.

Check 3 — Column purpose cannot be determined
Name columns "column1", "column2", etc. in left-to-right order. Do not skip any column.

EXTRACTION RULES

- One record per line item. Never merge two items into one record.
- Do not include header rows as records.
- Quantity can appear as a leading column, mid-column, or trailing suffix like "x 6".
  Always extract just the number.
- Remarks go into the notes/remarks column, never into the product name field.
- Free-text lines outside the main table:
  - Skip lines that are pure metadata with no product name.
  - For any line that names a product, extract it with low_confidence=true using the SAME
    keys already established for this document. Do not introduce new key names. Map values
    to the best-fit existing column, set unmatched columns to null, and add a "notes" key
    only for leftover text that has no matching column.
- Set missing values to null. Do not invent data.
- Preserve the original casing and spelling of product names.
- Set "low_confidence": true if text was blurry, handwritten, or ambiguous.
-  Ignore the customer names.
"""

IMAGE_PROMPT = (
    "You are an extraction engine. The image contains a product order.\n"
    + EXTRACTION_PROMPT
)


def build_pdf_prompt(text: str) -> str:
    return (
        "You are an extraction engine. Below is raw text extracted from a PDF product order.\n"
        + EXTRACTION_PROMPT
        + "\n\n--- PDF TEXT START ---\n" + text + "\n--- PDF TEXT END ---"
    )

# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------


def is_scanned_pdf(pdf_path: str) -> bool:
    doc = fitz.open(pdf_path)
    has_text = any(page.get_text("text").strip() for page in doc)
    doc.close()
    return not has_text


def extract_pdf_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        text = page.get_text("text").strip()
        pages.append(text)
    doc.close()
    return "\n\n--- PAGE BREAK ---\n\n".join(pages)

# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _get_response_text(response) -> str:
    try:
        return response.candidates[0].content.parts[0].text.strip()
    except (IndexError, AttributeError):
        return response.text.strip()


def _coerce_record(record: dict) -> dict:
    """
    Normalize field types across all schema modes.
    When headers are used (e.g. "menge"), the model sometimes returns numeric
    values as strings. Coerce any qty-hint field and any obviously numeric
    string value to int/float so validation and downstream code see numbers.
    """
    for key, val in record.items():
        if not isinstance(val, str):
            continue
        if key.lower() in QTY_KEY_HINTS:
            try:
                record[key] = int(val) if val.isdigit() else float(val)
            except ValueError:
                pass
        elif val.strip().lstrip('-').replace('.', '', 1).isdigit():
            try:
                record[key] = int(val) if '.' not in val else float(val)
            except ValueError:
                pass
    return record


def _parse_response(response) -> list:
    raw = _get_response_text(response)
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(cleaned)
    if isinstance(result, dict):
        result = [result]
    return [_coerce_record(r) for r in result]

# ---------------------------------------------------------------------------
# Extraction entry points
# ---------------------------------------------------------------------------


async def extract_from_image_async(path: str) -> tuple[list, str]:
    client = get_client()
    img_bytes = Path(path).read_bytes()
    mime_type = "image/png" if path.lower().endswith(".png") else "image/jpeg"

    start = time.perf_counter()
    response = await client.aio.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(data=img_bytes, mime_type=mime_type),
            IMAGE_PROMPT,
        ],
        config=GEMINI_CONFIG,
    )
    logger.info(f"Gemini vision call took {time.perf_counter() - start:.2f}s")
    return _parse_response(response), "vision"


async def extract_from_pdf_async(path: str) -> tuple[list, str]:
    if await asyncio.to_thread(is_scanned_pdf, path):
        return await extract_pdf_via_vision_async(path)

    client = get_client()
    parse_start = time.perf_counter()
    prompt = build_pdf_prompt(await asyncio.to_thread(extract_pdf_text, path))
    logger.info(f"PDF text extraction took {time.perf_counter() - parse_start:.2f}s")

    start = time.perf_counter()
    response = await client.aio.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=[prompt],
        config=GEMINI_CONFIG,
    )
    logger.info(f"Gemini pdf_text call took {time.perf_counter() - start:.2f}s")
    return _parse_response(response), "pdf_text"


def _render_pdf_pages_to_png(path: str) -> list[bytes]:
    doc = fitz.open(path)
    pages = [page.get_pixmap(matrix=fitz.Matrix(2, 2)).tobytes("png") for page in doc]
    doc.close()
    return pages


async def extract_pdf_via_vision_async(path: str) -> tuple[list, str]:
    client = get_client()
    render_start = time.perf_counter()
    pages = await asyncio.to_thread(_render_pdf_pages_to_png, path)
    logger.info(f"PDF page rendering ({len(pages)} pages) took {time.perf_counter() - render_start:.2f}s")

    async def _extract_page(img_bytes: bytes):
        start = time.perf_counter()
        response = await client.aio.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
                IMAGE_PROMPT,
            ],
            config=GEMINI_CONFIG,
        )
        logger.info(f"Gemini pdf_vision page call took {time.perf_counter() - start:.2f}s")
        return _parse_response(response)

    results = await asyncio.gather(*[_extract_page(b) for b in pages])
    all_records = [r for page_records in results for r in page_records]
    return all_records, "pdf_vision_fallback"

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _find_qty_value(record: dict):
    for key, val in record.items():
        if key.lower() in QTY_KEY_HINTS:
            return val
    return None


def _find_product_value(record: dict):
    for candidate in ("product_name", "product", "produkt", "item", "description",
                      "artikel", "name", "product_description"):
        if candidate in record and record[candidate]:
            return record[candidate]
    for key, val in record.items():
        if key not in ("low_confidence", "code") and isinstance(val, str) and val:
            return val
    return None


def validate_records(records: list) -> list:
    issues = []
    for i, r in enumerate(records):
        product = _find_product_value(r)
        if not product:
            issues.append(f"row {i}: no product name found")

        qty = _find_qty_value(r)
        is_low_conf = r.get("low_confidence", False)

        if qty is None and not is_low_conf:
            issues.append(
                f"row {i}: qty is null but low_confidence=false — product: {product!r}"
            )
        elif qty is not None:
            if not isinstance(qty, (int, float)):
                issues.append(f"row {i}: qty is not numeric → {qty!r}")
            elif qty <= 0:
                issues.append(f"row {i}: non-positive qty → {qty}")

    return issues
