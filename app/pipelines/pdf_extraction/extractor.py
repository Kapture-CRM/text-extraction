import re
import time

import pdfplumber
from rank_bm25 import BM25Okapi

from app.core.logger import get_logger

logger = get_logger("pdf-extraction")

SECTION_PATTERN = re.compile(
    r'\b([IVXLCDM]+\s*[\.\.\·]\s+[A-Z][A-Za-z\s\(\)]{4,60})',
    re.IGNORECASE
)

BM25_K1 = 1.5
BM25_B = 0.75
HEADING_BOOST = 2.0

STOPWORDS = {
    'the','and','for','with','this','that','from','are','was','has',
    'not','all','per','may','can','also','only','each','any','its',
    'shall','been','have','row','col','section','columns'
}


def clean_cell(cell):
    if cell is None:
        return ""
    return " ".join(str(cell).split())


def forward_fill_column(table, col_idx):
    last_val = ""
    for row in table:
        if col_idx < len(row):
            val = clean_cell(row[col_idx])
            if val:
                last_val = val
            else:
                row[col_idx] = last_val
    return table


def is_header_row(row):
    non_empty = [c for c in row if c]
    if not non_empty:
        return False
    avg_len = sum(len(c) for c in non_empty) / len(non_empty)
    has_numbers = any(any(ch.isdigit() for ch in c) for c in non_empty)
    return avg_len < 40 and not has_numbers


def extract_heading_above(page, table_bbox, band_px=50):
    t_top = table_bbox[1]
    words = page.extract_words()
    hw = sorted(
        [w for w in words if (t_top - band_px) <= w['top'] < t_top],
        key=lambda w: (w['top'], w['x0'])
    )
    raw = " ".join(w['text'] for w in hw).strip()
    match = SECTION_PATTERN.search(raw)
    return match.group(1).strip() if match else ""


def extract_intra_heading(raw_table):
    if not raw_table:
        return "", raw_table
    first_row = raw_table[0]
    non_none = [c for c in first_row if c is not None and str(c).strip()]
    if len(non_none) == 1 and len(first_row) > 2:
        return clean_cell(non_none[0]), raw_table[1:]
    return "", raw_table


def _word_in_bbox(word, bbox):
    x0, top, x1, bottom = bbox
    return word['x0'] >= x0 - 1 and word['x1'] <= x1 + 1 and word['top'] >= top - 1 and word['bottom'] <= bottom + 1


def extract_paragraph_text(page, table_bboxes: list[tuple], line_gap_px: float = 8.0) -> str:
    words = page.extract_words()
    non_table_words = [
        w for w in words
        if not any(_word_in_bbox(w, bbox) for bbox in table_bboxes)
    ]
    if not non_table_words:
        return ""

    lines: dict[float, list] = {}
    for w in sorted(non_table_words, key=lambda w: (round(w['top']), w['x0'])):
        key = round(w['top'])
        lines.setdefault(key, []).append(w)

    sorted_tops = sorted(lines.keys())
    paragraphs = []
    current_para: list[str] = []
    prev_bottom = None
    for top in sorted_tops:
        line_words = sorted(lines[top], key=lambda w: w['x0'])
        line_text = " ".join(w['text'] for w in line_words).strip()
        if not line_text:
            continue
        line_bottom = max(w['bottom'] for w in line_words)
        if prev_bottom is not None and (top - prev_bottom) > line_gap_px:
            if current_para:
                paragraphs.append(" ".join(current_para))
                current_para = []
        current_para.append(line_text)
        prev_bottom = line_bottom
    if current_para:
        paragraphs.append(" ".join(current_para))

    return "\n\n".join(p for p in paragraphs if p)


def get_all_tables_with_metadata(pdf_path: str, max_pages: int | None = None) -> list[dict]:
    start = time.perf_counter()
    all_tables = []
    paragraph_pages = []
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        pages = pdf.pages[:max_pages] if max_pages else pdf.pages
        if max_pages and max_pages < total_pages:
            logger.info(
                f"Opened PDF with {total_pages} page(s): {pdf_path}, "
                f"limiting extraction to first {max_pages} page(s)"
            )
        else:
            logger.info(f"Opened PDF with {total_pages} page(s): {pdf_path}")
        for page_num, page in enumerate(pages, start=1):
            raw_tables = page.extract_tables() or []
            table_objs = page.find_tables()
            prev_bottom = -999
            table_bboxes = []
            for t_idx, (raw, tobj) in enumerate(zip(raw_tables, table_objs)):
                bbox = tobj.bbox
                table_bboxes.append(bbox)
                gap = bbox[1] - prev_bottom
                above_heading = extract_heading_above(page, bbox)
                intra_heading, raw_stripped = extract_intra_heading(raw)
                heading = above_heading or intra_heading
                num_cols = len(raw_stripped[0]) if raw_stripped else 0
                is_cont = not heading and (bbox[1] < 15 or gap < 15)
                all_tables.append({
                    "page_num": page_num,
                    "table_idx": t_idx + 1,
                    "raw_rows": raw_stripped,
                    "bbox": bbox,
                    "heading": heading,
                    "num_cols": num_cols,
                    "is_continuation": is_cont,
                    "gap_from_prev": gap,
                })
                prev_bottom = bbox[3]
            if raw_tables:
                logger.info(f"Page {page_num}: found {len(raw_tables)} table(s)")

            paragraph_text = extract_paragraph_text(page, table_bboxes)
            if paragraph_text:
                paragraph_pages.append({"page_num": page_num, "text": paragraph_text})
    logger.info(
        f"Extracted {len(all_tables)} table(s) and paragraph text from "
        f"{len(paragraph_pages)} page(s) from {pdf_path} in {time.perf_counter() - start:.2f}s"
    )
    return all_tables, paragraph_pages


def group_by_section(all_tables: list[dict]) -> list[dict]:
    sections, current = [], None
    for t in all_tables:
        if t['heading']:
            current = {"heading": t['heading'], "tables": [t]}
            sections.append(current)
        elif current and t['is_continuation']:
            if t['num_cols'] == current['tables'][-1]['num_cols']:
                current['tables'].append(t)
            else:
                current = {"heading": "", "tables": [t]}
                sections.append(current)
        else:
            current = {"heading": "", "tables": [t]}
            sections.append(current)
    return sections


def table_to_labeled_text(merged_rows, heading=""):
    if not merged_rows:
        return ""
    table = [list(row) for row in merged_rows]
    num_cols = max(len(row) for row in table)
    for col_idx in range(num_cols):
        table = forward_fill_column(table, col_idx)
    cleaned = [[clean_cell(c) for c in row] for row in table]
    cleaned = [row for row in cleaned if any(c for c in row)]
    if not cleaned:
        return ""
    num_cols = max(len(row) for row in cleaned)
    lines = []
    if heading:
        lines.append(f"SECTION: {heading}")
        lines.append("")
    if num_cols == 2:
        for row in cleaned:
            k = row[0] if len(row) > 0 else ""
            v = row[1] if len(row) > 1 else ""
            if k and v:
                lines.append(f"{k}: {v}")
            elif k:
                lines.append(f"{k}: (not available)")
        return "\n".join(lines)
    n_hdr = 0
    for row in cleaned:
        if is_header_row(row):
            n_hdr += 1
        else:
            break
    n_hdr = max(1, min(n_hdr, 2))
    header_rows, data_rows = cleaned[:n_hdr], cleaned[n_hdr:]
    if num_cols == 4 and n_hdr == 1:
        dkv = sum(1 for r in data_rows if len(r) >= 4 and r[0] and r[2])
        if dkv >= len(data_rows) // 2:
            for row in data_rows:
                p = row + [""] * (4 - len(row))
                if p[0] and p[1]:
                    lines.append(f"{p[0]}: {p[1]}")
                if p[2] and p[3]:
                    lines.append(f"{p[2]}: {p[3]}")
            return "\n".join(lines)
    headers = []
    for ci in range(num_cols):
        parts = []
        for hr in header_rows:
            v = hr[ci] if ci < len(hr) else ""
            if v and v not in parts:
                parts.append(v)
        headers.append(" / ".join(parts) if parts else f"Col{ci+1}")
    lines.append("COLUMNS: " + " | ".join(h for h in headers if h))
    lines.append("")
    for ri, row in enumerate(data_rows):
        padded = row + [""] * (num_cols - len(row))
        pairs = [
            f"{headers[i]}={padded[i]}"
            for i in range(num_cols)
            if headers[i] and padded[i] and headers[i] != padded[i]
        ]
        if pairs:
            lines.append(f"Row {ri+1}: " + " | ".join(pairs))
    return "\n".join(lines)


def group_paragraphs_into_sections(paragraph_pages: list[dict], max_chars: int = 1500) -> list[dict]:
    sections = []
    for page in paragraph_pages:
        page_num = page['page_num']
        remaining = page['text']
        chunk_idx = 0
        while remaining:
            if len(remaining) <= max_chars:
                chunk, remaining = remaining, ""
            else:
                split_at = remaining.rfind("\n\n", 0, max_chars)
                if split_at <= 0:
                    split_at = max_chars
                chunk, remaining = remaining[:split_at].strip(), remaining[split_at:].strip()
            if not chunk:
                continue
            chunk_idx += 1
            match = SECTION_PATTERN.search(chunk)
            heading = match.group(1).strip() if match else f"Page {page_num} text"
            if chunk_idx > 1:
                heading = f"{heading} (cont.)"
            sections.append({"heading": heading, "content": chunk, "pages": [page_num]})
    return sections


def build_section_store(pdf_path: str, max_pages: int | None = None) -> list[dict]:
    all_tables, paragraph_pages = get_all_tables_with_metadata(pdf_path, max_pages=max_pages)
    table_sections = group_by_section(all_tables)
    logger.info(f"Grouped {len(all_tables)} table(s) into {len(table_sections)} section(s)")

    store = []
    sec_id = 0
    for sec in table_sections:
        merged_rows = []
        for t in sec['tables']:
            merged_rows.extend(t['raw_rows'])
        content = table_to_labeled_text(merged_rows, heading=sec['heading'])
        if not content:
            logger.info(f"Table section ({sec['heading'] or 'untitled'}) produced no content, skipping")
            continue
        sec_id += 1
        pages = list(dict.fromkeys(t['page_num'] for t in sec['tables']))
        store.append({
            'id': sec_id,
            'heading': sec['heading'] or f'Section {sec_id}',
            'content': content,
            'pages': pages,
            'tokens': len(content) // 4,
            'type': 'table',
        })

    paragraph_sections = group_paragraphs_into_sections(paragraph_pages)
    for sec in paragraph_sections:
        sec_id += 1
        store.append({
            'id': sec_id,
            'heading': sec['heading'],
            'content': sec['content'],
            'pages': sec['pages'],
            'tokens': len(sec['content']) // 4,
            'type': 'paragraph',
        })

    logger.info(
        f"Built section store with {len(store)} section(s) "
        f"({len(store) - len(paragraph_sections)} table, {len(paragraph_sections)} paragraph) from {pdf_path}"
    )
    return store


def tokenize(text: str) -> list[str]:
    return [
        w for w in re.findall(r'[a-zA-Z]{3,}', text.lower())
        if w not in STOPWORDS
    ]


def content_fingerprint(content: str) -> str:
    return re.sub(r'\s+', ' ', content).strip().lower()


def build_bm25_index(store: list[dict]):
    start = time.perf_counter()
    corpus = []
    heading_token_sets = []
    for sec in store:
        heading_tokens = tokenize(sec['heading'])
        content_tokens = tokenize(sec['content'])
        doc_tokens = heading_tokens * 2 + content_tokens
        corpus.append(doc_tokens)
        heading_token_sets.append(set(heading_tokens))
    bm25 = BM25Okapi(corpus, k1=BM25_K1, b=BM25_B)
    logger.info(
        f"Built BM25 index over {len(store)} section(s) in {time.perf_counter() - start:.3f}s"
    )
    return bm25, heading_token_sets


def bm25_search(
    query: str,
    bm25: BM25Okapi,
    store: list[dict],
    htoksets: list[set],
    top_k: int = 2,
    min_score: float = 0.1,
) -> list[dict]:
    query_tokens = tokenize(query)
    if not query_tokens:
        logger.info(f"Query {query!r} produced no usable tokens after stopword filtering")
        return []

    raw_scores = bm25.get_scores(query_tokens)
    query_token_set = set(query_tokens)
    boosted = []
    for i, (score, sec) in enumerate(zip(raw_scores, store)):
        if score < min_score:
            continue
        heading_hit = bool(query_token_set & htoksets[i])
        final_score = score * (HEADING_BOOST if heading_hit else 1.0)
        boosted.append((final_score, score, heading_hit, sec))

    boosted.sort(key=lambda x: -x[0])
    results = []
    seen_content = set()
    skipped_duplicates = 0
    for final_score, raw_score, heading_hit, sec in boosted:
        fingerprint = content_fingerprint(sec['content'])
        if fingerprint in seen_content:
            skipped_duplicates += 1
            continue
        seen_content.add(fingerprint)
        results.append({**sec, '_score': final_score, '_heading_hit': heading_hit})
        if len(results) >= top_k:
            break

    logger.info(
        f"Query {query!r}: {len(boosted)} section(s) above min_score={min_score}, "
        f"skipped {skipped_duplicates} duplicate(s), returning top {len(results)} (top_k={top_k})"
    )
    return results
