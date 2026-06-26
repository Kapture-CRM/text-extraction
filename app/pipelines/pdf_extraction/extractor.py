import re
import pdfplumber
from rank_bm25 import BM25Okapi

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


def get_all_tables_with_metadata(pdf_path: str) -> list[dict]:
    all_tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            raw_tables = page.extract_tables() or []
            table_objs = page.find_tables()
            prev_bottom = -999
            for t_idx, (raw, tobj) in enumerate(zip(raw_tables, table_objs)):
                bbox = tobj.bbox
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
    return all_tables


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


def build_section_store(pdf_path: str) -> list[dict]:
    all_tables = get_all_tables_with_metadata(pdf_path)
    sections = group_by_section(all_tables)
    store = []
    for sec_id, sec in enumerate(sections, start=1):
        merged_rows = []
        for t in sec['tables']:
            merged_rows.extend(t['raw_rows'])
        content = table_to_labeled_text(merged_rows, heading=sec['heading'])
        if not content:
            continue
        pages = list(dict.fromkeys(t['page_num'] for t in sec['tables']))
        store.append({
            'id': sec_id,
            'heading': sec['heading'] or f'Section {sec_id}',
            'content': content,
            'pages': pages,
            'tokens': len(content) // 4,
        })
    return store


def tokenize(text: str) -> list[str]:
    return [
        w for w in re.findall(r'[a-zA-Z]{3,}', text.lower())
        if w not in STOPWORDS
    ]


def build_bm25_index(store: list[dict]):
    corpus = []
    heading_token_sets = []
    for sec in store:
        heading_tokens = tokenize(sec['heading'])
        content_tokens = tokenize(sec['content'])
        doc_tokens = heading_tokens * 2 + content_tokens
        corpus.append(doc_tokens)
        heading_token_sets.append(set(heading_tokens))
    bm25 = BM25Okapi(corpus, k1=BM25_K1, b=BM25_B)
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
    for final_score, raw_score, heading_hit, sec in boosted[:top_k]:
        results.append({**sec, '_score': final_score, '_heading_hit': heading_hit})

    return results
