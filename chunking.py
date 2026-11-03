import bisect


# universal sliding window chunker — no domain assumptions, works on any document
def sliding_window_chunks(text, token_limit=128, overlap=32):
    words = text.split()
    chunks = []
    start = 0
    step = token_limit - overlap

    while start < len(words):
        chunk_words = words[start:start + token_limit]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
        start += step

    return chunks


def page_aware_chunks(words, page_boundaries, token_limit=128, overlap=32):
    """
    Sliding window chunker that returns (chunk_text, page_num) tuples.

    page_boundaries: list of (start_word_idx, page_num) sorted ascending,
                     built by the caller as it extracts text page-by-page.
    """
    if not page_boundaries:
        return [(c, 1) for c in sliding_window_chunks(" ".join(words), token_limit, overlap)]

    boundary_starts = [b[0] for b in page_boundaries]
    boundary_pages  = [b[1] for b in page_boundaries]

    chunks = []
    step   = token_limit - overlap
    start  = 0

    while start < len(words):
        chunk_words = words[start:start + token_limit]
        if chunk_words:
            # last boundary whose start_word_idx <= chunk start word
            idx  = bisect.bisect_right(boundary_starts, start) - 1
            page = boundary_pages[idx] if idx >= 0 else boundary_pages[0]
            chunks.append((" ".join(chunk_words), page))
        start += step

    return chunks




