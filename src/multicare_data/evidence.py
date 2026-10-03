import re


def exact_evidence(text: str, quote: str, start: int | None, end: int | None) -> bool:
    return (type(start) is int and type(end) is int and 0 <= start < end <= len(text)
            and bool(quote.strip()) and text[start:end] == quote)


def sentence_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(r"[^.!?\n]+(?:[.!?]+|$)", text) if m[0].strip()]


def unique_quote_span(text: str, quote: str) -> tuple[int, int] | None:
    if not quote or text.count(quote) != 1:
        return None
    start = text.index(quote)
    return start, start + len(quote)

