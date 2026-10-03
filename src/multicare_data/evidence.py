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


def align_whitespace_quote(text: str, quote: str) -> tuple[int, int] | None:
    """Map a unique whitespace-only match back to untouched source offsets."""
    def normalized(value):
        chars, offsets = [], []
        for match in re.finditer(r"(?P<space>(?:\s|\\r\\n|\\[nr])+)|.", value, re.DOTALL):
            token = match[0]
            chars.append(" " if match.lastgroup == "space" else token)
            offsets.append((match.start(), match.end()))
        return "".join(chars), offsets

    if not quote.strip():
        return None
    source, offsets = normalized(text)
    needle, _ = normalized(quote)
    needle = needle.strip()
    if not needle or source.count(needle) != 1:
        return None
    start = source.index(needle)
    span = offsets[start][0], offsets[start + len(needle) - 1][1]
    return span if unique_quote_span(text, text[span[0]:span[1]]) == span else None

