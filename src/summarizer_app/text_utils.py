from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List


_SENTENCE_RE = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?", re.UNICODE)
_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z0-9]+", re.UNICODE)


def normalize_text(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    text = normalize_text(text)
    sentences = [m.group(0).strip() for m in _SENTENCE_RE.finditer(text)]
    if sentences:
        return [s for s in sentences if s]
    return [text] if text else []


def tokenize(text: str) -> List[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t.strip()]


def build_term_frequency(texts: Iterable[str]) -> Counter:
    counter: Counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    return counter


def clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def estimate_token_budget(target_length: int | None, fallback_default: int = 96) -> int:
    if target_length is None:
        return fallback_default
    return clamp(int(target_length * 1.2), 32, 256)
