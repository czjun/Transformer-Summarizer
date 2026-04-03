from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .text_utils import build_term_frequency, split_sentences, tokenize


@dataclass
class SimpleSummaryResult:
    summary: str
    selected_sentences: List[str]


class SimpleExtractiveSummarizer:
    def __init__(self, max_sentences: int = 3):
        self.max_sentences = max_sentences

    def summarize(self, text: str, target_length: int | None = None) -> SimpleSummaryResult:
        sentences = split_sentences(text)
        if not sentences:
            return SimpleSummaryResult(summary="", selected_sentences=[])
        if len(sentences) == 1:
            return SimpleSummaryResult(summary=sentences[0], selected_sentences=sentences)

        freq = build_term_frequency(sentences)
        scored = []
        for idx, sentence in enumerate(sentences):
            tokens = tokenize(sentence)
            if not tokens:
                score = 0.0
            else:
                score = sum(freq[token] for token in tokens) / len(tokens)
            scored.append((score, idx, sentence))

        scored.sort(key=lambda item: (-item[0], item[1]))
        selected = sorted(scored[: self.max_sentences], key=lambda item: item[1])
        if target_length is not None:
            summary, kept = self._fit_target_length(selected, target_length)
            return SimpleSummaryResult(summary=summary, selected_sentences=kept)
        kept_sentences = [item[2] for item in selected]
        return SimpleSummaryResult(summary="".join(kept_sentences), selected_sentences=kept_sentences)

    def _fit_target_length(self, selected, target_length: int) -> tuple[str, List[str]]:
        kept: List[str] = []
        total = 0
        for _, _, sentence in selected:
            next_len = total + len(sentence)
            if kept and next_len > target_length:
                break
            kept.append(sentence)
            total = next_len
        if not kept:
            kept = [selected[0][2]]
        return "".join(kept), kept
