from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .config import SummarizationConfig
from .fallback import SimpleExtractiveSummarizer
from .text_utils import clamp, estimate_token_budget, normalize_text


try:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except Exception:  # pragma: no cover - dependency fallback
    torch = None
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None


@dataclass
class SummaryOutput:
    summary: str
    backend: str
    used_target_length: Optional[int]


class HybridSummarizer:
    def __init__(self, config: SummarizationConfig | None = None):
        self.config = config or SummarizationConfig()
        self.fallback = SimpleExtractiveSummarizer(max_sentences=self.config.fallback_sentences)
        self.tokenizer = None
        self.model = None
        self.backend_name = "fallback"
        self.device = self.config.device or ("cuda" if torch and torch.cuda.is_available() else "cpu")
        self._try_load_transformer()

    def _try_load_transformer(self) -> None:
        if AutoTokenizer is None or AutoModelForSeq2SeqLM is None:
            return
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_name)
            if torch is not None:
                self.model.to(self.device)
            self.backend_name = "transformer"
        except Exception:
            self.tokenizer = None
            self.model = None
            self.backend_name = "fallback"

    def summarize(self, text: str, target_length: int | None = None) -> SummaryOutput:
        text = normalize_text(text)
        if not text:
            return SummaryOutput(summary="", backend=self.backend_name, used_target_length=target_length)
        if self.backend_name == "transformer" and self.tokenizer and self.model:
            try:
                summary = self._summarize_with_transformer(text, target_length)
                return SummaryOutput(summary=summary, backend="transformer", used_target_length=target_length)
            except Exception:
                pass
        result = self.fallback.summarize(text, target_length=target_length)
        return SummaryOutput(summary=result.summary, backend="fallback", used_target_length=target_length)

    def summarize_batch(self, texts: List[str], target_length: int | None = None) -> List[SummaryOutput]:
        return [self.summarize(text, target_length=target_length) for text in texts]

    def _summarize_with_transformer(self, text: str, target_length: int | None) -> str:
        prompt = self._build_prompt(text, target_length)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_source_length,
        )
        if torch is not None:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        max_new_tokens = estimate_token_budget(target_length, self.config.max_target_length)
        min_new_tokens = max(16, int(max_new_tokens * 0.4))
        generated = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            num_beams=self.config.num_beams,
            length_penalty=self.config.length_penalty,
            no_repeat_ngram_size=self.config.no_repeat_ngram_size,
            early_stopping=True,
        )
        summary = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return summary.strip()

    def _build_prompt(self, text: str, target_length: int | None) -> str:
        if target_length is None:
            return f"请生成一段简洁摘要：{text}"
        return f"请根据目标长度 {target_length} 字生成摘要：{text}"

    def summarize_to_text(self, text: str, target_length: int | None = None) -> str:
        return self.summarize(text, target_length=target_length).summary

    def fallback_summary(self, text: str, target_length: int | None = None) -> str:
        return self.fallback.summarize(text, target_length=target_length).summary

    def target_length_bounds(self, target_length: int) -> tuple[int, int]:
        lower = clamp(int(target_length * 0.8), 24, 512)
        upper = clamp(int(target_length * 1.2), lower, 640)
        return lower, upper
