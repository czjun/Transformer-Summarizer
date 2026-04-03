from __future__ import annotations

from dataclasses import asdict

from .config import SummarizationConfig
from .engine import HybridSummarizer


try:
    from fastapi import FastAPI
    from pydantic import BaseModel, Field
except Exception:  # pragma: no cover
    FastAPI = None
    BaseModel = object
    Field = None


if FastAPI is not None:
    app = FastAPI(title="Transformer Summarizer", version="0.1.0")
    engine = HybridSummarizer(SummarizationConfig())

    class SummarizeRequest(BaseModel):
        text: str
        target_length: int | None = Field(default=120, description="Target length in Chinese characters.")

    class SummarizeResponse(BaseModel):
        summary: str
        backend: str
        target_length: int | None

    @app.get("/health")
    def health():
        return {"status": "ok", "backend": engine.backend_name}

    @app.post("/summarize", response_model=SummarizeResponse)
    def summarize(req: SummarizeRequest):
        result = engine.summarize(req.text, target_length=req.target_length)
        return SummarizeResponse(
            summary=result.summary,
            backend=result.backend,
            target_length=result.used_target_length,
        )
else:  # pragma: no cover
    app = None
