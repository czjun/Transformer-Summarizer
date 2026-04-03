from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


try:
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover
    Dataset = object


@dataclass
class SummarizationExample:
    article: str
    summary: str


def load_jsonl(path: str | Path) -> List[SummarizationExample]:
    items: List[SummarizationExample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            article = obj.get("article") or obj.get("text") or ""
            summary = obj.get("summary") or obj.get("label") or ""
            if article and summary:
                items.append(SummarizationExample(article=article, summary=summary))
    return items


class JsonlSummarizationDataset(Dataset):
    def __init__(self, path: str | Path, tokenizer, max_source_length: int = 1024, max_target_length: int = 160):
        self.examples = load_jsonl(path)
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        model_inputs = self.tokenizer(
            ex.article,
            max_length=self.max_source_length,
            truncation=True,
        )
        labels = self.tokenizer(
            text_target=ex.summary,
            max_length=self.max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


def batch_texts(examples: Iterable[SummarizationExample]) -> List[str]:
    return [item.article for item in examples]
