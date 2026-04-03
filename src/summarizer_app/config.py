from dataclasses import dataclass


@dataclass
class SummarizationConfig:
    model_name: str = "google/mt5-small"
    max_source_length: int = 1024
    min_target_length: int = 30
    max_target_length: int = 160
    num_beams: int = 4
    no_repeat_ngram_size: int = 3
    length_penalty: float = 1.0
    fallback_sentences: int = 3
    device: str | None = None
