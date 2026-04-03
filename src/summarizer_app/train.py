from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from .config import SummarizationConfig
from .data import JsonlSummarizationDataset


try:
    import torch
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )
except Exception:  # pragma: no cover - dependency fallback
    torch = None
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None
    DataCollatorForSeq2Seq = None
    Seq2SeqTrainer = None
    Seq2SeqTrainingArguments = None


def train_model(
    train_path: str | Path,
    output_dir: str | Path,
    config: SummarizationConfig | None = None,
    valid_path: str | Path | None = None,
) -> None:
    if AutoTokenizer is None or AutoModelForSeq2SeqLM is None or torch is None:
        raise RuntimeError("Training requires torch and transformers.")

    cfg = config or SummarizationConfig()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name)

    train_dataset = JsonlSummarizationDataset(train_path, tokenizer, cfg.max_source_length, cfg.max_target_length)
    eval_dataset = (
        JsonlSummarizationDataset(valid_path, tokenizer, cfg.max_source_length, cfg.max_target_length)
        if valid_path
        else None
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        learning_rate=3e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        predict_with_generate=True,
        num_train_epochs=3,
        logging_steps=25,
        save_steps=100,
        evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=100 if eval_dataset else None,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

