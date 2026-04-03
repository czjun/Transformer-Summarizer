from __future__ import annotations

import argparse
from pathlib import Path

from .config import SummarizationConfig
from .engine import HybridSummarizer
from .evaluate import evaluate_pairs
from .train import train_model
from .data import load_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transformer summarization system")
    sub = parser.add_subparsers(dest="command", required=True)

    summarize = sub.add_parser("summarize", help="Generate one summary")
    summarize.add_argument("--text", required=True)
    summarize.add_argument("--target-length", type=int, default=120)
    summarize.add_argument("--model-name", default=SummarizationConfig().model_name)

    train = sub.add_parser("train", help="Train a seq2seq model")
    train.add_argument("--train-path", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--valid-path", default=None)
    train.add_argument("--model-name", default=SummarizationConfig().model_name)

    evaluate = sub.add_parser("evaluate", help="Evaluate predictions on a JSONL file")
    evaluate.add_argument("--path", required=True)
    evaluate.add_argument("--model-name", default=SummarizationConfig().model_name)
    evaluate.add_argument("--target-length", type=int, default=120)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "summarize":
        engine = HybridSummarizer(SummarizationConfig(model_name=args.model_name))
        result = engine.summarize(args.text, target_length=args.target_length)
        print(result.summary)
        print(f"[backend={result.backend}]")
        return 0

    if args.command == "train":
        cfg = SummarizationConfig(model_name=args.model_name)
        train_model(args.train_path, args.output_dir, config=cfg, valid_path=args.valid_path)
        print("training finished")
        return 0

    if args.command == "evaluate":
        examples = load_jsonl(args.path)
        engine = HybridSummarizer(SummarizationConfig(model_name=args.model_name))
        pairs = []
        targets = []
        for ex in examples:
            pred = engine.summarize(ex.article, target_length=args.target_length).summary
            pairs.append((ex.summary, pred))
            targets.append(args.target_length)
        report = evaluate_pairs(pairs, target_lengths=targets)
        for key, value in report.items():
            print(f"{key}: {value:.4f}")
        return 0

    return 1

