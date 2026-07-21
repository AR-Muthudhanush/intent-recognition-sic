from __future__ import annotations

import argparse

from src.evaluate import evaluate
from src.train import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Bilingual TinyBERT intent recognition pipeline")
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "all"],
        required=True,
        help="train: train + quantize, evaluate: evaluate quantized model, all: train then evaluate",
    )
    args = parser.parse_args()

    if args.mode in {"train", "all"}:
        train()
    if args.mode in {"evaluate", "all"}:
        evaluate()


if __name__ == "__main__":
    main()
