"""
Convenience CLI entry point for the ML package.

Examples:
    python main.py train --run-name price_only
    python main.py train --run-name price_plus_sentiment --use-sentiment
    python main.py infer --model-dir models/price_prediction/price_only
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Factor Price Prediction ML CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train", add_help=False)
    subparsers.add_parser("infer", add_help=False)

    args, remaining = parser.parse_known_args()

    if args.command == "train":
        from src.pipeline.training_pipeline import main as train_main

        sys.argv = [sys.argv[0]] + remaining
        train_main()
    elif args.command == "infer":
        from src.pipeline.inference_pipeline import main as infer_main

        sys.argv = [sys.argv[0]] + remaining
        infer_main()


if __name__ == "__main__":
    main()
