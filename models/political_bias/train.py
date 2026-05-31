"""
Unified training entry point for political-bias models.

Delegates to the baseline or improved training script.

Usage:
    # Baseline training (standard AdamW, no auxiliary pre-training)
    python -m models.political_bias.train --dataset 10_BABE

    # Improved training (layerwise LR decay + auxiliary pre-training)
    python -m models.political_bias.train --mode improved --dataset 10_BABE

    # Pass additional flags through to the underlying script
    python -m models.political_bias.train --mode improved --dataset 10_BABE --lr-decay 0.8
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Political-bias training (baseline or improved)",
        add_help=False,
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "improved"],
        default="baseline",
        help="Training mode: 'baseline' (default) or 'improved' (layerwise decay + aux pre-training)",
    )
    args, remaining = parser.parse_known_args()

    # Swap sys.argv so the delegated main() sees only the remaining flags
    sys.argv = [sys.argv[0]] + remaining

    if args.mode == "improved":
        from models.political_bias.train_improved import main as _main
    else:
        from models.political_bias.train_baseline import main as _main

    _main()


if __name__ == "__main__":
    main()
