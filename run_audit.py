"""CLI entry: run an MLIP audit from a YAML config.

Examples
--------
python run_audit.py --config configs/llzo.yaml
python run_audit.py --config configs/llzo.yaml --model chgnet-0.3.0
python run_audit.py --config configs/llzo.yaml --device cuda
"""

from __future__ import annotations

import argparse

from mlip_audit.config import load_config
from mlip_audit.runner import run_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to audit YAML config")
    parser.add_argument(
        "--model",
        nargs="+",
        default=None,
        metavar="NAME",
        help=(
            "Run only these models (repeatable/space-separated)."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=("auto", "cpu", "cuda", "mps"),
        help=(
            "Override the device of every enabled model "
            "(default: auto, i.e. cuda when a GPU is usable, else cpu)"
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Debug: only use the first N structures",
    )
    args = parser.parse_args()
    run_audit(
        load_config(args.config),
        only_models=args.model,
        device_override=args.device,
        limit=args.limit or None,
    )


if __name__ == "__main__":
    main()
