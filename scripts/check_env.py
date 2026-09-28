#!/usr/bin/env python3
"""Fail early with actionable diagnostics before downloading a model."""

import importlib
import sys


REQUIRED = ["torch", "transformers", "datasets", "peft", "accelerate", "yaml"]


def main() -> int:
    missing = []
    for name in REQUIRED:
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    if missing:
        print("Missing packages: " + ", ".join(missing), file=sys.stderr)
        print("Run: pip install -r requirements-colab.txt", file=sys.stderr)
        return 1
    import torch

    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

