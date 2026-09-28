#!/usr/bin/env python3
"""Mount Google Drive when running inside a Colab runtime."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mount-point", default="/content/drive")
    args = parser.parse_args()
    try:
        from google.colab import drive
    except ImportError as exc:
        raise SystemExit("This command must run inside Google Colab.") from exc
    drive.mount(args.mount_point, force_remount=False)
    print(f"Google Drive mounted at {args.mount_point}")


if __name__ == "__main__":
    main()

