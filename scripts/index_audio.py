"""Create a tiny local inventory of downloaded source audio.

This is deliberately not a dataset contract. It just gives the team a visible,
reproducible list of files while we decide the final training format.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

SUPPORTED = {".wav", ".flac", ".mp3", ".opus", ".m4a"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, default=Path("data/indexes"))
    args = parser.parse_args()

    if not args.source.is_dir():
        raise SystemExit(f"Source directory does not exist: {args.source}")

    files = sorted(path for path in args.source.rglob("*") if path.suffix.lower() in SUPPORTED)
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / f"{args.dataset}.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "path", "bytes"])
        writer.writeheader()
        for path in files:
            writer.writerow(
                {"dataset": args.dataset, "path": path.relative_to(args.source), "bytes": path.stat().st_size}
            )
    print(f"Indexed {len(files)} audio files in {output}")


if __name__ == "__main__":
    main()
