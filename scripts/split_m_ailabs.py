"""Create reproducible book-level train/dev/test manifests for M-AILABS Russian.

The corpus has two narrators represented by only one book each. Those books stay
in train so training retains male and female voices; dev/test are held-out books
from the multi-book narrator. This avoids clip-level leakage while keeping the
first experiment practical.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path


def book_key(relative_path: str) -> tuple[str, str, str]:
    parts = Path(relative_path).parts
    try:
        marker = parts.index("by_book")
        return parts[marker + 1], parts[marker + 2], parts[marker + 3]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Unexpected M-AILABS path: {relative_path}") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=Path("data/indexes/m_ailabs_ru.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with args.index.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    books: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        books[book_key(row["path"])].append(row)

    by_speaker: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
    for key in books:
        by_speaker[key[:2]].append(key)

    # Keep speakers with a single book in train. From a multi-book speaker, put
    # one complete book into dev and one into test.
    candidates = [key for speaker_books in by_speaker.values() if len(speaker_books) > 1 for key in speaker_books]
    if len(candidates) < 2:
        raise SystemExit("Need at least two books from a repeated speaker for dev/test.")
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    assignments = {key: "train" for key in books}
    assignments[candidates[0]] = "dev"
    assignments[candidates[1]] = "test"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    writers: dict[str, csv.DictWriter] = {}
    handles = {}
    fields = ["dataset", "path", "bytes", "gender", "speaker", "book", "split"]
    try:
        for split in ("train", "dev", "test"):
            handle = (args.output_dir / f"m_ailabs_ru_{split}.csv").open("w", encoding="utf-8", newline="")
            handles[split] = handle
            writers[split] = csv.DictWriter(handle, fieldnames=fields)
            writers[split].writeheader()
        for key, book_rows in sorted(books.items()):
            split = assignments[key]
            gender, speaker, book = key
            for row in book_rows:
                writers[split].writerow({**row, "gender": gender, "speaker": speaker, "book": book, "split": split})
    finally:
        for handle in handles.values():
            handle.close()

    counts = Counter(assignments.values())
    print(f"Books: {len(books)} | train={counts['train']} dev={counts['dev']} test={counts['test']}")
    for key, split in sorted(assignments.items()):
        print(f"{split:5} {key[0]}/{key[1]}/{key[2]} ({len(books[key])} clips)")


if __name__ == "__main__":
    main()
