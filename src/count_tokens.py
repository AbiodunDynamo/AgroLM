"""
count_tokens.py

Scaffold for tracking pretraining corpus size against the Chinchilla budget
(~20 tokens/param). Run this against data/raw/ as new sources are added.

This uses a whitespace-split proxy count, NOT a real tokenizer — that's
intentional at this stage. We don't have a domain tokenizer yet (that's a
later step, see Section 7 of project notes), and a rough word-count proxy is
good enough to answer "are we in the tens-of-millions-of-tokens range or not."
Real BPE token count will run ~1.2-1.5x the word count for English prose;
revisit this estimate once the actual tokenizer exists.

Usage:
    python src/count_tokens.py --data-dir data/raw
"""

import argparse
import re
from pathlib import Path


def clean_text(raw: str) -> str:
    """Minimal cleaning: strip common boilerplate patterns.
    Extend this as we see what real pulled text looks like (PDF headers/
    footers, page numbers, nav menus from scraped HTML, etc.)."""
    text = re.sub(r"\s+", " ", raw)  # collapse whitespace
    text = re.sub(r"Page \d+ of \d+", "", text)  # common PDF footer pattern
    return text.strip()


def word_count(text: str) -> int:
    return len(text.split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument(
        "--ext", type=str, default=".txt",
        help="File extension to scan (convert PDFs to .txt first)"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.rglob(f"*{args.ext}"))

    if not files:
        print(f"No {args.ext} files found under {data_dir}. "
              f"Nothing to count yet.")
        return

    total_words = 0
    print(f"{'File':<60} {'Words':>10}")
    print("-" * 72)
    for f in files:
        raw = f.read_text(encoding="utf-8", errors="ignore")
        cleaned = clean_text(raw)
        wc = word_count(cleaned)
        total_words += wc
        print(f"{str(f.relative_to(data_dir)):<60} {wc:>10,}")

    print("-" * 72)
    print(f"{'TOTAL (word-count proxy)':<60} {total_words:>10,}")
    est_tokens_low = int(total_words * 1.2)
    est_tokens_high = int(total_words * 1.5)
    print(f"\nEstimated BPE token range: {est_tokens_low:,} - {est_tokens_high:,}")
    print("(Rough proxy only — replace with real tokenizer count once trained.)")

    print("\n--- Chinchilla check (20 tokens/param) ---")
    for params, label in [(1_000_000, "1M"), (3_000_000, "3M"),
                          (5_000_000, "5M"), (8_000_000, "8M")]:
        needed = params * 20
        status = "OK" if est_tokens_low >= needed else "UNDER"
        print(f"{label:>4} params needs ~{needed:,} tokens -> {status}")


if __name__ == "__main__":
    main()
