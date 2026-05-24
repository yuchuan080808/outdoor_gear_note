#!/usr/bin/env python3
"""
Normalize generated Markdown articles after the main content pipeline runs.

This keeps visible review-summary copy reader-friendly and removes older
ASCII score bars such as [####-] that render poorly in Hugo tables.
"""

from __future__ import annotations

import re
from pathlib import Path

from main import GeneratedArticleEnhancer, OUTPUT_DIR, PRODUCT_CACHE


SCORE_BAR_RE = re.compile(r"(\|\s*(Pros signal|Evidence depth|Complaint pressure)\s*\|\s*)\[([#-]{5})\]\s*")
COMPLAINT_NA_RE = re.compile(
    r"(\|\s*Complaint pressure\s*\|\s*(?:[^|]*?-\s*)?)(?:N/A|n/a)\s*(\|)",
    re.IGNORECASE,
)


def score_label(signal: str, score_bar: str) -> str:
    score = max(1, min(5, score_bar.count("#")))
    labels = {
        "Pros signal": {
            5: "Excellent buyer signal",
            4: "Strong buyer signal",
            3: "Moderate buyer signal",
            2: "Mixed buyer signal",
            1: "Weak buyer signal",
        },
        "Evidence depth": {
            5: "Very strong evidence",
            4: "Strong evidence",
            3: "Moderate evidence",
            2: "Thin evidence",
            1: "Limited evidence",
        },
        "Complaint pressure": {
            5: "High complaint pressure",
            4: "Elevated complaint pressure",
            3: "Moderate complaint pressure",
            2: "Low complaint pressure",
            1: "Very low complaint pressure",
        },
    }
    return labels.get(signal, labels["Evidence depth"])[score]


def normalize_legacy_score_bars(markdown: str) -> str:
    def replace_score(match: re.Match[str]) -> str:
        prefix, signal, score_bar = match.groups()
        return f"{prefix}{score_label(signal, score_bar)} - "

    markdown = SCORE_BAR_RE.sub(replace_score, markdown)
    markdown = COMPLAINT_NA_RE.sub(
        r"\1No clear recurring complaint theme surfaced in the customer-summary data. \2",
        markdown,
    )
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    return markdown


def fallback_normalize_content(content_dir: Path = OUTPUT_DIR) -> int:
    changed_count = 0
    for path in sorted(content_dir.rglob("*.md")):
        if path.name == "_index.md":
            continue
        original = path.read_text(encoding="utf-8")
        updated = normalize_legacy_score_bars(original)
        if updated != original:
            path.write_text(updated, encoding="utf-8")
            changed_count += 1
    return changed_count


def main() -> int:
    refreshed = GeneratedArticleEnhancer.refresh_existing_content(OUTPUT_DIR, PRODUCT_CACHE)
    fallback_fixed = fallback_normalize_content(OUTPUT_DIR)
    print(f"Normalized generated article content: refreshed={refreshed}, fallback_fixed={fallback_fixed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
