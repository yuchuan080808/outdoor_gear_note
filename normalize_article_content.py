#!/usr/bin/env python3
"""
Normalize generated Markdown articles after the main content pipeline runs.

This keeps visible review-summary copy reader-friendly and removes older
ASCII score bars such as [####-] that render poorly in Hugo tables.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from main import (
    GeneratedArticleEnhancer,
    OUTPUT_DIR,
    PRODUCT_CACHE,
    SEOArticleOptimizer,
    SEOResourceLinker,
    article_categories,
)


SCORE_BAR_RE = re.compile(r"(\|\s*(Pros signal|Evidence depth|Complaint pressure)\s*\|\s*)\[([#-]{5})\]\s*")
COMPLAINT_NA_RE = re.compile(
    r"(\|\s*Complaint pressure\s*\|\s*(?:[^|]*?-\s*)?)(?:N/A|n/a)\s*(\|)",
    re.IGNORECASE,
)
FRONTMATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.DOTALL)
CATEGORIES_BLOCK_RE = re.compile(r"(?m)^categories:\r?\n(?:[ \t]+-\s[^\r\n]*(?:\r?\n|$))+")


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


def _frontmatter_scalar(frontmatter: str, key: str) -> str:
    match = re.search(rf"^{re.escape(key)}:\s*(.*?)\s*$", frontmatter, re.MULTILINE)
    if not match:
        return ""
    raw = match.group(1).strip()
    if not raw:
        return ""
    try:
        parsed = json.loads(raw)
        return str(parsed)
    except json.JSONDecodeError:
        return raw.strip("\"'")


def normalize_article_categories(markdown: str) -> str:
    original_markdown = markdown
    if markdown.startswith("\ufeff"):
        markdown = markdown[1:]

    match = FRONTMATTER_RE.match(markdown)
    if not match:
        return original_markdown

    frontmatter = match.group(1)
    section = _frontmatter_scalar(frontmatter, "section")
    category_path = _frontmatter_scalar(frontmatter, "category_path")
    title = _frontmatter_scalar(frontmatter, "title")
    if not section or not category_path:
        return markdown

    category_name = category_path.split(">")[-1].strip() if category_path else title.removeprefix("Best ").strip()
    categories = article_categories(section, category_path, category_name)
    categories_block = "categories:\n" + "\n".join(f"  - {json.dumps(category)}" for category in categories) + "\n"

    if CATEGORIES_BLOCK_RE.search(frontmatter):
        updated_frontmatter = CATEGORIES_BLOCK_RE.sub(categories_block, frontmatter, count=1)
    elif re.search(r"(?m)^tags:\s*$", frontmatter):
        updated_frontmatter = re.sub(r"(?m)^tags:\s*$", f"{categories_block}tags:", frontmatter, count=1)
    else:
        updated_frontmatter = frontmatter.rstrip() + "\n" + categories_block.rstrip()

    if updated_frontmatter == frontmatter:
        return markdown if markdown != original_markdown else original_markdown
    return markdown[: match.start(1)] + updated_frontmatter + markdown[match.end(1) :]


def normalize_existing_categories(content_dir: Path = OUTPUT_DIR) -> int:
    changed_count = 0
    for path in sorted(content_dir.rglob("*.md")):
        if path.name == "_index.md":
            continue
        original = path.read_text(encoding="utf-8")
        updated = normalize_article_categories(original)
        if updated != original:
            path.write_text(updated, encoding="utf-8")
            changed_count += 1
    return changed_count


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
    seo_refreshed = SEOArticleOptimizer.refresh_existing_content(OUTPUT_DIR)
    link_refreshed = SEOResourceLinker.refresh_existing_content(OUTPUT_DIR)
    category_fixed = normalize_existing_categories(OUTPUT_DIR)
    fallback_fixed = fallback_normalize_content(OUTPUT_DIR)
    print(
        "Normalized generated article content: "
        f"feedback={refreshed}, seo={seo_refreshed}, links={link_refreshed}, "
        f"categories={category_fixed}, fallback_fixed={fallback_fixed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
