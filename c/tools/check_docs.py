#!/usr/bin/env python3
"""Validate local Markdown links and the research-report index."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any


MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")


def _markdown_files(root: Path) -> list[Path]:
    files = [root / "PLAN.md", root / "README.md"]
    files.extend(sorted((root / "docs").rglob("*.md")))
    return [path for path in files if path.is_file()]


def check_docs(root: Path) -> dict[str, Any]:
    root = root.resolve()
    documents = _markdown_files(root)
    failures: list[str] = []
    local_links = 0
    for document in documents:
        text = document.read_text(encoding="utf-8")
        for match in MARKDOWN_LINK.finditer(text):
            raw = match.group(1)
            if (
                "://" in raw
                or raw.startswith("#")
                or raw.startswith("mailto:")
            ):
                continue
            target = raw.split("#", 1)[0]
            if not target:
                continue
            local_links += 1
            resolved = (document.parent / target).resolve()
            try:
                resolved.relative_to(root)
            except ValueError:
                failures.append(
                    f"{document.relative_to(root)}: link escapes root: {raw}"
                )
                continue
            if not resolved.is_file():
                failures.append(
                    f"{document.relative_to(root)}: missing link target: {raw}"
                )

    research = root / "docs" / "research"
    index = research / "README.md"
    reports = (
        sorted(path for path in research.glob("*.md") if path.name != "README.md")
        if research.is_dir()
        else []
    )
    if not index.is_file():
        failures.append("docs/research/README.md: research index is missing")
        index_text = ""
    else:
        index_text = index.read_text(encoding="utf-8")
    unindexed = [
        path.name
        for path in reports
        if f"({path.name})" not in index_text
    ]
    failures.extend(f"research report is not indexed: {name}" for name in unindexed)

    return {
        "passed": not failures,
        "documents": len(documents),
        "local_links": local_links,
        "research_reports": len(reports),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    args = parser.parse_args()
    result = check_docs(args.root)
    print(
        "docs: "
        f"documents={result['documents']} "
        f"local_links={result['local_links']} "
        f"research_reports={result['research_reports']}"
    )
    for failure in result["failures"]:
        print(f"FAIL: {failure}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
