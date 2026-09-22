#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence


DEFAULT_FILES = ("PROJECT_STATUS.md", "docs/STE_STYLE.md")
MAX_WORDS = 25
DISALLOWED = {
    "and/or": "Use one clear relation.",
    "etc.": "Give the complete list or a named category.",
    "obviously": "Remove subjective emphasis.",
    "simply": "Remove subjective emphasis.",
    "very": "Use a measurable value.",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Check mechanical ASD-STE100-style rules in public status documents")
    parser.add_argument("files", nargs="*", default=list(DEFAULT_FILES))
    args = parser.parse_args()

    failures = check_files([Path(value) for value in args.files])
    if failures:
        for failure in failures:
            print(failure)
        return 2
    print(f"STE-style check passed for {len(args.files)} files.")
    return 0


def check_files(paths: Sequence[Path]) -> list[str]:
    failures: list[str] = []
    for path in paths:
        if not path.exists():
            failures.append(f"{path}: file does not exist")
            continue
        failures.extend(check_text(path, path.read_text(encoding="utf-8")))
    return failures


def check_text(path: Path, text: str) -> list[str]:
    failures = []
    in_code = False
    for number, source_line in enumerate(text.splitlines(), start=1):
        line = source_line.strip()
        if line.startswith("```"):
            in_code = not in_code
            continue
        if should_skip(line, in_code):
            continue
        prose = normalize_markdown(line)
        words = re.findall(r"[A-Za-z0-9]+(?:[._/-][A-Za-z0-9]+)*", prose)
        if len(words) > MAX_WORDS:
            failures.append(f"{path}:{number}: {len(words)} words; maximum is {MAX_WORDS}")
        lower = prose.lower()
        for phrase, guidance in DISALLOWED.items():
            if re.search(rf"(?<![A-Za-z]){re.escape(phrase)}(?![A-Za-z])", lower):
                failures.append(f"{path}:{number}: disallowed phrase '{phrase}'. {guidance}")
    return failures


def should_skip(line: str, in_code: bool) -> bool:
    return (
        in_code
        or not line
        or line.startswith("#")
        or line.startswith("|")
        or line.startswith("<!--")
        or line.startswith("[") and "](" in line
    )


def normalize_markdown(line: str) -> str:
    line = re.sub(r"^[-*+]\s+", "", line)
    line = re.sub(r"^\d+\.\s+", "", line)
    line = re.sub(r"\[([^]]+)]\([^)]+\)", r"\1", line)
    line = re.sub(r"[`*_>]", "", line)
    return line


if __name__ == "__main__":
    raise SystemExit(main())
