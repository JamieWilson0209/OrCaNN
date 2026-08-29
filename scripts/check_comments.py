#!/usr/bin/env python3
"""Flag comments that describe the repository's history instead of its code.

A comment is read by someone who has never seen the commit that prompted it.
"this used to fall back to 2.0" tells them about a version they cannot look at;
"the config value is the only source" tells them what the line in front of them
does. The first ages into a lie the moment the surrounding code moves again,
and it is the class of comment the 2026 sweep removed across the pipeline.

What this catches is only the mechanically detectable part: phrases that point
at a change, commit hashes, and version stamps. A comment that is merely stale
-- accurate prose about behaviour that no longer exists -- cannot be found by
grep and still needs a reader.

Usage
-----
    python scripts/check_comments.py                 # checks src/orcann
    python scripts/check_comments.py path [path ...] # checks what you name

Exits 1 if anything is flagged, so it works as a pre-commit or CI step.

check-comments: skip-file

Exemptions
----------
Sometimes the history *is* the message: a config value that was removed needs to
tell whoever still names it that it went and what to use instead. Put
``history-ok`` in the comment to say so deliberately::

    # history-ok: the message a user sees when their config names a dead value
"""
from __future__ import annotations

import ast
import io
import re
import sys
import tokenize
from pathlib import Path
from typing import Iterator, List, Tuple

# Each pattern names the thing it looks for, so the report says what is wrong
# rather than echoing a regex at the reader.
PATTERNS: List[Tuple[str, re.Pattern]] = [
    # "used to" and "no longer" are only about history when a subject points at
    # the code itself. "the window used to measure baseline" and "evenly spaced
    # days are no longer a timeline" are ordinary English about the subject
    # matter, and flagging them would teach everyone to ignore this check.
    ("refers to former behaviour", re.compile(
        r"\b(this|it|we|they|these|those|which)\s+(?:used to|no longer)\b", re.I)),
    ("refers to a change", re.compile(
        r"\b(in this branch|before this change|now that|"
        r"has been (?:changed|replaced|removed)|"
        r"removed in (?:this branch|[0-9a-f]{7,40}))\b", re.I)),
    # A preposition in front is what separates a commit reference from a hex
    # colour or an id that happens to be hexadecimal.
    ("names a commit", re.compile(
        r"\b(?:since|in|commit|see)\s+[0-9a-f]{7,40}\b", re.I)),
    # Parenthesised with a dot: "(v2.0)" is a stamp on our own code, while
    # "v7.3 MATLAB files" and "loadmat (v7)" name someone else's format.
    ("stamps a version", re.compile(r"\(\s*v\d+\.\d+[^)]*\)", re.I)),
]

EXEMPT = re.compile(r"history-ok", re.I)

# A file that quotes these patterns in order to explain them — this one — says
# so once at the top rather than marking every line it quotes.
SKIP_FILE = re.compile(r"check-comments:\s*skip-file", re.I)

# A bare hex run is only a commit reference if it is not obviously something
# else -- a colour, a hash-like id in a URL, a hex constant.
NOT_A_COMMIT = re.compile(r"(#[0-9a-fA-F]{3,8}\b)|0x[0-9a-fA-F]+|https?://")


def _comments(path: Path) -> Iterator[Tuple[int, str]]:
    """Every comment in a file, as (line number, text)."""
    with open(path, "rb") as fh:
        try:
            for tok in tokenize.tokenize(fh.readline):
                if tok.type == tokenize.COMMENT:
                    yield tok.start[0], tok.string
        except (tokenize.TokenError, SyntaxError, IndentationError):
            return


def _docstrings(path: Path) -> Iterator[Tuple[int, str]]:
    """Every module, class and function docstring, as (line number, text)."""
    try:
        tree = ast.parse(io.open(path, encoding="utf-8").read())
    except (SyntaxError, UnicodeDecodeError):
        return
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef,
                             ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if not doc:
                continue
            first = getattr(node, "body", [None])[0]
            start = getattr(first, "lineno", 1)
            # Report the offending line, not the top of the docstring.
            for offset, line in enumerate(doc.split("\n")):
                yield start + offset, line


def check(path: Path) -> List[Tuple[int, str, str]]:
    """Findings for one file, as (line, what is wrong, the text)."""
    found = []
    try:
        if SKIP_FILE.search(io.open(path, encoding="utf-8").read(4096)):
            return found
    except (OSError, UnicodeDecodeError):
        return found
    for lineno, text in list(_comments(path)) + list(_docstrings(path)):
        if EXEMPT.search(text):
            continue
        for label, pattern in PATTERNS:
            hit = pattern.search(text)
            if not hit:
                continue
            if label == "names a commit" and NOT_A_COMMIT.search(text):
                continue
            found.append((lineno, label, text.strip()))
            break
    return found


def main(argv: List[str]) -> int:
    roots = [Path(a) for a in argv[1:]] or [Path("src/orcann")]
    files = sorted(
        f for root in roots
        for f in ([root] if root.is_file() else root.rglob("*.py"))
        if "__pycache__" not in f.parts)

    total = 0
    for f in files:
        for lineno, label, text in check(f):
            total += 1
            print(f"{f}:{lineno}: {label}")
            print(f"    {text[:110]}")

    if total:
        print(f"\n{total} comment(s) describe the repository rather than the "
              f"code. Say what the line does now; if the history is genuinely "
              f"the message, mark it 'history-ok'.")
        return 1
    print(f"{len(files)} file(s) checked, no comments describing history.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
