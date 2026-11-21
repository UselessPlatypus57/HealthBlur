"""PHI detection in free text."""

from __future__ import annotations

from .clinical import is_clinical_text
from .known import KnownIdentifiers, harvest
from .patterns import Finding, find_name_like, find_patterns


def detect(text: str, known: KnownIdentifiers | None = None, *, include_weak: bool = True) -> list[Finding]:
    """All PHI findings for ``text``: known-identifier matches plus generic patterns."""
    findings = list(known.find(text)) if known else []
    findings += find_patterns(text)
    if include_weak:
        findings += find_name_like(text)
    else:
        findings = [f for f in findings if f.strength == "strong"]
    return findings


def mask(text: str) -> str:
    """Keep the first character of each alphanumeric run; star the rest."""
    out, prev_alnum = [], False
    for ch in text:
        if ch.isalnum():
            out.append(ch if not prev_alnum else "*")
            prev_alnum = True
        else:
            out.append(ch)
            prev_alnum = False
    return "".join(out)


__all__ = ["Finding", "KnownIdentifiers", "detect", "find_patterns", "harvest", "is_clinical_text", "mask"]
