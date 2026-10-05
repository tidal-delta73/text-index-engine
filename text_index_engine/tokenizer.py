"""Shared text normalization and tokenization for build and search."""
import unicodedata


def normalize_text(text: str) -> str:
    """Apply Unicode NFKC normalization followed by casefolding."""
    return unicodedata.normalize("NFKC", text).casefold()


def tokenize(text: str) -> list[str]:
    """Split text into terms: maximal runs of Unicode letters or digits.

    Returns the normalized terms in order of appearance.
    """
    terms: list[str] = []
    current: list[str] = []
    for ch in normalize_text(text):
        if ch.isalnum():
            current.append(ch)
        elif current:
            terms.append("".join(current))
            current = []
    if current:
        terms.append("".join(current))
    return terms
