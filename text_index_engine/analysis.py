"""Shared text processing: Unicode NFKC normalization, casefold, tokenization.

A token is a maximal run of Unicode letters or decimal digits, recognized
*after* normalization (so NFKC-composed characters tokenize as one unit).
Empty/whitespace-only text yields zero tokens while the document is still
indexed (so NOT can cover it).
"""
import unicodedata


def normalize(text: str) -> str:
    """NFKC normalize, casefold. Shared by build and search."""
    return unicodedata.normalize("NFKC", text).casefold()


def tokenize(text: str) -> list[str]:
    """Return normalized tokens in zero-based document order.

    A token is a maximal contiguous run of codepoints whose Unicode general
    category starts with L (letter) or N (number: Nd/Nl/No). Fractions,
    roman numerals, superscript digits etc. are handled by NFKC beforehand.
    """
    tokens: list[str] = []
    run: list[str] = []

    def _is_token_char(ch: str) -> bool:
        # L* = letter (Lu Ll Lt Lm Lo), N* = number (Nd Nl No).
        return unicodedata.category(ch)[0] in ("L", "N")

    for ch in normalize(text):
        if _is_token_char(ch):
            run.append(ch)
        else:
            if run:
                tokens.append("".join(run))
                run = []
    if run:
        tokens.append("".join(run))
    return tokens
