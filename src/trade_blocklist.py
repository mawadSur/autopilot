"""Do-not-trade blocklist: market topics we never want a position in.

Operator policy filter (alcohol, casino/gambling, adult content by default, plus
any operator-supplied terms). A market whose title/outcome matches a blocked
term is excluded from shadow candidates and never logged — and the exclusion is
reported (not silent), so it is always visible what was filtered out.

Matching is case-insensitive and WHOLE-WORD (regex word boundaries) to avoid
false positives like "winner" matching "wine" or "winnersexpo" matching a
substring. Multi-word terms (e.g. "adult film") are matched as a phrase.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Sequence

__all__ = [
    "DEFAULT_BLOCKED_TERMS",
    "load_blocklist",
    "blocked_term",
]

# Curated, conservative defaults. Kept deliberately specific to limit false
# positives; operators tune via --blocklist-file. Lowercased here.
DEFAULT_BLOCKED_TERMS = (
    # --- Alcohol ---
    "alcohol", "alcoholic", "beer", "wine", "liquor", "whiskey", "whisky",
    "vodka", "tequila", "bourbon", "brewery", "brewing", "champagne",
    # --- Casino / gambling ---
    "casino", "gambling", "gamble", "blackjack", "roulette", "baccarat",
    "slot machine", "sportsbook",
    # --- Adult content ---
    "porn", "pornography", "nsfw", "onlyfans", "xxx", "brothel",
    "adult film", "adult content", "adult video",
)


def load_blocklist(
    path: Optional[str] = None,
    *,
    extra_terms: Optional[Sequence[str]] = None,
    include_defaults: bool = True,
) -> List[str]:
    """Build the blocklist: defaults + an optional file (one term per line,
    ``#`` comments allowed) + ``extra_terms``. Terms are lowercased and
    de-duplicated, order preserved.
    """
    terms: List[str] = list(DEFAULT_BLOCKED_TERMS) if include_defaults else []
    if path:
        fp = Path(path)
        if fp.is_file():
            for raw in fp.read_text().splitlines():
                line = raw.strip()
                if line and not line.startswith("#"):
                    terms.append(line)
    if extra_terms:
        terms.extend(extra_terms)
    out: List[str] = []
    seen = set()
    for term in terms:
        norm = " ".join(str(term).strip().lower().split())
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def blocked_term(text: Optional[str], terms: Sequence[str]) -> Optional[str]:
    """Return the FIRST blocked term that appears as a whole word/phrase in
    ``text`` (case-insensitive), or ``None`` if nothing matches.

    Whole-word matching avoids false positives: "wine" does not match "winner",
    but does match "red wine market".
    """
    if not text or not terms:
        return None
    haystack = str(text)
    for term in terms:
        if not term:
            continue
        if re.search(r"\b" + re.escape(term) + r"\b", haystack, re.IGNORECASE):
            return term
    return None
