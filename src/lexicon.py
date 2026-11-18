"""
Lexicon parsing and matching for the Bias-in-Bios thesis pipeline.

Three lexicons, used by src/features.py:
  1. Big Two dictionary (Pietraszkiewicz, Formanowicz, Gustafsson Senden,
     Boyd, Sikstrom & Sczesny, 2019) —
     agency/communion, LIWC format. Two versions:
       version="A"  — data/lexicons/official/a_AgencyCommunion.dic (194/186),
                       the published main dictionary (Table A2 of the paper).
                       Main analysis from 2026-10-10.
       version="AB" — data/lexicons/agency_communion.dic (227/218),
                       union of A and B; used until 2026-10-10, now robustness variant.
     No silent default: a call without version raises ValueError.
  2. Achievement lexicon — two versions:
       version="A"  — 23 terms: exact set above + wild without excellent*/outstanding*
                       (those two entries only exist in B).
       version="AB" — 25 terms: full set including excellent*/outstanding*.
  3. Recognition lexicon — researcher-constructed, unchanged across versions.

Matching semantics (LIWC wildcard, no lemmatization): exact entries match the
lowercased token surface form; entries ending in '*' match by prefix.

Usage:
    from src.lexicon import load_lexicons
    lex = load_lexicons(version="A")   # main analysis
    lex = load_lexicons(version="AB")  # robustness variant
    lex.match_agency('accomplished')   # True
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

_DIC_PATHS = {
    "A":  Path("data/lexicons/official/a_AgencyCommunion.dic"),
    "AB": Path("data/lexicons/agency_communion.dic"),
}

_ACH_EXACT = {
    'overcome', 'overcomes', 'mastered', 'mastering', 'mastery',
    'win', 'wins', 'won', 'earn', 'earned', 'earning', 'earns',
}
# AB: 25 entries (includes excellent* and outstanding* from lexicon B)
_ACH_WILD_AB = sorted([
    'accomplish', 'achiev', 'attain', 'succeed', 'success', 'triumph',
    'excellent', 'outstanding', 'prevail', 'masterful', 'victor',
    'winner', 'winning',
])
# A: 23 entries — excellent* and outstanding* removed (B-only entries)
_ACH_WILD_A = sorted([w for w in _ACH_WILD_AB if w not in ('excellent', 'outstanding')])

_RECOG_EXACT = {'elected', 'selected', 'named'}
_RECOG_WILD = sorted(['award', 'recogni', 'appoint', 'honor', 'honour', 'feature', 'nominat'])


def parse_dic(path: Path):
    """
    Parse a LIWC-format .dic file into exact/wildcard term sets per category
    (1=agency, 2=communion).

    Returns: agency_exact, agency_wild (sorted stems),
             communion_exact, communion_wild (sorted stems),
             multi_word (list of (term, category, is_wildcard) for entries with spaces)
    """
    agency_exact, communion_exact = set(), set()
    agency_wild, communion_wild = [], []
    multi_word = []

    in_data = False
    pct_seen = 0
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line == '%':
            pct_seen += 1
            if pct_seen == 2:
                in_data = True
            continue
        if not in_data or not line or '\t' not in line:
            continue
        term, cat = line.rsplit('\t', 1)
        term = term.lower().strip()
        cat = cat.strip()

        if ' ' in term:
            is_wild = term.endswith('*')
            multi_word.append((term.rstrip('*'), cat, is_wild))
            continue

        is_wild = term.endswith('*')
        if cat == '1':
            agency_wild.append(term[:-1]) if is_wild else agency_exact.add(term)
        elif cat == '2':
            communion_wild.append(term[:-1]) if is_wild else communion_exact.add(term)

    return (agency_exact, sorted(agency_wild),
            communion_exact, sorted(communion_wild),
            multi_word)


def make_matcher(exact_set: set, wild_sorted: list) -> Callable[[str], bool]:
    """
    Returns match(tok_lower) -> bool.

    Wildcard lookup uses a single compiled alternation regex anchored at the
    start of the token. A binary-search-on-sorted-stems approach is NOT correct
    here: when one stem is itself a prefix of another (e.g. 'affect' / 'affection',
    'open' / 'open-mind'), the nearest sorted predecessor of a query like 'affects'
    can be the more specific stem ('affection'), whose startswith check then fails
    -- silently dropping a real match against the shorter stem ('affect').
    """
    if wild_sorted:
        pattern = re.compile('^(?:' + '|'.join(re.escape(w) for w in wild_sorted) + ')')
    else:
        pattern = None

    def match(tok_lower: str) -> bool:
        if tok_lower in exact_set:
            return True
        return pattern is not None and pattern.match(tok_lower) is not None
    return match


def count_multiword_communion(text_lower: str, multi_word_communion: list) -> int:
    """Count multi-word communion matches (e.g. 'common ground') in lowercased text."""
    count = 0
    for term, is_wild in multi_word_communion:
        pat = r'\b' + re.escape(term) + (r'\w*' if is_wild else r'') + r'\b'
        count += len(re.findall(pat, text_lower))
    return count


@dataclass
class Lexicons:
    match_agency: Callable[[str], bool]
    match_communion: Callable[[str], bool]
    match_achievement: Callable[[str], bool]
    match_recognition: Callable[[str], bool]
    multi_word_communion: list


def load_lexicons(version: str) -> Lexicons:
    """Parse the Big Two dictionary and build all matchers.

    version : "A"  — main analysis (Pietraszkiewicz et al. 2019, Table A2; 194/186)
              "AB" — robustness variant (A ∪ B; 227/218; used before 2026-10-10)

    Raises ValueError for any other value; no silent default.
    Call once per notebook.
    """
    if version not in _DIC_PATHS:
        raise ValueError(
            f"load_lexicons: version must be 'A' or 'AB', got {version!r}. "
            "Specify version='A' for the main analysis (Pietraszkiewicz 2019, Table A2) "
            "or version='AB' for the A∪B robustness variant."
        )
    dic_path = _DIC_PATHS[version]
    ach_wild = _ACH_WILD_A if version == "A" else _ACH_WILD_AB
    agency_exact, agency_wild, communion_exact, communion_wild, multi_word = parse_dic(dic_path)
    multi_word_communion = [(t, w) for t, c, w in multi_word if c == '2']
    return Lexicons(
        match_agency=make_matcher(agency_exact, agency_wild),
        match_communion=make_matcher(communion_exact, communion_wild),
        match_achievement=make_matcher(_ACH_EXACT, ach_wild),
        match_recognition=make_matcher(_RECOG_EXACT, _RECOG_WILD),
        multi_word_communion=multi_word_communion,
    )
