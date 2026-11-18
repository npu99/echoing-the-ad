"""
Substitution list v2 for Study 2 (Run 5).

Implements Parts 1.2 and 1.3 of the Run 5 specification:

  * load_rules()        -- read config/substitution_rules_v2.csv
  * load_fixed_terms()  -- read config/fixed_terms_v2.txt
  * check_rules()       -- checks before the application (Part 1.2): every source
                           word matches dictionary A in its arm, every candidate
                           passes verify_neutral_map against A u B
  * SubstitutionEngine  -- application per occurrence (Part 1.3) with one
                           protocol row for every lexicon hit

Design notes
------------
Matching is on the exact lowercased surface form: the rule table lists inflected
forms separately ('help', 'helping', 'helped'), so a token is only handled by a
rule whose source_word equals its surface form. A lexicon hit without a matching
rule stays untreated and is logged with a reason.

Rule choice follows the specification: a phrase rule (PH) wins; otherwise the
most specific syntactic position wins, in the order

    PH > V+O+V > V+V > V+O > N+DET > M > N > V > ADJ > ADV > *

Protection is evaluated before the rule is applied and always produces a reason
in the protocol, never a silent skip.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

# ── D1: pronouns and modal verbs are never replaced ──────────────────────────
# Fixed on 2026-10-11 before any result with the v2 list. These tokens form the
# group 'function_word_kept'; their share of all lexicon hits is reported.
FUNCTION_WORDS_KEPT = {
    "we", "us", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "will", "won't", "wo", "should", "shouldn't",
}
# Contractions of 'you' ("you'll", "you've", "you're", "you'd") are covered by
# this prefix; spaCy splits them, so the "'ll"/"'ve" parts never match a rule.
FUNCTION_WORD_PREFIXES = ("you'",)

# Entity labels that mark a token as part of a proper name.
PROPN_ENT_LABELS = {"ORG", "PERSON", "GPE", "FAC", "WORK_OF_ART", "LAW", "EVENT"}

# Most specific position first.
POSITION_PRIORITY = ["PH", "V+O+V", "V+V", "V+O", "N+DET", "M", "N", "V", "ADJ", "ADV", "*"]

OBJECT_PLACEHOLDER = "..."

ARM_MATCHER = {"communion": "match_communion", "agency": "match_agency"}

# Reasons an occurrence can stay untreated, in the order used by the coverage table.
UNTREATED_REASONS = [
    "fixed_term",
    "proper_name",
    "capitalised_mid_sentence",
    "hyphenated",
    "function_word_kept",
    "no_rule_for_word",
    "no_rule_for_position",
    "other_sense",
    "no_neutral_candidate",
    "protect_comp",
    "protect_mod",
    "validation",
]


def _is_adjectival(tok) -> bool:
    """
    True for adjectives and for participles in adjectival use ('an established
    firm', 'a trusted advisor', 'a caring physician'), which spaCy tags VERB/VBN
    or VERB/VBG with dep amod.
    """
    if tok.pos_ == "ADJ":
        return True
    return tok.tag_ in ("VBG", "VBN") and tok.dep_ == "amod"


def _is_verbal(tok) -> bool:
    """
    True for tokens in verbal use. A participle used attributively ('an
    established firm', 'a trusted advisor') is adjectival and left to the ADJ
    rule. A gerund coordinated with a verb ('working and teaching in Boston')
    is verbal even where spaCy tags it as a noun.
    """
    if tok.dep_ == "amod":
        return False
    if tok.pos_ == "VERB":
        return True
    return (tok.dep_ == "conj" and tok.head.pos_ == "VERB"
            and tok.lower_.endswith("ing"))


def is_function_word_kept(tok_lower: str) -> bool:
    """D1: pronouns and modal verbs, never replaced."""
    if tok_lower in FUNCTION_WORDS_KEPT:
        return True
    return any(tok_lower.startswith(p) for p in FUNCTION_WORD_PREFIXES)


# ── Rule table ───────────────────────────────────────────────────────────────

@dataclass
class Rule:
    rule_id: str
    arm: str
    source_word: str
    n_pool_word: int
    word_class: str
    position: str
    action: str
    pattern: str
    candidates: list[str]
    protect: str
    weak: bool
    note: str
    target_v1: str
    lex_entry: str
    # Filled by check_rules(): the first candidate that passes the A u B check.
    target: str | None = None
    candidate_failures: list[str] = field(default_factory=list)
    # Set to False by apply_validation() once a rule has failed D7.
    validated: bool = True

    @property
    def pattern_tokens(self) -> list[str]:
        return self.pattern.lower().split() if self.pattern else []


def load_rules(path: str | Path = "config/substitution_rules_v2.csv") -> list[Rule]:
    """Read the v2 rule table. Raises on a duplicate rule_id or an unknown field value."""
    rules: list[Rule] = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            cands = [c.strip() for c in row["candidates"].split("|") if c.strip()]
            rules.append(Rule(
                rule_id=row["rule_id"].strip(),
                arm=row["arm"].strip(),
                source_word=row["source_word"].strip().lower(),
                n_pool_word=int(row["n_pool_word"]) if row["n_pool_word"].strip() else 0,
                word_class=row["word_class"].strip(),
                position=row["position"].strip(),
                action=row["action"].strip(),
                pattern=row["pattern"].strip(),
                candidates=cands,
                protect=row["protect"].strip(),
                weak=row["weak"].strip().lower() == "x",
                note=row["note"].strip(),
                target_v1=row["target_v1"].strip(),
                lex_entry=row["lex_entry"].strip(),
            ))

    seen: set[str] = set()
    for r in rules:
        if r.rule_id in seen:
            raise ValueError(f"load_rules: duplicate rule_id {r.rule_id!r}")
        seen.add(r.rule_id)
        if r.arm not in ARM_MATCHER:
            raise ValueError(f"{r.rule_id}: unknown arm {r.arm!r}")
        if r.position not in POSITION_PRIORITY:
            raise ValueError(f"{r.rule_id}: unknown position {r.position!r}")
        if r.action not in ("replace", "delete", "other_sense"):
            raise ValueError(f"{r.rule_id}: unknown action {r.action!r}")
        if r.position == "PH" and not r.pattern:
            raise ValueError(f"{r.rule_id}: position PH without a pattern")
        if r.position == "PH" and r.source_word not in r.pattern_tokens:
            raise ValueError(
                f"{r.rule_id}: source_word {r.source_word!r} not in pattern {r.pattern!r}")
    return rules


def load_fixed_terms(path: str | Path = "config/fixed_terms_v2.txt") -> list[list[str]]:
    """Read the fixed terms as lowercased token lists, longest first."""
    terms = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip().lower()
        if line:
            terms.append(line.split())
    terms.sort(key=len, reverse=True)
    return terms


# ── Part 1.2: checks before the application ──────────────────────────────────

def check_rules(rules: list[Rule], lex_a, lex_ab) -> dict:
    """
    Part 1.2. Two checks, both reported with numbers:

      1. Every source_word matches dictionary A in its own arm. Rules that fail
         are discarded and listed (expected 0).
      2. Every candidate is checked with verify_neutral_map against A u B, so the
         replacement is neutral under both dictionary versions. The first
         candidate that passes is used. Rules without a passing candidate keep
         their occurrences untreated and are listed.

    Sets rule.target in place and returns a report.
    """
    from src.counterfactual import verify_neutral_map

    source_failures: list[Rule] = []
    no_candidate: list[Rule] = []
    delete_without_fallback: list[Rule] = []
    kept: list[Rule] = []

    for r in rules:
        matcher = getattr(lex_a, ARM_MATCHER[r.arm])
        if not matcher(r.source_word):
            source_failures.append(r)
            continue
        kept.append(r)

        if r.action == "other_sense":
            continue

        for cand in r.candidates:
            words = [w for w in cand.split() if w != OBJECT_PLACEHOLDER]
            conflicts = verify_neutral_map({r.source_word: " ".join(words)}, lex_ab, r.arm)
            if conflicts:
                r.candidate_failures.append(
                    f"{cand}: {'; '.join(c.strip() for c in conflicts)}")
            else:
                r.target = cand
                break

        if r.target is None:
            # A delete rule without a usable candidate can still delete; it only
            # lacks the replacement fallback. Everything else stays untreated.
            if r.action == "delete":
                delete_without_fallback.append(r)
            else:
                no_candidate.append(r)

    return {
        "n_rules_in": len(rules),
        "n_rules_kept": len(kept),
        "rules": kept,
        "source_failures": source_failures,
        "no_candidate": no_candidate,
        "delete_without_fallback": delete_without_fallback,
        "candidate_failures": [(r.rule_id, f) for r in kept for f in r.candidate_failures],
    }


def apply_validation(rules: list[Rule], failed_rule_ids: set[str]) -> None:
    """D7: mark rules that failed the rater validation; their occurrences stay untreated."""
    for r in rules:
        if r.rule_id in failed_rule_ids:
            r.validated = False


# ── Part 1.3: application per occurrence ─────────────────────────────────────

@dataclass
class Occurrence:
    """One protocol row per lexicon hit."""
    bio_id: object
    arm: str
    token: str
    position: int
    rule_id: str
    action_taken: str            # replaced | deleted | untreated
    reason_untreated: str
    target: str
    context_original: str
    context_manipulated: str
    word_class: str
    # True where a sentence-initial noun was read as a verb (see _verb_reading).
    verb_reading: bool = False
    # Words of the other dimension that a phrase replacement removes as well.
    side_effect: str = ""


@dataclass
class ManipulationV2:
    original: str
    manipulated: str
    occurrences: list[Occurrence]
    n_replaced: int
    n_deleted: int
    n_untreated: int
    n_verb_reading: int = 0


class SubstitutionEngine:
    """
    Applies the v2 list to one arm of one bio, on the original text, using a
    spaCy parse (en_core_web_lg).
    """

    def __init__(self, rules: list[Rule], fixed_terms: list[list[str]], lex_a,
                 context_tokens: int = 8):
        self.fixed_terms = fixed_terms
        self.context_tokens = context_tokens
        self.lex_a = lex_a
        self.by_word: dict[tuple[str, str], list[Rule]] = {}
        for r in rules:
            self.by_word.setdefault((r.arm, r.source_word), []).append(r)
        # Most specific position first; among phrase rules the longer phrase
        # first, so 'takes care of' is tried before 'care for'.
        for rs in self.by_word.values():
            rs.sort(key=lambda r: (POSITION_PRIORITY.index(r.position),
                                   -len(r.pattern_tokens)))
        # Index the fixed terms by their first token, so a bio is only scanned
        # for the terms that can actually start in it.
        self._terms_by_head: dict[str, list[list[str]]] = {}
        for term in fixed_terms:
            self._terms_by_head.setdefault(term[0], []).append(term)

    # ── protections ──────────────────────────────────────────────────────────

    def _fixed_term_spans(self, doc) -> set[int]:
        """Token indices covered by a fixed term (token sequence, case-insensitive)."""
        lowered = [t.lower_ for t in doc]
        covered: set[int] = set()
        n = len(lowered)
        for i, word in enumerate(lowered):
            for term in self._terms_by_head.get(word, ()):
                k = len(term)
                if i + k <= n and lowered[i:i + k] == term:
                    covered.update(range(i, i + k))
        return covered

    @staticmethod
    def _is_proper_name(tok) -> bool:
        return tok.pos_ == "PROPN" or tok.ent_type_ in PROPN_ENT_LABELS

    @staticmethod
    def _is_capitalised_mid_sentence(tok) -> bool:
        if not tok.text[:1].isupper():
            return False
        return tok.i != tok.sent.start

    @staticmethod
    def _is_hyphenated(tok) -> bool:
        doc = tok.doc
        if tok.i > 0 and doc[tok.i - 1].text == "-" and not doc[tok.i - 1].whitespace_:
            return True
        if tok.i + 1 < len(doc) and doc[tok.i + 1].text == "-" and not tok.whitespace_:
            return True
        return False

    @staticmethod
    def _in_noun_noun_compound(tok) -> bool:
        """protect = comp: the token is part of a noun-noun compound, on either side."""
        if tok.dep_ == "compound" and tok.head.pos_ in ("NOUN", "PROPN"):
            return True
        return any(c.dep_ == "compound" and c.pos_ in ("NOUN", "PROPN") for c in tok.children)

    @staticmethod
    def _modifies_following_noun(tok) -> bool:
        """protect = mod, and position M: the token determines a noun after it."""
        if (tok.dep_ in ("compound", "amod", "nmod")
                and tok.head.i > tok.i
                and tok.head.pos_ in ("NOUN", "PROPN")):
            return True
        # 'has teaching experience' is parsed with the gerund as root and the
        # noun as its object. A gerund whose only auxiliary is a form of 'have'
        # is nominal there, so it modifies the noun that follows it. The
        # progressive ('has been teaching students') keeps a 'be' auxiliary and
        # stays verbal.
        if tok.tag_ == "VBG" and tok.dep_ == "ROOT":
            auxes = [c for c in tok.children if c.dep_ in ("aux", "auxpass")]
            if len(auxes) == 1 and auxes[0].lemma_ == "have":
                doc = tok.doc
                nxt = doc[tok.i + 1] if tok.i + 1 < len(doc) else None
                if (nxt is not None and nxt.pos_ in ("NOUN", "PROPN")
                        and nxt.head.i == tok.i):
                    return True
        return False

    # ── position predicates ──────────────────────────────────────────────────

    @staticmethod
    def _clause_verb(tok, max_gap: int = 4):
        """
        A bare-infinitive verb governed by tok, as in 'help provide' and
        'help patients achieve'. spaCy attaches the object of the second verb to
        that verb, not to 'help', so the clause verb is the anchor and the object
        is whatever sits between the two verbs.

        The syntactic link (ccomp/xcomp) is tried first. Short bio sentences are
        sometimes parsed with the two verbs in sibling positions instead, so a
        following bare infinitive within max_gap tokens also counts, as long as
        nothing but a plain noun phrase stands between the two.
        """
        for c in tok.children:
            if (c.pos_ == "VERB" and c.tag_ == "VB"
                    and c.i > tok.i and c.dep_ in ("ccomp", "xcomp")):
                return c
        doc = tok.doc
        for j in range(tok.i + 1, min(len(doc), tok.i + 1 + max_gap)):
            t = doc[j]
            if t.pos_ == "VERB" and t.tag_ == "VB":
                return t
            if t.pos_ not in ("NOUN", "PROPN", "PRON", "DET", "ADJ", "NUM"):
                return None
        return None

    @staticmethod
    def _has_direct_object(tok) -> bool:
        return any(c.dep_ in ("dobj", "obj") and c.i > tok.i for c in tok.children)

    def _side_effect(self, doc, start: int, end: int, source_index: int, arm: str) -> str:
        """
        Words of the other dimension that a multi-token replacement removes as
        well. 'takes care of' is replaced in the communion arm, which also takes
        out the agency word 'takes'; that is noted here rather than left silent.
        """
        if end - start <= 1:
            return ""
        other = "agency" if arm == "communion" else "communion"
        matcher = getattr(self.lex_a, ARM_MATCHER[other])
        removed = [doc[j].lower_ for j in range(start, end)
                   if j != source_index and matcher(doc[j].lower_)]
        if not removed:
            return ""
        return f"removes {other} word(s): {', '.join(removed)}"

    @staticmethod
    def _verb_reading(tok, rules: list[Rule]) -> bool:
        """
        A source word that is the first word of a sentence, tagged as a noun, and
        has a V or V+O rule is read as a verb: in these biographies the name has
        been removed, so the first word of a sentence is the predicate of the
        missing subject ('practices in Los Angeles ...').

        A word that modifies the noun directly after it ('practice location is
        listed as ...') stays nominal; a predicate does not premodify a noun.
        """
        if tok.pos_ != "NOUN" or tok.i != tok.sent.start:
            return False
        if not any(r.position in ("V", "V+O") for r in rules):
            return False
        return not SubstitutionEngine._modifies_following_noun(tok)

    def _matches_position(self, tok, rule: Rule, verb_reading: bool = False):
        """
        Does the token sit in the position the rule asks for? Returns None for no
        match, otherwise a dict with the token span to edit and, for V+O+V, the
        object text.

        With verb_reading the nominal positions are suppressed and the verbal
        ones apply although spaCy tagged the token as a noun.
        """
        pos = rule.position
        if verb_reading and pos in ("N", "N+DET", "M"):
            return None

        if pos == "PH":
            span = self._match_pattern(tok, rule)
            return {"start": span[0], "end": span[1]} if span else None

        if pos == "*":
            return {"start": tok.i, "end": tok.i + 1}

        if pos == "N":
            return {"start": tok.i, "end": tok.i + 1} if tok.pos_ == "NOUN" else None

        if pos == "N+DET":
            if tok.pos_ != "NOUN":
                return None
            if any(c.dep_ in ("det", "poss") for c in tok.children):
                return {"start": tok.i, "end": tok.i + 1}
            return None

        if pos == "M":
            return ({"start": tok.i, "end": tok.i + 1}
                    if self._modifies_following_noun(tok) else None)

        if pos == "ADJ":
            return {"start": tok.i, "end": tok.i + 1} if _is_adjectival(tok) else None

        if pos == "ADV":
            return {"start": tok.i, "end": tok.i + 1} if tok.pos_ == "ADV" else None

        if pos in ("V", "V+V", "V+O", "V+O+V"):
            if not (verb_reading or _is_verbal(tok)):
                return None
            cv = self._clause_verb(tok)
            if pos == "V+V":
                return ({"start": tok.i, "end": tok.i + 1}
                        if cv is not None and cv.i == tok.i + 1 else None)
            if pos == "V+O+V":
                if cv is None or cv.i <= tok.i + 1:
                    return None
                return {"start": tok.i, "end": cv.i,
                        "object": tok.doc[tok.i + 1:cv.i].text.strip()}
            if pos == "V+O":
                if cv is not None or not self._has_direct_object(tok):
                    return None
                return {"start": tok.i, "end": tok.i + 1}
            # plain V: anything verbal the more specific codes did not take
            return {"start": tok.i, "end": tok.i + 1}

        return None

    def _match_pattern(self, tok, rule: Rule):
        """Locate the rule's word sequence around the token. Returns (start, end) or None."""
        doc = tok.doc
        pat = rule.pattern_tokens
        try:
            offset = pat.index(rule.source_word)
        except ValueError:
            return None
        start = tok.i - offset
        end = start + len(pat)
        if start < 0 or end > len(doc):
            return None
        if [t.lower_ for t in doc[start:end]] != pat:
            return None
        return start, end

    # ── delete conditions ────────────────────────────────────────────────────

    @staticmethod
    def _in_enumeration(tok) -> bool:
        if tok.dep_ == "conj":
            return True
        return any(c.dep_ in ("cc", "conj") for c in tok.children)

    def _can_delete(self, tok) -> bool:
        """
        Deletion is allowed for attributive adjectives directly before a noun and
        not inside an enumeration, and for adverbs.
        """
        if tok.pos_ == "ADV":
            return True
        if not _is_adjectival(tok):
            return False
        if tok.dep_ != "amod" or tok.head.i <= tok.i or tok.head.pos_ not in ("NOUN", "PROPN"):
            return False
        if self._in_enumeration(tok):
            return False
        doc = tok.doc
        # premodifier chain only: nothing but further modifiers in between
        for j in range(tok.i + 1, tok.head.i):
            if doc[j].pos_ not in ("ADJ", "NOUN", "PROPN", "ADV"):
                return False
        return True

    # ── main entry point ─────────────────────────────────────────────────────

    def apply(self, doc, arm: str, bio_id=None) -> ManipulationV2:
        matcher = getattr(self.lex_a, ARM_MATCHER[arm])
        fixed = self._fixed_term_spans(doc)
        pieces = [t.text + t.whitespace_ for t in doc]
        edits: list[tuple[int, int, str | None]] = []   # (start, end, replacement|None)
        occurrences: list[Occurrence] = []
        spans: list[tuple[int, int]] = []               # edit window per occurrence
        consumed: set[int] = set()

        for tok in doc:
            if tok.i in consumed:
                continue
            low = tok.lower_
            if not matcher(low):
                continue

            rules = self.by_word.get((arm, low), [])
            word_class = rules[0].word_class if rules else ""

            verb_reading = self._verb_reading(tok, rules)

            def log(rule_id, action_taken, reason, target, start=None, end=None,
                    _wc=None, side_effect=""):
                occurrences.append(Occurrence(
                    bio_id=bio_id, arm=arm, token=tok.text, position=tok.i,
                    rule_id=rule_id, action_taken=action_taken,
                    reason_untreated=reason, target=target,
                    context_original="", context_manipulated="",
                    word_class=word_class if _wc is None else _wc,
                    verb_reading=verb_reading, side_effect=side_effect,
                ))
                spans.append((tok.i if start is None else start,
                              tok.i + 1 if end is None else end))

            # D1 first: function words are never treated.
            if is_function_word_kept(low):
                log("", "untreated", "function_word_kept", "")
                continue

            # Protection before the rule, with a reason in the protocol.
            if tok.i in fixed:
                log("", "untreated", "fixed_term", "")
                continue
            if self._is_proper_name(tok):
                log("", "untreated", "proper_name", "")
                continue
            if self._is_capitalised_mid_sentence(tok):
                log("", "untreated", "capitalised_mid_sentence", "")
                continue
            if self._is_hyphenated(tok):
                log("", "untreated", "hyphenated", "")
                continue

            if not rules:
                log("", "untreated", "no_rule_for_word", "")
                continue

            chosen, span = None, None
            for r in rules:
                hit = self._matches_position(tok, r, verb_reading)
                if hit is not None:
                    chosen, span = r, hit
                    break
            if chosen is None:
                log("", "untreated", "no_rule_for_position", "")
                continue

            wc = chosen.word_class

            if chosen.protect == "comp" and self._in_noun_noun_compound(tok):
                log(chosen.rule_id, "untreated", "protect_comp", "", _wc=wc)
                continue
            if chosen.protect == "mod" and self._modifies_following_noun(tok):
                log(chosen.rule_id, "untreated", "protect_mod", "", _wc=wc)
                continue
            if not chosen.validated:
                log(chosen.rule_id, "untreated", "validation", "", _wc=wc)
                continue
            if chosen.action == "other_sense":
                log(chosen.rule_id, "untreated", "other_sense", "", _wc=wc)
                continue

            start, end = span["start"], span["end"]

            side = self._side_effect(doc, start, end, tok.i, arm)

            if chosen.action == "delete":
                if self._can_delete(tok):
                    edits.append((start, end, None))
                    consumed.update(range(start, end))
                    log(chosen.rule_id, "deleted", "", "", start, end, _wc=wc,
                        side_effect=side)
                    continue
                if chosen.target is None:
                    log(chosen.rule_id, "untreated", "no_neutral_candidate", "", _wc=wc)
                    continue
                repl = _preserve_case(tok.text, chosen.target)
                edits.append((start, end, repl))
                consumed.update(range(start, end))
                log(chosen.rule_id, "replaced", "", chosen.target, start, end, _wc=wc,
                    side_effect=side)
                continue

            # action == replace
            if chosen.target is None:
                log(chosen.rule_id, "untreated", "no_neutral_candidate", "", _wc=wc)
                continue
            target = chosen.target
            if OBJECT_PLACEHOLDER in target:
                target = target.replace(OBJECT_PLACEHOLDER, span.get("object", "")).strip()
                target = re.sub(r"\s{2,}", " ", target)
            anchor = tok.text if start == tok.i else doc[start].text
            repl = _preserve_case(anchor, target)
            edits.append((start, end, repl))
            consumed.update(range(start, end))
            log(chosen.rule_id, "replaced", "", chosen.target, start, end, _wc=wc,
                side_effect=side)

        manipulated = _render(doc, pieces, edits)
        for occ, (s, e) in zip(occurrences, spans):
            occ.context_original = _context(doc, s, e, self.context_tokens)
            occ.context_manipulated = _context_manipulated(
                doc, pieces, edits, s, e, self.context_tokens)

        return ManipulationV2(
            original=doc.text,
            manipulated=manipulated,
            occurrences=occurrences,
            n_replaced=sum(1 for o in occurrences if o.action_taken == "replaced"),
            n_deleted=sum(1 for o in occurrences if o.action_taken == "deleted"),
            n_untreated=sum(1 for o in occurrences if o.action_taken == "untreated"),
            n_verb_reading=sum(1 for o in occurrences if o.verb_reading),
        )


# ── rendering helpers ────────────────────────────────────────────────────────

def _preserve_case(original: str, replacement: str) -> str:
    """Match the capitalisation of the original token in the replacement."""
    if original.isupper() and len(original) > 1:
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _apply_edits(pieces: list[str], doc, edits, lo: int, hi: int) -> list[str]:
    """
    Apply every edit that lies inside [lo, hi) to a copy of pieces[lo:hi].

    Each token piece carries its own trailing whitespace, so a deleted token is
    dropped together with that whitespace and the neighbours keep exactly one
    separator. Nothing outside an edited place is touched: no cleanup runs over
    the whole text, which would also change spacing the original already had.
    """
    sub = list(pieces[lo:hi])
    sent_starts = {s.start for s in doc.sents}
    for start, end, repl in edits:
        if start < lo or end > hi:
            continue
        if repl is None:
            for j in range(start, end):
                sub[j - lo] = ""
            if start in sent_starts:
                nxt = _next_nonempty(sub, end - lo)
                if nxt is not None:
                    sub[nxt] = sub[nxt][:1].upper() + sub[nxt][1:]
        else:
            sub[start - lo] = repl + doc[end - 1].whitespace_
            for j in range(start + 1, end):
                sub[j - lo] = ""
        _fix_article_before(sub, doc, start, lo)
    return sub


# Words that begin with a vowel letter but a consonant sound, so they take 'a'
# ('a unit', 'a useful tool'), and words with a silent h, which take 'an'.
_CONSONANT_ONSET_PREFIXES = ("uni", "use", "usi", "usu", "util", "uto", "ubiq",
                             "eu", "one", "once")
_SILENT_H_PREFIXES = ("hour", "honest", "honor", "honour", "heir")
# Letters whose spoken name begins with a vowel, for acronyms ('an MBA').
_VOWEL_SOUNDING_LETTERS = set("AEFHILMNORSX")


def _takes_an(word: str) -> bool:
    """Whether the indefinite article in front of this word is 'an'."""
    bare = word.strip().strip("\"'([{.,;:!?")
    if not bare:
        return False
    lowered = bare.lower()
    if lowered.startswith(_SILENT_H_PREFIXES):
        return True
    if bare.isupper() and len(bare) > 1:
        return bare[0] in _VOWEL_SOUNDING_LETTERS
    if lowered.startswith(_CONSONANT_ONSET_PREFIXES):
        return False
    return lowered[0] in "aeiou"


def _fix_article_before(sub: list[str], doc, start: int, lo: int) -> None:
    """
    Adjust an article that now stands in front of a different word. Only the
    token directly before the edit is looked at, so no a/an elsewhere in the
    text is rewritten.
    """
    prev = start - 1
    if prev < lo:
        return
    piece = sub[prev - lo]
    bare = piece.strip()
    if bare.lower() not in ("a", "an"):
        return
    nxt = _next_nonempty(sub, start - lo)
    if nxt is None:
        return
    following = sub[nxt].strip()
    if not following or not following[0].isalpha():
        return
    base = "an" if _takes_an(following) else "a"
    if bare[0].isupper():
        base = base.capitalize()
    sub[prev - lo] = base + piece[len(bare):]


def _next_nonempty(out: list[str], start: int):
    for j in range(start, len(out)):
        if out[j].strip():
            return j
    return None


def _render(doc, pieces: list[str], edits) -> str:
    """Apply all edits to the token pieces and tidy the result."""
    if not edits:
        return doc.text
    return "".join(_apply_edits(pieces, doc, edits, 0, len(doc))).strip()


def _context(doc, start: int, end: int, k: int) -> str:
    lo = max(0, start - k)
    hi = min(len(doc), end + k)
    return doc[lo:hi].text.strip()


def _context_manipulated(doc, pieces, edits, start: int, end: int, k: int) -> str:
    """The same window after the edits, so a rater sees the changed place in context."""
    lo = max(0, start - k)
    hi = min(len(doc), end + k)
    return "".join(_apply_edits(pieces, doc, edits, lo, hi)).strip()
