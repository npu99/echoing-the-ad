"""
counterfactual.py — Lexicon-based bio manipulation for Study 2.

Two arms (decrease direction):
  decrease_communion(text, nlp, lex, extra_map=None) -> ManipulationResult
  decrease_agency(text, nlp, lex, extra_map=None)    -> ManipulationResult

Method: spaCy token-level substitution using the same Pietraszkiewicz
Big Two matchers as Study 1 src/features.py.

Synonym resolution order (no fallback constant):
  1. COMMUNION_TO_NEUTRAL / AGENCY_TO_NEUTRAL — curated, verified map.
  2. extra_map — WordNet-generated candidates, generated per run via
     discover_unmapped() + wordnet_candidates(), reviewed manually before use.
  3. No entry → token left unchanged. Unmapped tokens are flagged by
     discover_unmapped() so they can be handled before the full statistical run.

PROPN guard: proper-noun tokens are always skipped regardless of lexicon match
(protects job titles, company names, institution names). Requires the spaCy
tagger; compatible with disable=['parser', 'ner'].

Case preservation: ALL-CAPS → ALL-CAPS, Title → Title, lower → lower.
"""
from __future__ import annotations
from dataclasses import dataclass, field

# ── Verified-neutral synonym maps ─────────────────────────────────────────────
# All values confirmed NOT in agency or communion lexicon (verified 2026-09-08).
# Single-word fallback constants ("dedicated", "noted") removed — see module
# docstring for why and how gaps are handled instead.

COMMUNION_TO_NEUTRAL: dict[str, str] = {
    # adjectives
    "warm":          "measured",
    "warmly":        "systematically",
    "caring":        "dedicated",
    "kind":          "noted",
    "kindly":        "noted",
    "gentle":        "measured",
    "gently":        "systematically",
    "generous":      "substantial",
    "generously":    "substantially",
    "compassionate": "dedicated",
    "empathetic":    "attentive",
    "empathic":      "attentive",
    "nurturing":     "developing",
    "supportive":    "structured",
    "inclusive":     "broad",
    "harmonious":    "aligned",
    "collaborative": "coordinated",
    "friendly":      "noted",
    "loving":        "dedicated",
    "loyal":         "dedicated",
    "trusting":      "systematic",
    "trusted":       "dedicated",
    "welcoming":     "accessible",
    "understanding": "attentive",
    "close":         "near",
    "collective":    "combined",
    "human":         "individual",
    "public":        "general",
    # nouns
    "warmth":         "dedication",
    "kindness":       "consideration",
    "compassion":     "dedication",
    "empathy":        "attentiveness",
    "care":           "consideration",
    "nurturance":     "development",
    "community":      "framework",
    "togetherness":   "coordination",
    "harmony":        "coordination",
    "collaboration":  "coordination",
    "partnership":    "interaction",
    "connection":     "link",
    "connections":    "links",
    "relationship":   "interaction",
    "relationships":  "interactions",
    "trust":          "commitment",
    "friendship":     "interaction",
    "love":           "dedication",
    "generosity":     "dedication",
    "inclusion":      "accessibility",
    "support":        "process",
    "career":         "occupation",
    "educator":       "trainer",
    "family":         "household",
    "friend":         "acquaintance",
    "group":          "cluster",
    "groups":         "clusters",
    "law":            "statute",
    "member":         "delegate",
    "teacher":        "instructor",
    "teams":          "divisions",
    "volunteer":      "participant",
    # verbs
    "cares":        "attends",
    "cared":        "provided",
    "nurture":      "develop",
    "nurtures":     "develops",
    "nurtured":     "developed",
    "supports":     "provides",
    "supported":    "provided",
    "loves":        "engages",
    "loved":        "engaged",
    "share":        "provide",
    "shares":       "provides",
    "shared":       "provided",
    "sharing":      "providing",
    "connect":      "engage",
    "connects":     "engages",
    "connected":    "engaged",
    "connecting":   "engaging",
    "collaborate":  "manage",
    "collaborates": "manages",
    "collaborated": "managed",
    "collaborating":"managing",
    "mentor":       "guide",
    "mentors":      "guides",
    "mentored":     "guided",
    "mentoring":    "guiding",
    "affiliates":   "operates",
    "assisted":     "served",
    "assists":      "serves",
    "cooperates":   "functions",
    "counsels":     "advises",
    "educates":     "instructs",
    "help":         "handles",
    "helped":       "handled",
    "learned":      "studied",
    "recommends":   "indicates",
    "understand":   "comprehend",
    # round 2 additions — 13-occupation coverage check
    "educational":   "curricular",
    "social":        "organizational",
    "open":          "available",
    "helpful":       "useful",
    "civil":         "legal",
    "emotionally":   "mentally",
    "education":     "curriculum",
    "services":      "programs",
    "service":       "program",
    "team":          "division",
    "communities":   "districts",
    "fellowship":    "residency",
    "affiliations":  "associations",
    "families":      "households",
    "contributions": "outputs",
    "discussion":    "exchange",
    "discussions":   "exchanges",
    "counsel":       "advisor",
    "guidance":      "direction",
    "appreciation":  "acknowledgment",
    "humans":        "individuals",
    "approach":      "protocol",
    "approaches":    "protocols",
    "assistant":     "coordinator",
    "counseling":    "consultation",
    "taught":        "conveyed",
    "teaching":      "demonstrating",
    "teaches":       "demonstrates",
    "teachers":      "instructors",
    "affiliated":    "associated",
    "offers":        "delivers",
    "contributes":   "submits",
    "helping":       "facilitating",
    "accepts":       "admits",
    "accepted":      "admitted",
    "assisting":     "administering",
    "educating":     "instructing",
    "learn":         "study",
    "supporting":    "reinforcing",
    "volunteering":  "registering",
    "affected":      "impacted",
    "contributing":  "submitting",
}

AGENCY_TO_NEUTRAL: dict[str, str] = {
    # adjectives
    "ambitious":    "focused",
    "assertive":    "direct",
    "competitive":  "focused",
    "confident":    "noted",
    "independent":  "structured",
    "powerful":     "notable",
    "determined":   "dedicated",
    "excellent":    "notable",
    "outstanding":  "notable",
    "exceptional":  "notable",
    "active":       "operational",
    "sure":         "certain",
    # nouns
    "ambition":         "focus",
    "confidence":       "assurance",
    "independence":     "discretion",
    "power":            "capacity",
    "determination":    "stance",
    "excellence":       "quality",
    "achievement":      "outcome",
    "achievements":     "outcomes",
    "accomplishment":   "outcome",
    "accomplishments":  "outcomes",
    "success":          "outcome",
    "successes":        "outcomes",
    "leadership":       "management",
    "mastery":          "quality",
    "victory":          "result",
    "triumph":          "result",
    "strength":         "caliber",
    "strengths":        "qualities",
    "experience":       "background",
    "experiences":      "backgrounds",
    "expert":           "specialist",
    "experts":          "specialists",
    "information":      "data",
    "knowledge":        "foundation",
    "needs":            "requirements",
    "practice":         "method",
    "practices":        "methods",
    "skills":           "techniques",
    "succession":       "sequence",
    # verbs
    "achieve":      "complete",
    "achieves":     "completes",
    "achieved":     "completed",
    "achieving":    "completing",
    "accomplish":   "complete",
    "accomplishes": "completes",
    "accomplished": "completed",
    "accomplishing":"completing",
    "succeed":      "deliver",
    "succeeds":     "delivers",
    "succeeded":    "delivered",
    "lead":         "manage",
    "leads":        "manages",
    "led":          "managed",
    "leading":      "managing",
    "master":       "develop",
    "masters":      "develops",
    "mastered":     "developed",
    "mastering":    "developing",
    "excel":        "perform",
    "excels":       "performs",
    "excelled":     "performed",
    "win":          "deliver",
    "wins":         "delivers",
    "won":          "delivered",
    "winning":      "delivering",
    "earn":         "produce",
    "earns":        "produces",
    "earned":       "produced",
    "earning":      "producing",
    "overcome":     "address",
    "overcomes":    "addresses",
    "overcame":     "addressed",
    "drive":        "guide",
    "drives":       "guides",
    "driven":       "guided",
    "driving":      "guiding",
    "creating":     "generating",
    "doing":        "conducting",
    "establishing": "founding",
    "known":        "documented",
    "making":       "assembling",
    "obtained":     "received",
    "practicing":   "applying",
    "pursued":      "concluded",
    "takes":        "processes",
    # round 2 additions — 13-occupation coverage check
    "professional":     "qualified",
    "professionally":   "technically",
    "professionals":    "technicians",
    "responsible":      "designated",
    "practical":        "applied",
    "successful":       "fruitful",
    "strong":           "robust",
    "advanced":         "specialized",
    "recognized":       "acknowledged",
    "strongly":         "notably",
    "creative":         "innovative",
    "difficult":        "challenging",
    "willing":          "prepared",
    "strategic":        "systematic",
    "experienced":      "encountered",
    "needed":           "required",
    "responsibilities": "assignments",
    "expertise":        "grounding",
    "activity":         "function",
    "activities":       "functions",
    "goals":            "benchmarks",
    "self":             "identity",
    "manager":          "administrator",
    "opportunity":      "prospect",
    "autonomy":         "latitude",
    "intelligence":     "cognition",
    "resilience":       "stability",
    "insight":          "perspective",
    "resolution":       "conclusion",
    "leader":           "coordinator",
    "priority":         "precedence",
    "pursuit":          "trajectory",
    "risk":             "variable",
    "control":          "monitoring",
    "create":           "generate",
    "creates":          "generates",
    "created":          "generated",
    "obtaining":        "gathering",
    "taking":           "assuming",
    "establish":        "found",
    "established":      "founded",
    "pursuing":         "working",
    "organize":         "order",
    "organized":        "ordered",
    "thinking":         "pondering",
}


# ── Core types ────────────────────────────────────────────────────────────────

@dataclass
class ManipulationResult:
    original: str
    manipulated: str
    n_substitutions: int
    substituted_tokens: list[tuple[str, str]] = field(default_factory=list)


# ── Implementation ────────────────────────────────────────────────────────────

def _preserve_case(original: str, replacement: str) -> str:
    """Match the capitalisation of original in replacement."""
    if original.isupper():
        return replacement.upper()
    if original and original[0].isupper():
        return replacement.capitalize()
    return replacement


def _manipulate(text: str, nlp, lex_match_fn, synonym_map: dict[str, str],
                extra_map: dict[str, str] | None = None,
                exclude: set[str] | None = None) -> ManipulationResult:
    doc = nlp(text)
    parts: list[str] = []
    n_subs = 0
    substituted: list[tuple[str, str]] = []

    for tok in doc:
        # PROPN/AUX guard — skip proper nouns and auxiliaries regardless of
        # lexicon match. PROPN protects institution names, job titles, etc.
        # AUX protects "will" (future marker) which the agency lexicon
        # matches but which should never be substituted as a content word.
        if tok.pos_ in ('PROPN', 'AUX'):
            parts.append(tok.text + tok.whitespace_)
            continue

        lower = tok.lower_
        if lex_match_fn(lower):
            # exclude set: caller-specified source words to skip (e.g. known bad substitutions)
            if exclude and lower in exclude:
                parts.append(tok.text + tok.whitespace_)
                continue
            # Resolution order: curated map → WordNet extension → leave unchanged
            repl_lower = synonym_map.get(lower)
            if repl_lower is None and extra_map:
                repl_lower = extra_map.get(lower)

            if repl_lower is not None:
                repl = _preserve_case(tok.text, repl_lower)
                parts.append(repl + tok.whitespace_)
                substituted.append((tok.text, repl))
                n_subs += 1
            else:
                # No synonym available — leave token unchanged.
                # Use discover_unmapped() to surface these before the full run.
                parts.append(tok.text + tok.whitespace_)
        else:
            parts.append(tok.text + tok.whitespace_)

    return ManipulationResult(
        original=text,
        manipulated=''.join(parts),
        n_substitutions=n_subs,
        substituted_tokens=substituted,
    )


def decrease_communion(text: str, nlp, lex,
                       extra_map: dict[str, str] | None = None,
                       exclude: set[str] | None = None) -> ManipulationResult:
    """Replace communion-lexicon tokens with neutral synonyms.

    extra_map: WordNet-generated candidates from wordnet_candidates(), covering
               lemmas absent from COMMUNION_TO_NEUTRAL. Generate per bio-pool
               scope via discover_unmapped() + wordnet_candidates(); review the
               printed table before using in the full statistical run.
    exclude: source words to skip entirely (not replaced). Load from
             config/substitution_exclusions.csv; None = unchanged behaviour (Run 3).
    """
    return _manipulate(text, nlp, lex.match_communion, COMMUNION_TO_NEUTRAL, extra_map, exclude)


def decrease_agency(text: str, nlp, lex,
                    extra_map: dict[str, str] | None = None,
                    exclude: set[str] | None = None) -> ManipulationResult:
    """Replace agency-lexicon tokens with neutral synonyms.

    exclude: source words to skip entirely. None = unchanged behaviour (Run 3).
    """
    return _manipulate(text, nlp, lex.match_agency, AGENCY_TO_NEUTRAL, extra_map, exclude)


# ── WordNet candidate generation ──────────────────────────────────────────────

def _spacy_to_wn_pos(spacy_pos: str):
    """Map spaCy Universal POS tag to WordNet POS constant. Returns None for unmappable tags."""
    from nltk.corpus import wordnet as wn
    return {'VERB': wn.VERB, 'NOUN': wn.NOUN, 'ADJ': wn.ADJ, 'ADV': wn.ADV}.get(spacy_pos)


def discover_unmapped(texts: list[str], nlp, lex_match_fn,
                      synonym_map: dict[str, str]) -> list[tuple[str, str]]:
    """
    Find (token_lower, spaCy_pos) pairs that:
      - match the lexicon
      - have no curated entry in synonym_map
      - are not PROPN (those are skipped by _manipulate anyway)

    Pass all bio-pool texts for the current run scope so coverage is
    pool-specific, not dictionary-wide.

    Returns sorted list for determinism.
    """
    found: set[tuple[str, str]] = set()
    for text in texts:
        doc = nlp(text)
        for tok in doc:
            lower = tok.lower_
            if (lex_match_fn(lower)
                    and lower not in synonym_map
                    and tok.pos_ != 'PROPN'):
                found.add((lower, tok.pos_))
    return sorted(found)


def wordnet_candidates(unmapped: list[tuple[str, str]],
                       lex) -> dict[tuple[str, str], str | None]:
    """
    For each (lemma, pos) pair from discover_unmapped(), find the first
    WordNet synonym that:
      - belongs to the same POS category
      - is a single word (no underscores / spaces)
      - is NOT in either lexicon (agency or communion)

    Returns dict: (lemma, pos) → candidate word, or None if no clean
    candidate found. Caller must supply manual replacements for None cases
    before the full statistical run — do NOT add a fallback constant.
    """
    import nltk
    try:
        from nltk.corpus import wordnet as wn
        wn.synsets('test')
    except LookupError:
        nltk.download('wordnet', quiet=True)
        from nltk.corpus import wordnet as wn

    results: dict[tuple[str, str], str | None] = {}
    for lemma, pos in unmapped:
        wn_pos = _spacy_to_wn_pos(pos)
        if wn_pos is None:
            results[(lemma, pos)] = None
            continue
        candidate = None
        for synset in wn.synsets(lemma, pos=wn_pos):
            for ln in synset.lemma_names():
                if '_' in ln:
                    continue  # skip multi-word synset lemmas
                word = ln.lower()
                if word == lemma:
                    continue
                if not lex.match_agency(word) and not lex.match_communion(word):
                    candidate = word
                    break
            if candidate:
                break
        results[(lemma, pos)] = candidate
    return results


def print_candidate_table(comm_candidates: dict, ag_candidates: dict) -> None:
    """
    Print (lemma, POS, WordNet candidate) table for manual review before
    locking candidates into the curated maps or the full-run extra_map.
    """
    for label, cands in [('COMMUNION', comm_candidates), ('AGENCY', ag_candidates)]:
        print(f"\n=== {label} — unmapped lemmas + WordNet candidates "
              f"(REVIEW BEFORE USE) ===")
        print(f"{'Lemma':<24s}  {'POS':<6s}  {'Candidate':<24s}  Note")
        print("-" * 74)
        for (lemma, pos), cand in sorted(cands.items()):
            note = 'supply manually' if cand is None else 'review'
            print(f"{lemma:<24s}  {pos:<6s}  {str(cand):<24s}  {note}")


# ── Lexicon-neutrality verification ──────────────────────────────────────────

def verify_neutral_map(synonym_map: dict[str, str], lex, label: str) -> list[str]:
    """
    Return a list of conflict strings for any replacement word that is itself
    in the agency or communion lexicon. Run whenever synonym maps are edited
    or WordNet candidates are added via extra_map.
    """
    conflicts = []
    for orig, repl in synonym_map.items():
        for word in repl.split():
            a = lex.match_agency(word.lower())
            c = lex.match_communion(word.lower())
            if a or c:
                conflicts.append(
                    f"  {label}: '{orig}' -> '{repl}' | agency={a} communion={c}")
    return conflicts


if __name__ == '__main__':
    from src.lexicon import load_lexicons
    lex = load_lexicons(version="AB")  # verify_neutral_map uses AB (the larger set)
    print("=== Verifying COMMUNION_TO_NEUTRAL ===")
    for c in verify_neutral_map(COMMUNION_TO_NEUTRAL, lex, "communion"):
        print(c)
    print("=== Verifying AGENCY_TO_NEUTRAL ===")
    for c in verify_neutral_map(AGENCY_TO_NEUTRAL, lex, "agency"):
        print(c)
    print("Verification complete.")
