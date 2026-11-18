"""
Feature extraction for the Bias-in-Bios thesis pipeline.

Four linguistic features computed from spaCy Docs (full pipeline required --
tagger for POS tags, parser for dependency labels including auxpass):

  F1 agentic_communal_ratio   (agency - communion) / (agency + communion + 1)
  F2 passive_ratio            auxpass tokens / VERB tokens
  F3 achievement_verb_rate    achievement-lexicon hits / word_count * 100
  F4 attribution_passive_rate external-attribution verbs in passive voice / word_count * 100

Usage:
    from src.lexicon import load_lexicons
    from src.features import extract_features
    lex = load_lexicons(version="A")   # or version="AB" for robustness variant
    df_features = extract_features(df, nlp, lex, text_col='text_clean')
"""

import pandas as pd

from src.lexicon import count_multiword_communion


def agentic_communal_ratio(doc, text: str, lex) -> tuple:
    """
    Returns (agency_hits, communion_hits, ratio).
    ratio = (A - C) / (A + C + 1) -- bounded (-1, +1).
    """
    agency_hits = sum(1 for t in doc if lex.match_agency(t.lower_))
    communion_hits = (sum(1 for t in doc if lex.match_communion(t.lower_))
                      + count_multiword_communion(text.lower(), lex.multi_word_communion))
    ratio = (agency_hits - communion_hits) / (agency_hits + communion_hits + 1)
    return agency_hits, communion_hits, ratio


def passive_ratio(doc, word_count: int) -> tuple:
    """
    Returns (auxpass_count, verb_count, passive_ratio, passive_per_100w).
    Not bounded at 1: auxpass-tagged and VERB-tagged tokens are different POS
    classes, so auxpass_count can exceed verb_count for short, aux-heavy bios.
    """
    auxpass_count = sum(1 for t in doc if t.dep_ == 'auxpass')
    verb_count = sum(1 for t in doc if t.pos_ == 'VERB')
    ratio = auxpass_count / max(verb_count, 1)
    per_100w = auxpass_count / max(word_count, 1) * 100
    return auxpass_count, verb_count, ratio, per_100w


def achievement_verb_rate(doc, word_count: int, lex) -> tuple:
    """Returns (matched_terms, rate_per_100_words)."""
    hits = [t.lower_ for t in doc if lex.match_achievement(t.lower_)]
    rate = len(hits) / max(word_count, 1) * 100
    return hits, rate


def external_attribution_rate(doc, word_count: int, lex, require_auxpass: bool = True) -> tuple:
    """
    Returns (hits, rate_per_100_words, examples).

    require_auxpass=True (primary, validated on the full Bias-in-Bios dataset:
        4.19% coverage): counts recognition-lexicon verbs that are heads of a
        passive auxiliary, e.g. "was awarded". auxpass operationalises the
        passive-attribution construct itself and filters bare-noun false positives.
    require_auxpass=False: counts recognition-lexicon verbs in any voice.
        Diagnostic only -- use to re-check the primary/fallback decision if
        applying this to a different dataset where auxpass coverage may differ.
    """
    examples = []
    if require_auxpass:
        for t in doc:
            if t.dep_ == 'auxpass' and lex.match_recognition(t.head.lower_):
                examples.append(f"{t.text} {t.head.text}")
    else:
        for t in doc:
            if lex.match_recognition(t.lower_):
                examples.append(t.text)
    rate = len(examples) / max(word_count, 1) * 100
    return len(examples), rate, examples


def extract_features(df: pd.DataFrame, nlp, lex, text_col: str = 'text_clean',
                      word_count_col: str = 'word_count',
                      use_auxpass_for_f4: bool = True, batch_size: int = 256) -> pd.DataFrame:
    """
    Run the full feature pipeline over df[text_col] using nlp.pipe() and return
    a copy of df with six new columns: agentic_communal_ratio, passive_ratio,
    achievement_verb_rate, attribution_passive_rate, agency_count, communion_count.

    agency_count/communion_count are the raw F1 hit counts underlying the ratio
    -- needed for the agency/communion decomposition in Notebook 04, since they
    cannot be recovered from the ratio alone ((A-C)/(A+C+1) has two unknowns).
    """
    texts = df[text_col].tolist()
    word_counts = df[word_count_col].tolist()

    acr_list, pr_list, avr_list, apr_list = [], [], [], []
    agency_list, communion_list = [], []
    for doc, text, wc in zip(nlp.pipe(texts, batch_size=batch_size), texts, word_counts):
        agency_hits, communion_hits, acr = agentic_communal_ratio(doc, text, lex)
        _, _, pr, _ = passive_ratio(doc, wc)
        _, avr = achievement_verb_rate(doc, wc, lex)
        _, apr, _ = external_attribution_rate(doc, wc, lex, require_auxpass=use_auxpass_for_f4)

        acr_list.append(acr)
        pr_list.append(pr)
        avr_list.append(avr)
        apr_list.append(apr)
        agency_list.append(agency_hits)
        communion_list.append(communion_hits)

    result = df.copy()
    result['agentic_communal_ratio'] = acr_list
    result['passive_ratio'] = pr_list
    result['achievement_verb_rate'] = avr_list
    result['attribution_passive_rate'] = apr_list
    result['agency_count'] = agency_list
    result['communion_count'] = communion_list
    return result
