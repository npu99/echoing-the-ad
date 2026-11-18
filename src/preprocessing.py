"""
Preprocessing helpers for the Bias-in-Bios thesis pipeline.

Two-stage gender scrubbing following De-Arteaga et al. (2019):
  1. remove_gender_markers — removes pronouns/titles via regex
  2. remove_person_names   — removes gender-indicative first names via spaCy NER + gender_guesser

Usage in downstream notebooks:
    from src.preprocessing import load_ner, remove_gender_markers, remove_person_names
    nlp_ner, guesser = load_ner()
    text_clean = remove_person_names(remove_gender_markers(raw_text), nlp_ner, guesser)
"""

import re
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PARQUET_FILES = [
    'data/train-00000-of-00001-0ab65b32c47407e8.parquet',
    'data/test-00000-of-00001-5598c840ce8de1ee.parquet',
    'data/dev-00000-of-00001-e6551072fff26949.parquet',
]

OCCUPATION_MAP = {
    0: 'accountant',        1: 'architect',          2: 'attorney',
    3: 'chiropractor',      4: 'comedian',           5: 'composer',
    6: 'dentist',           7: 'dietitian',          8: 'dj',
    9: 'filmmaker',        10: 'interior_designer',  11: 'journalist',
   12: 'model',            13: 'nurse',             14: 'painter',
   15: 'paralegal',        16: 'pastor',            17: 'personal_trainer',
   18: 'photographer',     19: 'physician',         20: 'poet',
   21: 'professor',        22: 'psychologist',      23: 'rapper',
   24: 'software_engineer', 25: 'surgeon',          26: 'teacher',
   27: 'yoga_teacher',
}

GENDER_MAP = {0: 'Male', 1: 'Female'}

# Exact De-Arteaga et al. (2019) scrubbing list.
# Gendered nouns (man, woman, men, women) are intentionally excluded —
# they are content words, not subject-referencing markers.
_MARKER_RE = re.compile(
    r'\b(?:he|she|her|his|him|hers|himself|herself|mr|mrs|ms)\b',
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Raw data loading
# ---------------------------------------------------------------------------

def load_raw_df() -> pd.DataFrame:
    """Load and merge all three HuggingFace parquet splits into one DataFrame."""
    df = pd.concat(
        [pd.read_parquet(f'hf://datasets/LabHC/bias_in_bios/{f}') for f in PARQUET_FILES],
        ignore_index=True,
    )
    df['occupation']   = df['profession'].map(OCCUPATION_MAP)
    df['gender_label'] = df['gender'].map(GENDER_MAP)
    df['word_count']   = df['hard_text'].str.split().str.len()
    return df


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    EDA-decided cleaning steps:
      1. Drop duplicate bios (809, 0.20%)
      2. Remove rapper occupation (136 Female — below OLS reliability threshold)
    """
    df = df.drop_duplicates(subset='hard_text').reset_index(drop=True)
    df = df[df['occupation'] != 'rapper'].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Stage 1: Gender marker removal
# ---------------------------------------------------------------------------

def remove_gender_markers(text: str) -> str:
    """
    Remove explicit gender markers following De-Arteaga et al. (2019).
    Gaps collapsed to single space; no placeholder token inserted.
    """
    text = _MARKER_RE.sub('', text)
    return re.sub(r'\s+', ' ', text).strip()


# ---------------------------------------------------------------------------
# Stage 2: First-name removal (spaCy NER + gender_guesser)
# ---------------------------------------------------------------------------

def load_ner(model: str = 'en_core_web_lg'):
    """
    Load the spaCy NER model (NER component only) and gender_guesser Detector.
    Call once at the top of a notebook; pass the returned objects to
    remove_person_names / batch_remove_person_names.

    Returns: (nlp_ner, guesser)
    """
    import spacy
    import gender_guesser.detector as gg
    nlp_ner = spacy.load(model, disable=['tok2vec', 'tagger', 'parser',
                                          'attribute_ruler', 'lemmatizer'])
    guesser = gg.Detector()
    return nlp_ner, guesser


def _apply_name_removal(doc, text: str, guesser) -> str:
    """
    Core name-removal logic operating on a pre-computed spaCy Doc.
    Separated from NER inference so it can be used inside nlp.pipe() loops.
    """
    to_remove = []
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            first_token = ent.text.split()[0]
            g = guesser.get_gender(first_token)
            if g in ('male', 'female', 'mostly_male', 'mostly_female'):
                to_remove.append(re.escape(ent.text))
    if not to_remove:
        return text
    pattern = re.compile('|'.join(to_remove), re.IGNORECASE)
    text = pattern.sub('', text)
    return re.sub(r'\s+', ' ', text).strip()


def remove_person_names(text: str, nlp_ner, guesser) -> str:
    """Single-text wrapper — use for spot-checks. For bulk processing use nlp_ner.pipe()."""
    return _apply_name_removal(nlp_ner(text), text, guesser)


def batch_remove_person_names(texts, nlp_ner, guesser, batch_size: int = 512):
    """
    Efficient batch version using nlp_ner.pipe().
    texts : list or Series of strings (already marker-scrubbed)
    Returns: list of cleaned strings
    """
    results = []
    for doc, text in zip(nlp_ner.pipe(texts, batch_size=batch_size), texts):
        results.append(_apply_name_removal(doc, text, guesser))
    return results
