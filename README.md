# Echoing the Ad: Gendered Language in Professional Self-Descriptions and Style Fit in AI-Based Screening

Code and results of the master's thesis. Two studies share one pipeline: Study 1
measures how women and men describe themselves in professional biographies, and
Study 2 asks what three AI-based screeners make of that wording when they rank
the same biographies against real job advertisements.

Every number in the thesis can be found in the stored output of the cell that
printed it, and `THESIS_MAP.md` says which section of which notebook that is.
**Notebooks 01 to 04, 04b, 06, 06a, 07 and 07a were not re-executed for
submission. Notebooks 04c and 06b were re-executed on 18 November 2026 to draw
Figures 1 and 3 and to add the descriptive checks of sections 33 to 35; the
numbers of the main analysis are unchanged.**

## Notebooks

| Notebook | What it does | Thesis |
| --- | --- | --- |
| `01_eda.ipynb` | Loads Bias in Bios, cleans it, describes the corpus and compares the share of women per occupation with the BLS | Chapter 3, Figure A1, Tables A1 and A3 |
| `02_preprocessing.ipynb` | Removes gender markers and first names, validates the removal and runs the gender-leakage test | Chapter 3, Table A3 |
| `03_feature_engineering.ipynb` | Builds the four linguistic features from the dictionaries and writes the feature table | Chapter 3, Tables A3 and B1 |
| `04_study1_analysis.ipynb` | Study 1: OLS with occupation effects, the two hurdle models, effects per occupation and the composition analyses | Chapter 4, Tables 1 and E2 |
| `04b_study1_robustness_lexicon_AB.ipynb` | Study 1 under the union of both dictionaries and without the function words | Table E1 |
| `04c_thesis_figures_study1.ipynb` | Draws the two Study 1 figures and prints the counting rule behind them | Figures 2 and 3, Table A3 |
| `06_study2_run5.ipynb` | Study 2, main analysis: substitution, content gate, Arm 1 ranking, the two tests fixed in advance and every table the thesis reports | Chapter 5, Tables 2, 3, A3, C1, C2, C3, E3 and E5 |
| `06a_arm2_embeddings_kaggle.ipynb` | Arm 2: embeddings of the search encoder, on a Kaggle GPU | Table D1 |
| `06b_study2_supplements.ipynb` | Supplements: the conditional tests on Arm 2 and Arm 3, the gate diagnosis, the figures of Study 2 and the appendix sections | Figures 1, 4, 5, 6 and E1, Tables A1, A2, A4, C2, C4, D1 and E3 to E10 |
| `07_study2_llm_screening.ipynb` | Arm 3: analysis of the language-model ratings, with the validity and repeatability checks | Chapter 5, Table D1 |
| `07a_llm_scoring_kaggle.ipynb` | Arm 3: the rating run itself, on a Kaggle GPU | Table D1 |

```
01 -> 02 -> 03 -> 04 -> 04b
                   |
                   +--> 04c
03 -> 06 -> 06b
      |      ^
      |      |
      +-> 06a (Kaggle GPU, embeddings of Arm 2)
      |      |
      +-> 07a (Kaggle GPU, ratings of Arm 3) -> 07 ---+
```

Each notebook reads what the one before it wrote. `06a` and `07a` run on Kaggle
with a GPU; their result files are brought back into `outputs/study2_run5/`
before `06b` and `07` are run, so `06b` comes last.

## Data

Neither corpus is part of this repository. Both keep the terms of their own
sources, and redistributing them is not covered by the licence of this code.

* Biographies: `LabHC/bias_in_bios`,
  <https://huggingface.co/datasets/LabHC/bias_in_bios>. Notebook 01 reads the
  three published parquet files over the network, so nothing has to be prepared
  for Study 1.
* Advertisements: LinkedIn Job Postings (2023-2024) by Arsh Koneru, version 13,
  <https://www.kaggle.com/datasets/arshkon/linkedin-job-postings>. Download the
  dataset and put `job_postings.csv` at
  `data/job_descriptions/job_postings.csv` before notebook 06 is run.
* Dictionaries: the agency and communion dictionary of Pietraszkiewicz et al.
  (2019) in `data/lexicons/`, in the published main version and in the union
  used for the robustness check.

## Running it

`environment/requirements.txt` holds the pinned versions and
`environment/README.md` describes the environment and the two Kaggle sessions.
Notebook 02 printed:

```
python 3.13.9, spacy 3.8.14, en_core_web_lg 3.8.0, gender-guesser 0.4.0,
scikit-learn 1.6.1, pandas 2.2.3, numpy 2.1.3, pyarrow 19.0.0, scipy 1.15.3,
statsmodels 0.14.4, matplotlib 3.10.0, seaborn 0.13.2,
sentence-transformers 6.0.1, transformers 5.16.1, torch 2.14.0
```

The two Kaggle runs used the versions of their own image: `06a`
`sentence-transformers 5.7.0` with `torch 2.11.0+cu128`, and `07a`
`ollama 0.40.3` with the model `qwen3:8b`, digest `500a1f067a9f`.

The runtimes the notebooks print are 26 minutes for the name removal in `02`,
34 minutes for the pass over 393,977 biographies in `03`, 2.2 hours for the
Arm 1 embeddings in `06`, and on a Kaggle GPU T4 x2 23 minutes for the
embeddings of `06a` and 3.5 hours for the 49,400 ratings of `07a`. The other
notebooks run in minutes; `04`, whose two hurdle models are the slow step, and
the remaining steps of `06` do not print their time.

## The final run

Study 2 was run five times and the thesis reports the last run only. The
earlier runs differed in the substitution protocol, the content gate and the
sample, and their outputs are not part of this repository. The output folder
`outputs/study2_run5/`, the tag `RUN_TAG` in the notebooks and the name of
notebook 06 carry the internal name of that run, and section 16 of notebook 06
compares it with the run before it.

## Layout

```
README.md       this file
THESIS_MAP.md   every table and figure of the thesis, and where it is printed
LICENSE         MIT, for the code of this repository
.gitignore      the corpora, the caches, the credentials
notebooks/      the eleven notebooks above, with their stored outputs
src/            the modules the notebooks import
config/         the substitution rules, the fixed terms and the occupation table
data/           the dictionaries; the two corpora are not included
outputs/        the figures and the result tables of the thesis
environment/    requirements.txt and a note on the environment
```

The result tables the thesis draws its numbers from are
`outputs/study1_results/` for Study 1, `outputs/study2_run5/` for Study 2 and
`outputs/study2_run5/arm3/main/` for the language model. Data files with one row
per biography, advertisement or rating are not part of this repository, and
neither are the embedding caches.
