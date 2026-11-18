# Echoing the Ad: Gendered Language in Professional Self-Descriptions and Style Fit in AI-Based Screening

Code and results of the master's thesis. Two studies share one pipeline: Study 1
measures how women and men describe themselves in professional biographies, and
Study 2 asks what three AI-based screeners make of that wording when they rank
the same biographies against real job advertisements.

**Notebooks 01 to 04, 04b, 06, 06a, 07 and 07a were not re-executed for
submission. Notebooks 04c and 06b were re-executed on 18 November 2026 to draw
Figures 1 and 3 and to add the descriptive checks of sections 33 to 35; the
numbers of the main analysis are unchanged.** Every number in the thesis can be
found in the stored output of the cell that printed it.

## Notebooks

| Notebook | What it does | Where it appears in the thesis |
| --- | --- | --- |
| `01_eda.ipynb` | Loads Bias in Bios, cleans it (duplicates, `rapper`), describes the corpus and compares the share of women per occupation with the BLS | Chapter 3 (data), Figure A1, Tables A1 and A3 |
| `02_preprocessing.ipynb` | Removes gender markers and first names, validates the removal and runs the gender-leakage test | Chapter 3, Table A3 |
| `03_feature_engineering.ipynb` | Builds the four linguistic features from the dictionaries and writes the feature table | Chapter 3, Tables A3 and B1 |
| `04_study1_analysis.ipynb` | Study 1: OLS with occupation effects, the two hurdle models, effects per occupation and the composition analyses | Chapter 4, Table 1, Table E2 |
| `04b_study1_robustness_lexicon_AB.ipynb` | Study 1 under the union of both dictionaries and without the function words | Table E1 |
| `04c_thesis_figures_study1.ipynb` | Draws the two Study 1 figures and prints the counting rule behind them | Figures 2 and 3, Table A3 (effect of the cleaning) |
| `06_study2_run5.ipynb` | Study 2, main run: substitution, content gate, Arm 1 ranking, the two tests fixed in advance and every table the thesis reports | Chapter 5, Tables 2 and 3, Tables A3, C1, C2, C3, E3 and E5 |
| `06a_arm2_embeddings_kaggle.ipynb` | Arm 2: embeddings of the search encoder, on a Kaggle GPU | Table D1 |
| `06b_study2_supplements.ipynb` | Supplements: the conditional tests on Arm 2 and Arm 3, the gate diagnosis, the figures of Study 2 and the appendix sections | Figures 1, 4, 5, 6 and E1, Tables A1, A2, A4, C2, C4, D1, E3, E4, E5, E6, E7, E8, E9 and E10 |
| `07_study2_llm_screening.ipynb` | Arm 3: analysis of the language-model ratings, with the validity and repeatability checks | Chapter 5, Table D1 |
| `07a_llm_scoring_kaggle.ipynb` | Arm 3: the rating run itself, on a Kaggle GPU | Table D1 |

The chapter column follows the version of the thesis submitted with this
repository.

## Tables and figures of the appendix

| Number | What it holds | Where its numbers are printed |
| --- | --- | --- |
| Table A1 | biographies and advertisements per occupation | `01_eda.ipynb` for the biographies, `06b` section 29 for the advertisements |
| Figure A1 | the share of women per occupation, against the BLS | `01_eda.ipynb` |
| Table A2 | the advertisements by occupation | `06b` section 29 |
| Table A3 | sources and preparation of the two corpora | `01`, `02`, `03` and `04c` for the biographies, `06` sections 3 to 5 for the advertisements |
| Table A4 | keyword-matched advertisements whose title names a neighbouring role | `06b` section 34 |
| Table B1 | the word lists | `data/lexicons/`; `03` counts with them |
| Table C1 | examples of the substitution rules | `06` section 3 |
| Table C2 | the rules, their coverage and their validation | `06` sections 3 and 14.10, `06b` sections 20 and 26 |
| Table C3 | agreement of the two raters on the list | `06` section 14.10 |
| Table C4 | four edited biographies, before and after | `06b` section 31 |
| Table D1 | how the three screeners are configured | `06b` section 22, with `06a` and `07a` for the two runs on Kaggle |
| Table D2 | the recruiting models that were checked | desk research on the model pages before Study 2, not from a notebook |
| Table E1 | Study 1 under the union of the dictionaries and without the function words | `04b` |
| Table E2 | the further results of Study 1: count part of the hurdle models, length, occupations | `04` |
| Table E3 | the median change in rank after the communal edit, by occupation | `06b` section 27, tests from `06` section 14.1 and `06b` section 5 |
| Table E4 | the distribution of the change in rank after the communal edit | `06b` section 27 |
| Table E5 | the change per replaced marker and by author gender | `06` section 14.3, `06b` sections 15 and 16 |
| Table E6 | similarity to the own and to the other advertisements | `06b` section 17.3 |
| Table E7 | the experiment and T1 on the three measures of the advertisement | `06b` section 25 |
| Figure E1 | the agentic arm and the focal sample | `06b` section 21 |
| Table E8 | the lead per occupation against the share of women and seniority | `06b` section 32 |
| Table E9 | the upper tenth of a ranking, read as a shortlist | `06b` section 33 |
| Table E10 | T1, T2 and the experiment without the advertisements of Table A4 | `06b` section 35 |

The advertisements were selected and matched in `06` sections 3 to 5; the
earlier notebook that did this is not part of this repository.

## Figures and tables in `outputs/`

| File | Thesis |
| --- | --- |
| `outputs/figures/fig1_framework.png` | Figure 1, the research model |
| `outputs/study1_results/fig3_forest_thesis.png` | Figure 2, the six effects of Study 1 on one d scale |
| `outputs/study1_results/fig4_communion_by_occupation_thesis.png` | Figure 3, the difference in communal words per occupation |
| `outputs/figures/fig5_study2_design.png` | Figure 4, the design of Study 2 in three steps |
| `outputs/figures/fig6_experiment.png` | Figure 5, the experiment |
| `outputs/figures/fig7_original_rankings.png` | Figure 6, style fit and the gender gap in the original rankings |
| `outputs/figures/fig2_share_women.png` | Figure A1, the share of women per occupation |
| `outputs/figures/figE1_experiment_by_arm_sample.png` | Figure E1, the agentic arm and the focal sample |
| `outputs/tables/tableA2_ads_by_occupation.csv` | Table A2, the advertisements by occupation |

The file names carry the numbers the figures had while they were written; the
table above is the only place where the two numberings meet. The remaining
files under `outputs/` are the result tables the thesis draws its numbers from:
`outputs/study1_results/` for Study 1, `outputs/study2_run5/` for Study 2 and
`outputs/study2_run5/arm3/main/` for the language model. Data files with one row
per biography, advertisement or rating are not part of this repository, and
neither are the embedding caches.

## Order and dependencies

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
before `06b` and `07` are run. `06b` reads the tables of `06` and `07`, so it
comes last.

## Runtime

| Notebook | Machine | Roughly |
| --- | --- | --- |
| 01, 04c, 07 | CPU | minutes |
| 02 | CPU | 30 minutes, of which 26 are the name removal with spaCy NER |
| 03 | CPU | 34 minutes for the full pass over 393,977 biographies |
| 04 | CPU | under an hour; the two hurdle models are the slow part |
| 04b | CPU | 8 minutes for the two recounts |
| 06 | CPU | about 3 hours, of which 2.2 are the Arm 1 embeddings |
| 06a | Kaggle GPU T4 x2 | 23 minutes |
| 06b | CPU | 5 minutes |
| 07a | Kaggle GPU T4 x2 | 3.5 hours for 49,400 ratings |

## Data

* Biographies: `LabHC/bias_in_bios` on Hugging Face,
  <https://huggingface.co/datasets/LabHC/bias_in_bios>. Notebook 01 reads the
  three published parquet files directly over the network; the corpus is not
  part of this repository.
* Job advertisements: LinkedIn Job Postings (2023-2024) by Arsh Koneru on
  Kaggle, version 13,
  <https://www.kaggle.com/datasets/arshkon/linkedin-job-postings>. The file is
  expected at `data/job_descriptions/job_postings.csv` and is not part of this
  repository either.
* Dictionaries: the agency and communion dictionary of Pietraszkiewicz et al.
  (2019) in `data/lexicons/`, with the file of the published main version and
  the one of the union used for the robustness check.

Neither corpus is part of this repository. Notebook 01 downloads the
biographies from Hugging Face when it runs, so nothing has to be prepared for
Study 1. The advertisements have to be fetched by hand: download the Kaggle
dataset named above and put `job_postings.csv` at
`data/job_descriptions/job_postings.csv` before notebook 06 is run. Both
sources keep the terms of their own licences, which the two links state; this
repository licenses its code only, and redistributing either corpus is not
covered by it.

## What "Run 5" means

Study 2 was run five times. Run 5 is the fifth and final run and the only one
the thesis reports; the earlier runs changed the substitution protocol, the
content gate and the sample, and their outputs are not part of this repository.

The two tests the thesis treats as fixed in advance, T1 and T2, were written
down on 2026-10-05 and the dictionary they use on 2026-10-10, both before Run 5
was evaluated. They were not fixed before the data had been seen: the earlier
runs used the same biographies and the same advertisements, and it was those
runs that suggested H4. T1 and T2 are therefore fixed in advance of the final
evaluation, not of the study, and the thesis reports them as such.

The execution counters of `07a_llm_scoring_kaggle.ipynb` begin at 4 and run to
13 without a gap: the rating session had already executed three cells when the
first cell of the notebook was run. The run itself is complete in one session
(`llm_run_metadata.json`: 49,400 ratings, `session_hours_used` 3.497 of the 11
hours allowed).

## Versions

From the printed output of notebook 02, section 9:

```
python 3.13.9, spacy 3.8.14, en_core_web_lg 3.8.0, gender-guesser 0.4.0,
scikit-learn 1.6.1, pandas 2.2.3, numpy 2.1.3, pyarrow 19.0.0, scipy 1.15.3,
statsmodels 0.14.4, matplotlib 3.10.0, seaborn 0.13.2,
sentence-transformers 6.0.1, transformers 5.16.1, torch 2.14.0
```

The two Kaggle runs used the versions of their own image: notebook 06a
`sentence-transformers 5.7.0` with `torch 2.11.0+cu128`, and notebook 07a
`ollama 0.40.3` with the model `qwen3:8b`, digest `500a1f067a9f`. The pinned
set for a local run is `environment/requirements.txt`.

## Layout

```
README.md    this file
LICENSE      MIT, for the code of this repository
.gitignore   the corpora, the caches, the credentials
notebooks/   the eleven notebooks above, with their stored outputs
src/         the modules the notebooks import
config/      the substitution rules, the fixed terms and the occupation table
data/        the dictionaries; the two corpora are not included
outputs/     the figures and the result tables of the thesis
environment/ requirements.txt and a note on the environment
```
