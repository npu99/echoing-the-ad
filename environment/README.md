# Python environment

The notebooks run in the Anaconda environment that backs the Jupyter kernel named
`base`:

```
<your Python>\anaconda3\python.exe   (Python 3.13.9)
```

This is the environment that produced every result in `outputs/`. A second
interpreter exists on the same machine (`C:\Python310`, Python 3.10.5). It is **not**
used by the notebooks; a bare `python` on the command line may resolve to it, which is
a known source of confusion.

## Versions

`requirements.txt` lists the packages that the analysis depends on, pinned to the
versions verified on 2026-10-10.

The spaCy model is installed as a package and is not on PyPI:

```
python -m spacy download en_core_web_lg      # 3.8.0
```

## Why these versions are pinned

On 2026-10-10 the dictionary switch from A ∪ B to A was validated by recounting all
393,977 biographies and comparing against the archived feature table
(`notebooks/archive/validation_lexicon_switch_2026-10.ipynb`, point 2). The recount
reproduced `agency_count`, `communion_count`, `achievement_verb_rate` and `word_count`
with zero deviations.

Tokenisation decides those counts, so the spaCy and model versions are part of the
result. The check passed with spaCy 3.8.14 (this environment) and, independently, with
spaCy 3.8.16 under the other interpreter — both against the same model, `en_core_web_lg`
3.8.0. Tokenisation is therefore stable across those spaCy versions for this corpus, but
the model version is the one to keep fixed: F2 and F4 read the tagger and parser output
(`token.pos_`, `token.dep_`), which a different model version can change.

## Note on OpenMP

Running the notebooks' code through a plain script in this environment can abort with
`OMP: Error #15` (two OpenMP runtimes in one process, from MKL-backed NumPy and
spaCy's `thinc`/`blis`). The Jupyter kernel is not affected. Run the notebooks in
Jupyter rather than as scripts. Do not set `KMP_DUPLICATE_LIB_OK=TRUE` for runs whose
numbers are used: the workaround is documented as possibly producing wrong results
silently.

## The two GPU runs

Arm 2 of Study 2 (the embeddings of the search encoder) and Arm 3 (the ratings of
the screening model) do not run on the local machine. They run on Kaggle with the
accelerator **GPU T4 x2** and Internet on, each as Save and Run All, and their
notebooks are `06a_arm2_embeddings_kaggle.ipynb` and
`07a_llm_scoring_kaggle.ipynb`. The versions of that session are the ones recorded
in the metadata file each run writes, not the local ones.

| | Arm 2, embeddings | Arm 3, ratings (main run) |
|---|---|---|
| model | `Qwen/Qwen3-Embedding-0.6B` | `qwen3:8b` under Ollama 0.40.3 |
| identity of the weights | revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` | digest `500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41` |
| parameters | 595,776,512, loaded as bfloat16, stored as float32 | 8B, served by Ollama |
| what it produces | 1024 dimensions, normalised, 6,221 biographies and 2,975 advertisements, no text truncated at `max_seq_length` 3,294 | 49,400 pairs, 105 of them without a value |
| decoding | not applicable | temperature 0, seed 42, `num_predict` 1, `num_ctx` 4096, 20 top logprobs |
| runtime of the session | about 10 min | 3.50 h (the pilot of 6,888 pairs: 0.62 h) |
| torch | 2.11.0+cu128 | not applicable |
| transformers / sentence-transformers | 5.16.1 / 5.7.0 | not applicable |

The local environment above produced everything else, including Arm 1 of Study 2
(`sentence-transformers/all-mpnet-base-v2`, 384 tokens per window) and all tables
and figures of the two studies.
