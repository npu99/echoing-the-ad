"""study2_core — the shared functions of Study 2.

Reference implementation. The logic is tested on synthetic data
(test_study2_core.py); the tests against a real tokenizer and a real model run
on the machine that produces the results.

Contents
  1. Embedding long texts without truncation  pack_chunks, encode_long, jd_similarity
  2. Ranks                                    pool_pct_per_jd, indiv_cf_pct_per_jd,
                                              focal_pct_per_jd, delta_z_per_jd
                                              (+ percentile_ranks / individual_cf_ranks
                                               as in notebook 05, Run 3)
  3. Evaluation                               treatment_table, jd_language_test,
                                              strat_perm_p, loo_correlation
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, pearsonr, rankdata, spearmanr, wilcoxon

TOL = 1e-9  # a smaller |Δ| counts as "no change of rank" (float noise ~1e-14)


# ═════════════════════════════════════════════════════════════════════════════
# 1. Embedding long texts without truncation
# ═════════════════════════════════════════════════════════════════════════════
def _word_spans(text: str, tokenizer):
    """List of (start_char, end_char, n_tokens) per whitespace-separated word.

    Uses the offsets of a fast tokenizer; otherwise every word is tokenised on
    its own."""
    try:
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offs = [(s, e) for (s, e) in enc["offset_mapping"] if e > s]
    except Exception:
        offs = None
    spans = []
    if offs is not None:
        for s, e in offs:
            if spans and s <= spans[-1][1]:
                # The token adjoins the previous one, so it is the same word
                ps, pe, pn = spans[-1]
                spans[-1] = (ps, max(pe, e), pn + 1)
            else:
                spans.append((s, e, 1))
        return spans
    pos = 0
    for w in text.split():
        s = text.index(w, pos)
        e = s + len(w)
        pos = e
        spans.append((s, e, max(len(tokenizer.tokenize(w)), 1)))
    return spans


def pack_chunks(text: str, tokenizer, max_tokens: int, fits_tokens: int | None = None):
    """Split `text` at word boundaries into consecutive windows of at most
    `max_tokens` tokens each (special tokens not counted). The windows do not
    overlap and nothing is lost. A text of at most `fits_tokens` tokens
    (default: `max_tokens`) comes back unchanged as a single window.

    Returns a list of (chunk_text, n_tokens)."""
    spans = _word_spans(text, tokenizer)
    total = sum(n for _, _, n in spans)
    if total <= (max_tokens if fits_tokens is None else fits_tokens):
        return [(text, total)]
    # A single word that is too long on its own (a URL, say) is split by
    # characters, in proportion to its token count
    fixed = []
    for s, e, n in spans:
        if n <= max_tokens:
            fixed.append((s, e, n))
        else:
            parts = int(np.ceil(n / max_tokens))
            cuts = np.linspace(s, e, parts + 1).astype(int)
            for a, b in zip(cuts[:-1], cuts[1:]):
                if b > a:
                    fixed.append((int(a), int(b), int(np.ceil(n / parts))))
    chunks, cur_s, cur_e, cur_n = [], None, None, 0
    for s, e, n in fixed:
        if cur_s is not None and cur_n + n > max_tokens:
            chunks.append((text[cur_s:cur_e], cur_n))
            cur_s, cur_n = None, 0
        if cur_s is None:
            cur_s = s
        cur_e, cur_n = e, cur_n + n
    if cur_s is not None:
        chunks.append((text[cur_s:cur_e], cur_n))
    return chunks


def encode_long(texts, model, prefix: str = "", batch_size: int = 32, margin: int = 4):
    """Embed texts of any length: split into windows, embed every window,
    average them weighted by their token count, and normalise to unit length.

    For every text that the model would not truncate the result is identical to
    model.encode(prefix + text, normalize_embeddings=True). Only longer texts
    are split (window = the limit of the model minus `margin` tokens of
    headroom).

    Returns a dict:
      mean     (n, d)  document embedding (weighted mean, normalised)
      chunks   (m, d)  embeddings of all windows (normalised), in document order
      owner    (m,)    index of the document of each window (ascending)
      n_tokens (m,)    token count of each window
      n_chunks (n,)    windows per document
      window           the window size in tokens that was used
    """
    tok = model.tokenizer
    try:
        n_special = tok.num_special_tokens_to_add(pair=False)
    except Exception:
        n_special = 2
    n_prefix = len(tok(prefix, add_special_tokens=False)["input_ids"]) if prefix else 0
    fits = int(model.max_seq_length) - n_special - n_prefix
    window = fits - margin
    if window < 16:
        raise ValueError(f"window too small ({window} tokens)")

    chunk_texts, owner, n_tok = [], [], []
    for i, t in enumerate(texts):
        for c, n in pack_chunks(t if isinstance(t, str) else "", tok, window, fits_tokens=fits):
            chunk_texts.append(prefix + c)
            owner.append(i)
            n_tok.append(n)
    owner, n_tok = np.asarray(owner), np.asarray(n_tok, dtype=float)

    chunk_emb = np.asarray(model.encode(chunk_texts, normalize_embeddings=True,
                                        batch_size=batch_size, show_progress_bar=False))
    w = np.where(n_tok > 0, n_tok, 1.0)
    mean = np.zeros((len(texts), chunk_emb.shape[1]))
    np.add.at(mean, owner, chunk_emb * w[:, None])
    mean /= np.linalg.norm(mean, axis=1, keepdims=True)
    return {"mean": mean, "chunks": chunk_emb, "owner": owner, "n_tokens": n_tok,
            "n_chunks": np.bincount(owner, minlength=len(texts)), "window": window}


def jd_similarity(bio_emb: np.ndarray, jd: dict, how: str = "mean") -> np.ndarray:
    """Cosine similarity (n_bios, n_jds) between biography and advertisement
    embeddings.

    how = 'mean'  the whole advertisement (weighted mean of all windows)  [main]
          'max'   its best matching window (MaxP)                  [sensitivity]
          'first' its first window only (truncation, as in Run 3)   [comparison]

    For the ranking of the biographies against one advertisement, 'mean' is
    equivalent to the weighted mean of the similarities to its windows."""
    if how == "mean":
        return bio_emb @ jd["mean"].T
    starts = np.r_[0, np.cumsum(jd["n_chunks"])[:-1]]
    if how == "first":
        return bio_emb @ jd["chunks"][starts].T
    if how == "max":
        return np.maximum.reduceat(bio_emb @ jd["chunks"].T, starts, axis=1)
    raise ValueError(how)


# ═════════════════════════════════════════════════════════════════════════════
# 2. Ranks
# ═════════════════════════════════════════════════════════════════════════════
def pool_pct_per_jd(sim_pool: np.ndarray) -> np.ndarray:
    """(n, J) percentile rank of every biography of the pool against every
    advertisement (0 = last, 100 = first)."""
    n, J = sim_pool.shape
    out = np.empty((n, J))
    for j in range(J):
        ranks = np.argsort(np.argsort(-sim_pool[:, j]))
        out[:, j] = (n - 1 - ranks) / (n - 1) * 100
    return out


def indiv_cf_pct_per_jd(sim_pool: np.ndarray, sim_manip: np.ndarray) -> np.ndarray:
    """(n, J) percentile rank of biography i against every advertisement when
    only biography i is manipulated and the other n-1 biographies of the pool
    keep their original text (the individual counterfactual)."""
    n, J = sim_pool.shape
    out = np.empty((n, J))
    for j in range(J):
        oc, mc = sim_pool[:, j], sim_manip[:, j]
        n_above = n - np.searchsorted(np.sort(oc), mc, side="right") - (oc > mc).astype(int)
        out[:, j] = (n - 1 - n_above) / (n - 1) * 100
    return out


def focal_pct_per_jd(sim_pool: np.ndarray, sim_focal: np.ndarray) -> np.ndarray:
    """(n_focal, J) percentile rank of a biography that is not in the pool,
    against the n_pool original biographies of the pool: the share of the pool
    that it is not beaten by."""
    n_pool, J = sim_pool.shape
    out = np.empty((sim_focal.shape[0], J))
    for j in range(J):
        n_above = n_pool - np.searchsorted(np.sort(sim_pool[:, j]), sim_focal[:, j], side="right")
        out[:, j] = (n_pool - n_above) / n_pool * 100
    return out


def delta_z_per_jd(sim_pool: np.ndarray, sim_orig: np.ndarray, sim_manip: np.ndarray) -> np.ndarray:
    """(n, J) shift of the similarity in standard deviations of the pool, per
    advertisement. The continuous counterpart of the rank: it moves even when no
    competitor is overtaken."""
    sd = sim_pool.std(axis=0, ddof=1)
    return (sim_manip - sim_orig) / sd[None, :]


def percentile_ranks(sim_matrix: np.ndarray) -> np.ndarray:
    """As in notebook 05 (cell 9cc21a68)."""
    return pool_pct_per_jd(sim_matrix).mean(axis=1)


def individual_cf_ranks(sim_orig: np.ndarray, sim_manip: np.ndarray) -> np.ndarray:
    """As in notebook 05 (cell 34166adb)."""
    return indiv_cf_pct_per_jd(sim_orig, sim_manip).mean(axis=1)


# ═════════════════════════════════════════════════════════════════════════════
# 3. Evaluation
# ═════════════════════════════════════════════════════════════════════════════
def _safe_wilcoxon(x):
    x = np.asarray(x, dtype=float)
    x = x[np.abs(x) > TOL]
    if len(x) < 10:
        return np.nan
    try:
        return float(wilcoxon(x, alternative="two-sided")[1])
    except ValueError:
        return np.nan


def treatment_table(df: pd.DataFrame, group_cols, delta_col: str = "delta",
                    nsubs_col: str = "n_subs") -> pd.DataFrame:
    """Per group: how many biographies the manipulation changed, in which
    direction the rank moves, how large the effect is among the changed
    biographies, and whether it grows with the number of substitutions."""
    rows = []
    for key, sub in df.groupby(list(group_cols), sort=False):
        key = key if isinstance(key, tuple) else (key,)
        d = sub[delta_col].to_numpy(float)
        ns = sub[nsubs_col].to_numpy(float)
        tr = ns >= 1
        row = dict(zip(group_cols, key))
        row.update({
            "n": len(sub), "share_unchanged": float((~tr).mean()),
            "share_delta_zero": float((np.abs(d) <= TOL).mean()),
            "share_delta_neg": float((d < -TOL).mean()),
            "share_delta_pos": float((d > TOL).mean()),
            "mean_all": float(d.mean()), "median_all": float(np.median(d)),
            "p_wilcoxon_all": _safe_wilcoxon(d),
            "n_treated": int(tr.sum()),
            "mean_treated": float(d[tr].mean()) if tr.any() else np.nan,
            "median_treated": float(np.median(d[tr])) if tr.any() else np.nan,
            "p_wilcoxon_treated": _safe_wilcoxon(d[tr]),
        })
        if tr.sum() >= 10 and np.std(ns[tr]) > 0:
            row["rho_dose_abs"], row["p_dose_abs"] = spearmanr(ns[tr], np.abs(d[tr]))
            row["rho_dose_signed"], row["p_dose_signed"] = spearmanr(ns[tr], d[tr])
        rows.append(row)
    return pd.DataFrame(rows)


def strat_perm_p(strata, x, y, lab, n_perm: int = 5000, seed: int = 42):
    """Two-sided permutation p-values for (a) Spearman(x, y) and (b) the
    difference of the mean of y between lab == +1 and lab == -1. x and lab are
    permuted together within the strata (unit = advertisement, stratum =
    occupation).

    Returns rho, p_rho, diff, p_diff."""
    strata, x, y, lab = map(np.asarray, (strata, x, y, lab))
    rng = np.random.default_rng(seed)
    rx, ry = rankdata(x), rankdata(y)
    ry_c = ry - ry.mean()

    def stats(rx_, lab_):
        rc = rx_ - rx_.mean()
        rho = float(rc @ ry_c / np.sqrt((rc @ rc) * (ry_c @ ry_c)))
        hi, lo = lab_ == 1, lab_ == -1
        diff = float(y[hi].mean() - y[lo].mean()) if hi.any() and lo.any() else np.nan
        return rho, diff

    rho_obs, diff_obs = stats(rx, lab)
    idx_by = [np.where(strata == s)[0] for s in pd.unique(strata)]
    ge_rho = ge_diff = 0
    for _ in range(n_perm):
        perm = np.arange(len(x))
        for idx in idx_by:
            perm[idx] = rng.permutation(idx)
        r_, d_ = stats(rx[perm], lab[perm])
        ge_rho += abs(r_) >= abs(rho_obs) - 1e-12
        ge_diff += (abs(d_) >= abs(diff_obs) - 1e-12) if np.isfinite(diff_obs) else 0
    p_diff = (1 + ge_diff) / (n_perm + 1) if np.isfinite(diff_obs) else np.nan
    return rho_obs, (1 + ge_rho) / (n_perm + 1), diff_obs, p_diff


def jd_language_test(delta_j: dict, dens: dict, n_perm: int = 5000, seed: int = 42):
    """Does the effect of the manipulation depend on the wording of the
    advertisement? Unit = advertisement.

    delta_j[occ]  (n_jds,)  mean effect against advertisement j, averaged over
                            the changed biographies (Δ rank or Δz)
    dens[occ]     (n_jds,)  density of the advertisement in the dimension under
                            consideration (per cent of its words)

    Returns (a table per occupation, a dict for the pooled test). For the
    matching dimension the hypothesis expects rho < 0 and (high − low) < 0."""
    rows, S, X, Y, L = [], [], [], [], []
    for occ in delta_j:
        dj, de = np.asarray(delta_j[occ], float), np.asarray(dens[occ], float)
        if len(dj) < 10 or np.std(de) == 0:
            continue
        rho, p = spearmanr(de, dj)
        q1, q2 = np.quantile(de, [1 / 3, 2 / 3])
        lab = np.where(de >= q2, 1, np.where(de <= q1, -1, 0)) if q2 > q1 else np.zeros(len(de), int)
        hi, lo = lab == 1, lab == -1
        row = {"occ": occ, "n_jds": len(de), "rho": rho, "p_rho": p}
        if hi.sum() >= 10 and lo.sum() >= 10:
            row.update({"n_jd_hi": int(hi.sum()), "n_jd_lo": int(lo.sum()),
                        "delta_hi": float(dj[hi].mean()), "delta_lo": float(dj[lo].mean()),
                        "diff_hi_lo": float(dj[hi].mean() - dj[lo].mean()),
                        "p_diff": float(mannwhitneyu(dj[hi], dj[lo], alternative="two-sided")[1])})
        rows.append(row)
        S.append(np.full(len(de), occ)); X.append(de - de.mean()); Y.append(dj - dj.mean()); L.append(lab)
    if not rows:
        return pd.DataFrame(), {}
    S, X, Y, L = map(np.concatenate, (S, X, Y, L))
    rho, p_rho, diff, p_diff = strat_perm_p(S, X, Y, L, n_perm=n_perm, seed=seed)
    return pd.DataFrame(rows), {"n_jds": len(X), "n_occ": len(rows), "rho": rho, "p_rho": p_rho,
                                "diff_hi_lo": diff, "p_diff": p_diff}


def loo_correlation(x, y, labels) -> pd.DataFrame:
    """Pearson and Spearman over all points, plus leave-one-out per point."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rows = [{"excluded": "(none)", "n": len(x),
             "pearson_r": pearsonr(x, y)[0], "pearson_p": pearsonr(x, y)[1],
             "spearman_r": spearmanr(x, y)[0], "spearman_p": spearmanr(x, y)[1]}]
    for i, lab in enumerate(labels):
        m = np.arange(len(x)) != i
        rows.append({"excluded": lab, "n": int(m.sum()),
                     "pearson_r": pearsonr(x[m], y[m])[0], "pearson_p": pearsonr(x[m], y[m])[1],
                     "spearman_r": spearmanr(x[m], y[m])[0], "spearman_p": spearmanr(x[m], y[m])[1]})
    return pd.DataFrame(rows)
