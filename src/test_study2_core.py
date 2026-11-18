"""Tests for study2_core. Call:  python test_study2_core.py   (or pytest)

Part 1 (synthetic) runs anywhere and has to pass completely.
Part 2 (real model) runs only where sentence-transformers is installed and the
model can be loaded; it runs on the machine that produces the results.
"""
import hashlib
import re

import numpy as np
import pandas as pd

import study2_core as core


# ── Stubs: a tokenizer with offsets, a model that truncates ─────────────────
class StubTok:
    """Whitespace tokenizer; a word of more than 6 characters falls into
    pieces of three, as subwords do."""
    def __init__(self, fast=True):
        self.fast = fast

    def _pieces(self, text):
        out = []
        for m in re.finditer(r"\S+", text):
            w, s = m.group(), m.start()
            if len(w) <= 6:
                out.append((s, s + len(w)))
            else:
                out += [(s + k, min(s + k + 3, s + len(w))) for k in range(0, len(w), 3)]
        return out

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False, **kw):
        if return_offsets_mapping and not self.fast:
            raise NotImplementedError
        offs = self._pieces(text)
        out = {"input_ids": list(range(len(offs) + (2 if add_special_tokens else 0)))}
        if return_offsets_mapping:
            out["offset_mapping"] = ([(0, 0)] + offs + [(0, 0)]) if add_special_tokens else offs
        return out

    def tokenize(self, text):
        return [text[s:e] for s, e in self._pieces(text)]

    def num_special_tokens_to_add(self, pair=False):
        return 2


class StubModel:
    """Embedding = normierte Summe gehashter Token-Vektoren; schneidet wie ein echtes
    Modell nach max_seq_length Tokens (inkl. 2 Spezial-Tokens) ab."""
    def __init__(self, max_seq_length=40, dim=64, fast=True):
        self.max_seq_length, self.dim, self.tokenizer = max_seq_length, dim, StubTok(fast)
        self.seen_token_counts = []

    def _vec(self, piece):
        seed = int(hashlib.md5(piece.encode()).hexdigest()[:8], 16)
        return np.random.default_rng(seed).normal(size=self.dim)

    def encode(self, texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False):
        out = []
        for t in texts:
            pieces = self.tokenizer.tokenize(t)
            self.seen_token_counts.append(len(pieces) + 2)
            pieces = pieces[: self.max_seq_length - 2]
            v = np.sum([self._vec(p) for p in pieces], axis=0) if pieces else np.ones(self.dim)
            out.append(v / np.linalg.norm(v))
        return np.array(out)


def _text(rng, n_words):
    vocab = ["care", "lead", "team", "patient", "responsibilities", "analysis", "support",
             "drive", "communication", "we", "are", "an", "equal", "opportunity", "employer",
             "https://example.com/a/very/long/url/that/never/ends/and/keeps/going/on/forever/and/ever"]
    return " ".join(rng.choice(vocab[:-1], size=n_words)) + ("  \n " + vocab[-1] if n_words % 7 == 0 else "")


# ── Part 1: the synthetic tests ─────────────────────────────────────────────
def test_pack_chunks_covers_text_and_respects_limit():
    rng = np.random.default_rng(0)
    for fast in (True, False):
        tok = StubTok(fast)
        for n_words in [0, 1, 5, 30, 77, 200, 701]:
            text = _text(rng, n_words)
            for limit in [16, 34, 100]:
                chunks = core.pack_chunks(text, tok, limit)
                assert all(len(tok.tokenize(c)) <= limit for c, _ in chunks), (n_words, limit)
                joined = "".join(c for c, _ in chunks)
                assert re.sub(r"\s+", "", joined) == re.sub(r"\s+", "", text), "text lost or duplicated"
                if len(tok.tokenize(text)) <= limit:
                    assert len(chunks) == 1 and chunks[0][0] == text
                if fast:
                    assert sum(n for _, n in chunks) >= len(tok.tokenize(text))


def test_encode_long_equals_plain_encode_for_short_texts_and_differs_for_long():
    rng = np.random.default_rng(1)
    model = StubModel(max_seq_length=40)
    long_ = [_text(rng, n) for n in [120, 300]]
    for prefix in ["", "query: "]:
        n_pre = len(model.tokenizer.tokenize(prefix))
        # three texts that the model does not yet truncate, the last one exactly full
        short = []
        for target in [5, 20, model.max_seq_length - 2 - n_pre]:
            words = []
            while len(model.tokenizer.tokenize(" ".join(words + ["care"]))) <= target:
                words.append("care")
            short.append(" ".join(words))
        assert len(model.tokenizer.tokenize(short[-1])) == model.max_seq_length - 2 - n_pre
        model.seen_token_counts = []
        out = core.encode_long(short + long_, model, prefix=prefix)
        assert max(model.seen_token_counts) <= model.max_seq_length, "the model truncated one of the windows"
        plain = model.encode([prefix + t for t in short + long_])
        assert np.abs(out["mean"][:3] - plain[:3]).max() < 1e-12, "a text that is not truncated has to come back identical"
        assert np.abs(out["mean"][3:] - plain[3:]).max() > 1e-3, "a long text: truncation must not equal the whole text"
        assert np.allclose(np.linalg.norm(out["mean"], axis=1), 1.0)
        assert (out["n_chunks"][:3] == 1).all() and (out["n_chunks"][3:] > 1).all()
        assert (np.diff(out["owner"]) >= 0).all()


def test_jd_similarity_modes():
    rng = np.random.default_rng(2)
    model = StubModel(max_seq_length=40)
    jds = [_text(rng, n) for n in [10, 150, 90, 400]]
    bios = [_text(rng, n) for n in [12, 20, 9, 25, 14]]
    jd = core.encode_long(jds, model)
    B = core.encode_long(bios, model)["mean"]
    S_mean, S_max, S_first = (core.jd_similarity(B, jd, h) for h in ("mean", "max", "first"))
    starts = np.r_[0, np.cumsum(jd["n_chunks"])[:-1]]
    for j in range(len(jds)):
        ch = jd["chunks"][jd["owner"] == j]
        w = jd["n_tokens"][jd["owner"] == j]
        assert np.allclose(S_max[:, j], (B @ ch.T).max(axis=1))
        assert np.allclose(S_first[:, j], B @ ch[0])
        # 'mean' ranks the biographies like the weighted mean of the window similarities
        avg = (B @ ch.T) @ w / w.sum()
        assert (np.argsort(S_mean[:, j]) == np.argsort(avg)).all()
    assert np.allclose(S_mean[:, 0], S_max[:, 0]) and np.allclose(S_mean[:, 0], S_first[:, 0])  # one window


def _nb05_percentile_ranks(sim_matrix):          # verbatim from notebook 05, cell 9cc21a68
    n_bios, n_jds = sim_matrix.shape
    pct = np.zeros(n_bios)
    for j in range(n_jds):
        col = sim_matrix[:, j]
        ranks = np.argsort(np.argsort(-col))
        pct += (n_bios - 1 - ranks) / (n_bios - 1) * 100
    return pct / n_jds


def _nb05_individual_cf_ranks(sim_orig, sim_manip):   # verbatim from notebook 05, cell 34166adb
    n_bios, n_jds = sim_orig.shape
    rank_sum = np.zeros(n_bios)
    for j in range(n_jds):
        orig_col, manip_col = sim_orig[:, j], sim_manip[:, j]
        sorted_orig = np.sort(orig_col)
        n_above_tot = n_bios - np.searchsorted(sorted_orig, manip_col, side='right')
        n_above_oth = n_above_tot - (orig_col > manip_col).astype(int)
        rank_sum += (n_bios - 1 - n_above_oth) / (n_bios - 1) * 100
    return rank_sum / n_jds


def test_ranks_reproduce_run3_functions_and_brute_force():
    rng = np.random.default_rng(3)
    for rounding in (None, 2):                       # two decimals -> many ties
        so = rng.normal(size=(40, 25)); sm = so + rng.normal(scale=0.3, size=so.shape)
        sm[:8] = so[:8]                              # the unchanged biographies
        if rounding is not None:
            so, sm = so.round(rounding), sm.round(rounding)
        assert np.allclose(core.percentile_ranks(so), _nb05_percentile_ranks(so), atol=1e-10)
        assert np.allclose(core.individual_cf_ranks(so, sm), _nb05_individual_cf_ranks(so, sm), atol=1e-10)
        # Brute force: the pool in which only biography i is manipulated
        n, J = so.shape
        bf = np.zeros((n, J))
        for i in range(n):
            for j in range(J):
                others = np.delete(so[:, j], i)
                bf[i, j] = (n - 1 - (others > sm[i, j]).sum()) / (n - 1) * 100
        assert np.allclose(core.indiv_cf_pct_per_jd(so, sm), bf)
        if rounding is None:                         # without ties: unchanged => Δ = 0
            d = core.individual_cf_ranks(so, sm) - core.percentile_ranks(so)
            assert np.abs(d[:8]).max() < 1e-9
        # The focal biographies, which are not in the pool
        sf = rng.normal(size=(7, J)) if rounding is None else rng.normal(size=(7, J)).round(rounding)
        bf_f = np.array([[(n - (so[:, j] > sf[i, j]).sum()) / n * 100 for j in range(J)] for i in range(7)])
        assert np.allclose(core.focal_pct_per_jd(so, sf), bf_f)
        assert np.allclose(core.delta_z_per_jd(so, so, sm), (sm - so) / so.std(axis=0, ddof=1))


def _planted(effect, seed, n_occ=4, n_jds=90, n_bios=60):
    rng = np.random.default_rng(seed)
    delta_j, dens = {}, {}
    for k in range(n_occ):
        de = rng.uniform(1, 10, n_jds)
        delta_j[f"occ{k}"] = -effect * 0.2 * de + rng.normal(scale=0.8, size=n_jds) + k
        dens[f"occ{k}"] = de + 3 * k
    return delta_j, dens


def test_jd_language_test_power_and_calibration():
    tab, pool = core.jd_language_test(*_planted(1.0, 0), n_perm=999)
    assert pool["rho"] < -0.3 and pool["p_rho"] <= 0.002 and pool["diff_hi_lo"] < 0 and pool["p_diff"] <= 0.002
    assert (tab["rho"] < 0).all()
    ps_rho, ps_diff = [], []
    for s in range(200):
        _, pool = core.jd_language_test(*_planted(0.0, 100 + s), n_perm=199, seed=s)
        ps_rho.append(pool["p_rho"]); ps_diff.append(pool["p_diff"])
    fp_rho, fp_diff = np.mean(np.array(ps_rho) < 0.05), np.mean(np.array(ps_diff) < 0.05)
    print(f"    false-positive rate under the null (200 runs): rho {fp_rho:.3f}, high-low {fp_diff:.3f} (nominal 0.05)")
    assert fp_rho < 0.10 and fp_diff < 0.10


def test_treatment_table_and_loo():
    rng = np.random.default_rng(5)
    n = 300
    ns = rng.poisson(1.2, n)
    d = np.where(ns > 0, -0.5 * ns + rng.normal(size=n), 0.0)
    d[ns > 0][:3] = 1e-14
    df = pd.DataFrame({"occ": rng.choice(["a", "b"], n), "arm": "communion", "delta": d, "n_subs": ns})
    t = core.treatment_table(df, ["occ", "arm"])
    assert np.allclose(t["share_delta_zero"] + t["share_delta_neg"] + t["share_delta_pos"], 1.0)
    assert (t["share_delta_zero"] >= t["share_unchanged"] - 1e-12).all()
    assert (t["mean_treated"] < 0).all() and (t["rho_dose_signed"] < 0).all()
    x = np.array([1.05, .99, .90, .83, .65, .53, .42, .27, .19, .07, .04, -.04, -.15])   # Run 3, Section 9k
    y = np.array([-.2032, -1.5471, .8112, -.6382, .0996, .9092, .3086, .5799, .3630, -.0777, .4606, 1.4968, .5381])
    loo = core.loo_correlation(x, y, list("abcdefghijklm"))
    assert abs(loo.loc[0, "pearson_r"] - (-0.569)) < 0.002 and abs(loo.loc[0, "pearson_p"] - 0.0423) < 0.001
    assert len(loo) == 14


# ── Part 2: the real model ──────────────────────────────────────────────────
def real_model_checks(model_name="sentence-transformers/all-mpnet-base-v2"):
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
    except Exception as e:                                   # noqa: BLE001
        print(f"  SKIPPED (the model does not load: {type(e).__name__}: {str(e)[:80]})")
        return None
    rng = np.random.default_rng(7)
    short = ["She counsels clients on employment matters and leads a team of five.",
             "Registered nurse with ten years of experience in intensive care."]
    long_ = [_text(rng, 900), _text(rng, 1500)]
    out = core.encode_long(short + long_, model)
    plain = model.encode(short + long_, normalize_embeddings=True)
    dev_short = float(np.abs(out["mean"][:2] - plain[:2]).max())
    tok_counts = [len(model.tokenizer(c)["input_ids"]) for t in long_
                  for c, _ in core.pack_chunks(t, model.tokenizer, out["window"])]
    print(f"  model {model_name}: max_seq_length={model.max_seq_length}, window={out['window']}")
    print(f"  short texts: max |encode_long - encode| = {dev_short:.2e}   (target < 1e-5)")
    print(f"  long texts: windows per text = {out['n_chunks'][2:].tolist()}, "
          f"largest window including special tokens = {max(tok_counts)} (target <= {model.max_seq_length})")
    ok = dev_short < 1e-5 and max(tok_counts) <= model.max_seq_length and (out["n_chunks"][2:] > 1).all()
    print("  result:", "OK" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  OK  {t.__name__}")
    print(f"\nPart 1: {len(tests)} synthetic tests passed.\n\nPart 2: the real model")
    real_model_checks()
