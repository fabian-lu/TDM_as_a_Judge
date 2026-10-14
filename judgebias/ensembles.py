"""Ensembles of typed decision models, computed from the stored answers (no API calls).

    from judgebias import ensembles as E
    M = E.Matrix("calm")                          # P(A) of every model for the requests all models answered
    E.evaluate(M, M.combine(range(8), "vote"))    # excess per test, mean excess, accuracy
    E.subsets(M)                                  # all 255 subsets, chosen on one half of the items, tested on the other
    E.both_orders()                               # one model asked in both orders, on the items of the position test

Combinations of the models' answers to one request:
    mean     mean of P(A)
    median   median of P(A)
    logit    mean of log-odds of P(A)
    vote     majority of the models' verdicts (a tie gives no verdict)
A combined answer is then read like a single model's: verdicts, the 50/50 rule, RR, CR, excess.
"""
import hashlib
import itertools

import numpy as np
import pandas as pd

from . import authority, design, metrics, run

COMBINERS = ("mean", "median", "logit", "vote")


def _half(item):
    """0 or 1, a fixed split of the items into two halves."""
    return int(hashlib.md5(item.encode()).hexdigest(), 16) % 2


def _verdict(p, r1_is_a):
    """+1 if the answer favours R1, -1 if R2, 0 for a 50/50 answer."""
    v = np.where(np.abs(p - 0.5) < metrics.TIE, 0, np.where(p > 0.5, 1, -1))
    return v * np.where(r1_is_a, 1, -1)


class Matrix:
    """P(A) of every model (rows) for every request that all models answered (columns), with the indices
    that link each perturbed prompt to its original and its repetition."""

    def __init__(self, fmt="calm", models=None):
        rows = design.build_design(citations=authority.load_citations())
        by_id = {r["id"]: r for r in rows}
        self.models = models or [m for m in run.MODELS if "mercury" not in m]
        self.names = [m.split("/")[1].replace("-experimental", "") for m in self.models]
        recs = [{(r["id"], r["repeat"]): r for r in run.load_records(m, fmt)} for m in self.models]
        keys = [k for k in recs[0] if all(k in R and R[k]["ok"] for R in recs)]
        kidx = {k: i for i, k in enumerate(keys)}
        self.P = np.array([[R[k]["p_A"] for k in keys] for R in recs])
        pert = [r for r in rows if r["bias"] != "original" and (r["id"], 0) in kidx
                and (r["original"], 0) in kidx and (r["original"], 1) in kidx]
        tests = metrics.separate_additions(pd.DataFrame(
            [{"bias": r["bias"], "variant": r["variant"]} for r in pert])).bias.values
        self.tests = tests
        self.i_hat = np.array([kidx[(r["id"], 0)] for r in pert])
        self.i_y = np.array([kidx[(r["original"], 0)] for r in pert])
        self.i_rep = np.array([kidx[(r["original"], 1)] for r in pert])
        self.lab_hat = np.array([r["r1_label"] == "A" for r in pert])
        self.lab_orig = np.array([by_id[r["original"]]["r1_label"] == "A" for r in pert])
        self.half = np.array([_half(r["item"]) for r in pert])
        orig = [r for r in rows if r["bias"] == "original" and (r["id"], 0) in kidx]
        self.o_idx = np.array([kidx[(r["id"], 0)] for r in orig])
        self.o_lab = np.array([r["r1_label"] == "A" for r in orig])
        self.o_ds = np.array([r["dataset"] for r in orig])
        self.o_half = np.array([_half(r["item"]) for r in orig])

    def combine(self, subset, how="mean"):
        """The combined P(A) of the models in `subset` (indices) for every request."""
        X = self.P[list(subset)]
        if how == "mean":
            return X.mean(axis=0)
        if how == "median":
            return np.median(X, axis=0)
        if how == "logit":
            L = np.log(np.clip(X, 1e-6, 1 - 1e-6) / np.clip(1 - X, 1e-6, 1 - 1e-6))
            return 1 / (1 + np.exp(-L.mean(axis=0)))
        if how == "vote":                    # 0.99 / 0.01 / 0.5 from the majority of verdicts
            margin = (X > 0.5).sum(axis=0) - (X < 0.5).sum(axis=0)
            return 0.5 + 0.49 * np.sign(margin)
        raise ValueError(how)


def evaluate(M, p, half=None):
    """Excess per test (for fallacy oversight RR + CR - 1), mean excess over all tests and accuracy on
    the original alignment prompts, for a combined answer `p`. `half` restricts to one half of the items."""
    y, yr, yh = _verdict(p[M.i_y], M.lab_orig), _verdict(p[M.i_rep], M.lab_orig), _verdict(p[M.i_hat], M.lab_hat)
    ok = (y != 0) & (yr != 0) & (yh != 0)
    if half is not None:
        ok &= M.half == half
    out = {}
    for t in np.unique(M.tests):
        m = ok & (M.tests == t)
        rr, cr = np.mean(yh[m] == y[m]), np.mean(yr[m] == y[m])
        out[t] = rr + cr - 1 if t == "fallacy" else cr - rr         # fallacy: unchanged verdicts beyond chance
    out["mean excess"] = float(np.mean(list(out.values())))
    v = _verdict(p[M.o_idx], M.o_lab)
    m = (M.o_ds == "alignment") & (v != 0)
    if half is not None:
        m &= M.o_half == half
    out["accuracy"] = float(np.mean(v[m] == 1))
    return out


def judges(M, cost=None):
    """One row per single model and per combination of all models, with cost (sum) and latency (slowest model)
    when `cost` (tables/time_and_cost.csv, CALM format, with the column `model` renamed to `judge`) is given."""
    out = []
    for i, name in enumerate(M.names):
        out.append({"judge": name, "kind": "single", **evaluate(M, M.P[i])})
    for how in COMBINERS:
        out.append({"judge": f"all {len(M.names)}, {how}", "kind": "ensemble",
                    **evaluate(M, M.combine(range(len(M.names)), how))})
    df = pd.DataFrame(out)
    if cost is not None:
        c = cost.set_index("judge")
        single = df.kind == "single"
        df.loc[single, "usd_per_1000"] = df.loc[single, "judge"].map(c.usd_per_1000)
        df.loc[single, "latency_s"] = df.loc[single, "judge"].map(c.median_latency_s)
        df.loc[~single, "usd_per_1000"] = c.loc[M.names, "usd_per_1000"].sum()
        df.loc[~single, "latency_s"] = c.loc[M.names, "median_latency_s"].max()
    return df


def subsets(M, how="mean"):
    """Every non-empty subset of the models: mean excess and accuracy on each half of the items."""
    out = []
    n = len(M.names)
    for k in range(1, n + 1):
        for s in itertools.combinations(range(n), k):
            p = M.combine(s, how)
            a, b = evaluate(M, p, half=0), evaluate(M, p, half=1)
            out.append({"subset": "+".join(M.names[i] for i in s), "k": k, "excess_A": a["mean excess"],
                        "excess_B": b["mean excess"], "accuracy_A": a["accuracy"], "accuracy_B": b["accuracy"]})
    return pd.DataFrame(out)


def both_orders(fmt="calm", n_boot=10_000, seed=0):
    """Per model, on the items of the position test: accuracy when the pair is judged in one order, and when the
    probabilities of both orders are averaged (which removes position bias by construction), with a 95% bootstrap
    interval over items for the change. A 50/50 answer has no verdict and is left out, as everywhere in the paper,
    so each accuracy counts the items with a verdict."""
    rows = design.build_design(citations=authority.load_citations())
    by_id = {r["id"]: r for r in rows}
    rng = np.random.default_rng(seed)
    out = []
    for m in [x for x in run.MODELS if "mercury" not in x]:
        R = {(r["id"], r["repeat"]): r for r in run.load_records(m, fmt)}
        one, both, agree = [], [], []                       # per item: (has verdict, correct)
        for r in (x for x in rows if x["bias"] == "position"):
            a, b = R.get((r["original"], 0)), R.get((r["id"], 0))
            if not (a and b and a["ok"] and b["ok"]):
                continue
            p1 = metrics._prob(by_id[r["original"]], a, "R1")      # P(R1) with the original order
            p2 = metrics._prob(r, b, "R1")                          # P(R1) with the two responses swapped
            v1, v2, vb = (None if abs(p - 0.5) < metrics.TIE else p > 0.5 for p in (p1, p2, (p1 + p2) / 2))
            one.append((v1 is not None, bool(v1)))
            both.append((vb is not None, bool(vb)))
            if v1 is not None and v2 is not None:
                agree.append(v1 == v2)
        one, both = np.array(one, float), np.array(both, float)
        acc = lambda x, w=None: (x[:, 1] * x[:, 0]).sum() / x[:, 0].sum() if w is None else (w @ (x[:, 1] * x[:, 0])) / (w @ x[:, 0])
        w = rng.multinomial(len(one), np.full(len(one), 1.0 / len(one)), size=n_boot)
        change = acc(both, w) - acc(one, w)
        out.append({"model": m.split("/")[1].replace("-experimental", ""), "items": len(one),
                    "accuracy_one_order": acc(one), "accuracy_both_orders": acc(both),
                    "change": acc(both) - acc(one), "change_lo": np.percentile(change, 2.5),
                    "change_hi": np.percentile(change, 97.5), "orders_agree": np.mean(agree)})
    return pd.DataFrame(out)
