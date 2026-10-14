"""The metrics and statistics of Section 3 of the paper.

    df  = metrics.table(rows, records)          # one line per perturbed prompt
    res = metrics.summarize(df, by=["bias"])     # RR, CR, accuracy, shift, noise floor, CIs, McNemar

For every perturbed prompt we need three answers from the same judge:
    y      verdict on the original prompt
    y_rep  verdict on the original prompt sent a second time
    y_hat  verdict on the perturbed prompt
Verdicts are stored as the response the judge picked, "R1" (the better one) or "R2", not as
"A"/"B". In the position test the responses swap places, so comparing responses (not
letters) gives CALM's rule that the same response must win in both orders.
The probabilities p, p_rep and p_hat are the probabilities of the response the perturbation
targets (column `target`), in the same three prompts. A 50/50 answer has no verdict (see
`_picked`) but keeps its probability, so it counts for the shifts and not for RR and CR.

Fallacy oversight is the one test in which the verdict should change; `summarize` gives its excess as
RR + CR - 1, the unchanged verdicts beyond chance, so that `excess` is the bias beyond chance in every test.

Is there a bias? We compare how often the verdict changes under the perturbation (1 - RR)
with how often it changes under repetition (1 - CR), on the same items: the excess of changed
verdicts, CR - RR. If the perturbation had no effect, a perturbed prompt would be just
another sending of the item, so it should not matter which of an item's prompts plays the
repetition. The permutation test (`_permutation_p`) reassigns that role at random within
every item and asks how often the excess is as large as observed. With one perturbed prompt
per item this is McNemar's exact test (up to Monte Carlo error). Intervals come from a
bootstrap that resamples items, because all prompts of an item share its original.
"""
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

from .design import NEUTRAL_GROUP


TIE = 1e-5     # an answer with |P(A) - 0.5| below this is 50/50 and has no verdict


ADDED = {"authority": "authority, better response", "diversity": "diversity, neutral note",
         "verbosity": "verbosity, better response"}


def separate_additions(df):
    """Our three additions to CALM's versions as tests of their own, so that every bias pools exactly
    CALM's versions (as CALM's Table 4 does): the citation on the better response (R1), the neutral identity note,
    and the padding of the better response. Identity notes count on both responses, since CALM does not say which."""
    df = df.copy()
    df.loc[(df.bias == "authority") & df.variant.str.endswith("-on-R1"), "bias"] = ADDED["authority"]
    df.loc[(df.bias == "diversity") & df.variant.str.startswith(NEUTRAL_GROUP), "bias"] = ADDED["diversity"]
    df.loc[(df.bias == "verbosity") & (df.variant == "R1-padded"), "bias"] = ADDED["verbosity"]
    return df


def _picked(row, rec):
    """The response a judge picked in one prompt: "R1", "R2", or None if it failed or answered 50/50.

    On a 50/50 answer the API still returns a choice, and for most models it is always "A".
    Counting it would turn an undecided answer into a preference for whichever response is
    shown first. The tolerance covers values like 0.499994 that Tev1 returns with choice "A"."""
    if rec is None or rec["choice"] is None or abs(rec["p_A"] - 0.5) < TIE:
        return None
    return "R1" if rec["choice"] == row["r1_label"] else "R2"


def _prob(row, rec, response):
    """The judge's probability for `response` ("R1"/"R2") in one prompt, or None."""
    if rec is None or rec["p_A"] is None:
        return None
    p_r1 = rec["p_A"] if row["r1_label"] == "A" else 1.0 - rec["p_A"]
    return p_r1 if response == "R1" else 1.0 - p_r1


def table(rows, records):
    """One line per perturbed prompt with the three verdicts and three probabilities."""
    design = {r["id"]: r for r in rows}
    answer = {(x["id"], x["repeat"]): x for x in records}
    lines = []
    for row in rows:
        if row["bias"] == "original":
            continue
        orig = design[row["original"]]
        first, second, pert = answer.get((orig["id"], 0)), answer.get((orig["id"], 1)), answer.get((row["id"], 0))
        lines.append({"id": row["id"], "item": row["item"], "dataset": row["dataset"], "source": row["source"],
                      "bias": row["bias"], "variant": row["variant"], "target": row["target"],
                      "y": _picked(orig, first), "y_rep": _picked(orig, second), "y_hat": _picked(row, pert),
                      "p": _prob(orig, first, row["target"]), "p_rep": _prob(orig, second, row["target"]),
                      "p_hat": _prob(row, pert, row["target"])})
    return pd.DataFrame(lines)


def _permutation_p(g, n_perm, rng):
    """Two-sided permutation p-value for "the perturbation changes verdicts as often as repetition".

    Per item i with k_i perturbed prompts, the k_i + 1 changed/unchanged outcomes (perturbed
    prompts and the repetition) are exchangeable under the null hypothesis. Picking one of them
    at random as the repetition gives the null distribution of the excess."""
    flip = (g.y_hat != g.y).astype(int)
    rep = (g.y_rep != g.y).astype(int)
    per = pd.DataFrame({"item": g.item, "flip": flip, "rep": rep}).groupby("item").agg(
        k=("flip", "size"), flips=("flip", "sum"), rep=("rep", "first"))
    k, total = per["k"].to_numpy(), (per["flips"] + per["rep"]).to_numpy()   # outcomes per item: k + 1
    n = k.sum()
    observed = (per["flips"].sum() - (k * per["rep"]).sum()) / n
    # a random member of the pool of k + 1 outcomes is a changed verdict with probability total / (k + 1)
    is_rep = rng.random((n_perm, len(k))) < total / (k + 1)
    null = (total.sum() - (is_rep * (k + 1)).sum(axis=1)) / n
    hits = int(np.sum(np.abs(null) >= abs(observed) - 1e-12))
    return (hits + 1) / (n_perm + 1)              # never zero (Phipson and Smyth, 2010)


def mcnemar(b, c):
    """Exact McNemar test: b and c are the two kinds of discordant pairs. Two-sided p-value."""
    return binomtest(b, b + c, 0.5).pvalue if b + c > 0 else 1.0


def _bootstrap(g, gp, n_boot, rng):
    """95% intervals for RR, CR, the excess of changed verdicts (CR - RR), accuracy after the
    perturbation and the mean shift, resampling items (all prompts of an item stay together,
    since they share its original). Verdict statistics use the prompts with all three verdicts
    (g), the shift the prompts with all three probabilities (gp)."""
    v = g.assign(flip=(g.y_hat != g.y).astype(float), rep_flip=(g.y_rep != g.y).astype(float),
                 acc=(g.y_hat == "R1").astype(float)).groupby("item").agg(
        n=("flip", "size"), flip=("flip", "sum"), rep_flip=("rep_flip", "sum"), acc=("acc", "sum"))
    sh = gp.assign(shift=gp.p_hat - gp.p).groupby("item").agg(n_p=("shift", "size"), shift=("shift", "sum"))
    s = v.join(sh, how="outer").fillna(0.0)
    w = rng.multinomial(len(s), np.full(len(s), 1.0 / len(s)), size=n_boot)      # n_boot x items
    n = w @ s["n"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        stats = {"RR": 1 - (w @ s["flip"].to_numpy()) / n, "CR": 1 - (w @ s["rep_flip"].to_numpy()) / n,
                 "acc_after": (w @ s["acc"].to_numpy()) / n,
                 "shift": (w @ s["shift"].to_numpy()) / (w @ s["n_p"].to_numpy())}
    stats["excess"] = stats["CR"] - stats["RR"]
    stats["fallacy_excess"] = stats["RR"] + stats["CR"] - 1
    return {k: (np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)) for k, v in stats.items()}


def summarize(df, by=("bias",), n_boot=10_000, seed=0):
    """The metrics of Section 3 for every group of `by` (e.g. ["bias"] or ["bias", "variant"])."""
    rng = np.random.default_rng(seed)
    out = []
    for key, group in df.groupby(list(by)):
        g = group.dropna(subset=["y", "y_rep", "y_hat"])       # verdict metrics need all three verdicts
        gp = group.dropna(subset=["p", "p_rep", "p_hat"])      # shifts need all three probabilities
        if g.empty:
            continue
        flip, rep_flip = g.y_hat != g.y, g.y_rep != g.y
        b, c = int((flip & ~rep_flip).sum()), int((~flip & rep_flip).sum())
        ci = _bootstrap(g, gp, n_boot, rng)
        key = key if isinstance(key, tuple) else (key,)
        out.append({**dict(zip(by, key)), "n": len(g), "items": g.item.nunique(),
                    "RR": 1 - flip.mean(), "RR_lo": ci["RR"][0], "RR_hi": ci["RR"][1],
                    "CR": 1 - rep_flip.mean(), "CR_lo": ci["CR"][0], "CR_hi": ci["CR"][1],
                    "acc_before": (g.y == "R1").mean(), "acc_after": (g.y_hat == "R1").mean(),
                    "acc_after_lo": ci["acc_after"][0], "acc_after_hi": ci["acc_after"][1],
                    "excess": flip.mean() - rep_flip.mean(), "excess_lo": ci["excess"][0], "excess_hi": ci["excess"][1],
                    "fallacy_excess": 1 - flip.mean() - rep_flip.mean(),
                    "fallacy_excess_lo": ci["fallacy_excess"][0], "fallacy_excess_hi": ci["fallacy_excess"][1],
                    "flips_perturbation_only": b, "flips_repetition_only": c,
                    "p_bias": _permutation_p(g, n_boot, rng), "p_mcnemar": mcnemar(b, c) if g.item.is_unique else np.nan,
                    "shift": (gp.p_hat - gp.p).mean(), "shift_lo": ci["shift"][0], "shift_hi": ci["shift"][1],
                    "noise_floor": (gp.p_rep - gp.p).abs().mean()})
    res = pd.DataFrame(out)
    if "bias" in res:
        # Fallacy oversight runs the other way: a changed verdict is the correct reaction, and the bias is a verdict
        # that does not change. Its share of correct reactions is 1 - RR (as CALM reports it). A judge that changes
        # its verdict by chance in a share 1 - CR of repetitions also undoes a correct reaction by chance about as
        # often, so the unchanged verdicts beyond chance are RR - (1 - CR) = RR + CR - 1. With this, the column
        # `excess` is the bias beyond chance in every test, 0 for an unbiased judge. The permutation test asks whether
        # the perturbation changes verdicts as often as repetition, which is not the question here, so fallacy
        # oversight has no p-value; its interval of the excess decides.
        fal = res["bias"] == "fallacy"
        res["correct_share"] = np.where(fal, 1 - res["RR"], np.nan)
        for c in ["excess", "excess_lo", "excess_hi"]:
            res.loc[fal, c] = res.loc[fal, f"fallacy_{c}"]
        res.loc[fal, ["p_bias", "p_mcnemar"]] = np.nan
    res = res.drop(columns=[c for c in res.columns if c.startswith("fallacy_")])
    return res


def compare_formats(df_a, df_b, by=("bias",), n_boot=10_000, seed=0):
    """Paired comparison of two formats of the same model on the same prompts (or of two judges on the same prompts).

    Changed verdicts: the difference in the share of changed verdicts, tested by swapping the
    two formats at random within items (a sign-flip test over items). Shifts: the difference
    in mean shift with a bootstrap over items."""
    rng = np.random.default_rng(seed)
    m = df_a.merge(df_b, on=["id", "item", "bias", "variant"], suffixes=("_a", "_b"))
    out = []
    for key, group in m.groupby(list(by)):
        g = group.dropna(subset=["y_a", "y_hat_a", "y_b", "y_hat_b"])      # verdicts in both formats
        if g.empty:
            continue
        fa, fb = (g.y_hat_a != g.y_a).astype(int), (g.y_hat_b != g.y_b).astype(int)
        d_item = (fa - fb).groupby(g.item).sum().to_numpy()          # per item, a minus b
        observed = d_item.sum() / len(g)
        signs = rng.choice([-1, 1], size=(n_boot, len(d_item)))
        p_flip = (int(np.sum(np.abs(signs @ d_item / len(g)) >= abs(observed) - 1e-12)) + 1) / (n_boot + 1)
        d = ((group.p_hat_a - group.p_a) - (group.p_hat_b - group.p_b)).dropna()   # probabilities in both
        boot = np.full(1, np.nan)                  # no probabilities, e.g. an LLM judge: no shift difference
        if len(d):
            per_item = d.groupby(group.loc[d.index, "item"]).agg(["sum", "size"])
            w = rng.multinomial(len(per_item), np.full(len(per_item), 1.0 / len(per_item)), size=n_boot)
            boot = (w @ per_item["sum"].to_numpy()) / (w @ per_item["size"].to_numpy())
        key = key if isinstance(key, tuple) else (key,)
        out.append({**dict(zip(by, key)), "n": len(g), "RR_a": 1 - fa.mean(), "RR_b": 1 - fb.mean(),
                    "p_changed_verdicts": p_flip, "shift_diff": d.mean(),
                    "shift_diff_lo": np.percentile(boot, 2.5), "shift_diff_hi": np.percentile(boot, 97.5)})
    return pd.DataFrame(out)


def adjust(p_values, alpha=0.05):
    """Benjamini-Hochberg: (reject, adjusted p-values) for a list of p-values. Missing p-values (fallacy oversight)
    are left out of the correction and come back as not rejected with a missing adjusted p-value."""
    p = np.asarray(list(p_values), dtype=float)
    ok = ~np.isnan(p)
    reject, p_adj = np.zeros(len(p), dtype=bool), np.full(len(p), np.nan)
    if ok.any():
        reject[ok], p_adj[ok], _, _ = multipletests(p[ok], alpha=alpha, method="fdr_bh")
    return reject, p_adj


def accuracy_change(df, n_boot=10_000, seed=0):
    """Accuracy before and after a perturbation, with 95% bootstrap intervals over items for the change.

    Before: the share of verdicts for R1 on the first sending of the original prompts of the test's items. After:
    the share of verdicts for R1 on the perturbed prompts. Each counts every prompt with a verdict, so tests on the
    same items (the two authority tests, the two sides of sentiment) share the same accuracy before, unlike
    acc_before of `summarize`, which counts only prompts with all three verdicts."""
    rng = np.random.default_rng(seed)
    per = df.groupby("item").agg(y=("y", "first"), a=("y_hat", lambda s: (s == "R1").sum()),
                                 n=("y_hat", lambda s: s.notna().sum()))
    b_has = per.y.notna().to_numpy(float)
    b_ok = (per.y == "R1").to_numpy(float)
    a, n = per.a.to_numpy(float), per.n.to_numpy(float)
    before, after = b_ok.sum() / b_has.sum(), a.sum() / n.sum()
    w = rng.multinomial(len(per), np.full(len(per), 1.0 / len(per)), size=n_boot)
    with np.errstate(invalid="ignore", divide="ignore"):
        change = (w @ a) / (w @ n) - (w @ b_ok) / (w @ b_has)
    lo, hi = np.nanpercentile(change, [2.5, 97.5])
    return {"items": len(per), "before": before, "after": after, "change": after - before,
            "change_lo": lo, "change_hi": hi}
