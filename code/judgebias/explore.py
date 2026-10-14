"""One long table for the exploratory models: one line per model, format and perturbed prompt.

    from judgebias import explore
    d = explore.long_table()          # pandas DataFrame
    explore.save(d)                   # outputs/exploratory/long.csv.gz, read by the R scripts in R/

Columns, besides those of metrics.table:
    model, format     the judge and the request format
    flip, rep_flip    1 if the verdict changed under the perturbation / on the plain repeat (NaN if a verdict is missing)
    shift             p_hat - p, the change in the probability of the targeted response
    margin            |P(R1) - 0.5| on the first sending of the original: how decided the judge was
    orig_correct      1 if the judge picked the better response on the original
    len_r1, len_r2    characters of the better and the worse response in the original prompt
    log_len_ratio     log(len_r1 / len_r2)
    source            the dataset the item comes from (short name)
"""
import numpy as np
import pandas as pd

from . import authority, design, metrics, paths, run
from .calm import BLOCK

DATA = paths.EXPLORATORY
SOURCES = {"Intel/orca_dpo_pairs": "Orca-DPO", "jondurbin/py-dpo-v0.1": "Py-DPO",
           "yleo/emerton_dpo_pairs_judge": "Emerton-DPO", "jondurbin/truthy-dpo-v0.1": "Truthy-DPO",
           "xDAN2099/xDAN-SFT-DPO-Roleplay-NSFW-with-lf": "Roleplay"}


def _lengths(rows):
    """{item: (characters of R1, characters of R2)}, read from the original prompts."""
    out = {}
    for r in rows:
        if r["bias"] != "original":
            continue
        a, b = [m["text"] for m in BLOCK.finditer(r["prompt"])][:2]
        out[r["item"]] = (len(a), len(b)) if r["r1_label"] == "A" else (len(b), len(a))
    return out


def long_table(models=None, formats=tuple(run.FORMATS)):
    """All perturbed prompts of all models and formats, with the features above."""
    rows = design.build_design(citations=authority.load_citations())
    lengths = _lengths(rows)
    parts = []
    for m in models or [x for x in run.MODELS if "mercury" not in x]:
        for f in formats:
            df = metrics.separate_additions(metrics.table(rows, run.load_records(m, f)))
            parts.append(df.assign(model=m.split("/")[1].replace("-experimental", ""), format=f))
    d = pd.concat(parts, ignore_index=True)
    both = d[["y", "y_hat"]].notna().all(axis=1)
    d["flip"] = np.where(both, (d.y_hat != d.y).astype(float), np.nan)
    d["rep_flip"] = np.where(d[["y", "y_rep"]].notna().all(axis=1), (d.y_rep != d.y).astype(float), np.nan)
    d["shift"] = d.p_hat - d.p
    p_r1 = np.where(d.target == "R1", d.p, 1 - d.p)           # the judge's P(R1) on the original
    d["margin"] = np.abs(p_r1 - 0.5)
    d["orig_correct"] = np.where(d.y.notna(), (d.y == "R1").astype(float), np.nan)
    d["len_r1"] = d.item.map(lambda i: lengths[i][0])
    d["len_r2"] = d.item.map(lambda i: lengths[i][1])
    d["log_len_ratio"] = np.log(d.len_r1 / d.len_r2)
    d["source"] = d.source.map(lambda s: SOURCES.get(s, s))
    return d


def save(d, name="long.csv.gz"):
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / name
    d.drop(columns=["y", "y_rep", "y_hat"]).to_csv(path, index=False)
    return path


def ensemble_records(models, fmt="calm"):
    """Records of an ensemble judge: per request the mean P(A) of `models` (only requests all of them answered)."""
    recs = [{(r["id"], r["repeat"]): r for r in run.load_records(m, fmt)} for m in models]
    out = []
    for key in recs[0]:
        ps = [r[key]["p_A"] for r in recs if key in r and r[key]["ok"]]
        if len(ps) == len(models):
            p = float(np.mean(ps))
            out.append({"id": key[0], "repeat": key[1], "ok": True, "p_A": p,
                        "choice": "A" if p > 0.5 else "B" if p < 0.5 else None})
    return out


def criterion_shifts(models=None):
    """Decomposed format: for every perturbed prompt and each of the six criteria, the shift of the
    probability of the targeted response (perturbed minus first sending of the original)."""
    from .calm import FACTOR_IDS
    rows = design.build_design(citations=authority.load_citations())
    by_id = {r["id"]: r for r in rows}
    out = []
    for m in models or [x for x in run.MODELS if "mercury" not in x]:
        recs = {(r["id"], r["repeat"]): r for r in run.load_records(m, "decomposed")}
        for r in rows:
            if r["bias"] == "original":
                continue
            a, o = recs.get((r["id"], 0)), recs.get((r["original"], 0))
            if not (a and o and a["ok"] and o["ok"]):
                continue
            for k in FACTOR_IDS:
                after = metrics._prob(r, {"p_A": a["p_A_questions"][k]}, r["target"])
                before = metrics._prob(by_id[r["original"]], {"p_A": o["p_A_questions"][k]}, r["target"])
                out.append({"model": m.split("/")[1].replace("-experimental", ""), "bias": r["bias"],
                            "variant": r["variant"], "criterion": k, "shift": after - before})
    return metrics.separate_additions(pd.DataFrame(out))


def profile_correlation(path=paths.TABLES / "comparison_with_calm.csv"):
    """Do the typed models and CALM's LLM judges find the same cues hard?

    Per version of CALM's tests (its Table 8, a test without versions counts once), the median
    robustness rate and excess over the judges of each kind, and the Spearman correlation between the two kinds.
    Versions of one test are not independent, so the same correlation is also given per test (versions averaged).
    Returns (profile, correlations)."""
    from scipy.stats import spearmanr
    c = pd.read_csv(path)
    c = c.assign(kind=np.where(c.kind == "typed", "typed", "llm"))
    prof = c.groupby(["bias", "variant", "kind"])[["value", "excess"]].median().unstack("kind").dropna()
    prof.columns = [f"{m}_{k}" for m, k in prof.columns]
    tests = prof.groupby("bias").mean()
    out = []
    for level, t in [("version", prof), ("test", tests)]:
        for m in ["value", "excess"]:
            rho, p = spearmanr(t[f"{m}_typed"], t[f"{m}_llm"])
            out.append({"level": level, "measure": "RR" if m == "value" else "excess", "n": len(t), "rho": rho, "p": p})
    return prof.reset_index(), pd.DataFrame(out)


def model_similarity(path=paths.TABLES / "comparison_with_calm.csv"):
    """Is a typed model whose architecture is closer to an LLM also closer to CALM's LLM judges?

    Per typed model, the Spearman correlation between its robustness rate and excess per version of CALM's tests
    and the median of CALM's six LLM judges. Exploratory: eight models, two architectures unknown."""
    from scipy.stats import spearmanr
    c = pd.read_csv(path)
    llm = c[c.kind != "typed"].groupby(["bias", "variant"])[["value", "excess"]].median()
    out = []
    for m, g in c[c.kind == "typed"].groupby("judge"):
        g = g.set_index(["bias", "variant"])[["value", "excess"]].join(llm, rsuffix="_llm").dropna()
        out.append({"model": m, "n": len(g), "rho_RR": spearmanr(g.value, g.value_llm)[0],
                    "rho_excess": spearmanr(g.excess, g.excess_llm)[0]})
    return pd.DataFrame(out).sort_values("rho_excess", ascending=False)
