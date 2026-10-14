"""Builds assets/site-data.js for the website from the released results (sibling folder ../TDM_as_a_Judge).

    python tools/build_data.py
"""
import difflib, gzip, json, re
from pathlib import Path
import numpy as np
import pandas as pd

SITE = Path(__file__).resolve().parents[1]
REPO = SITE.parent / "TDM_as_a_Judge"
T = REPO / "code" / "outputs" / "tables"
D = REPO / "data"
EXP = REPO / "code" / "outputs" / "exploratory"
DELTA = 0.05

TDMS = [("jev-1.13", "Jev 1.13", "typesafe_jev-1.13"), ("kev-4b", "Kev 4B", "jaredpalmer_kev-4b"),
        ("tev1-4b", "Tev1 4B", "togethercomputer_tev1-4b-experimental"), ("solar-decide", "Solar Decide", "upstage_solar-decide"),
        ("d1", "D1", "liquid_d1"), ("pplx-decider-v1.1-27b", "Perplexity Decider", "perplexity_pplx-decider-v1.1-27b"),
        ("clef", "Clef", "cloudflare_clef"), ("clef-flash", "Clef Flash", "cloudflare_clef-flash")]
LLMS = [("ChatGPT", "GPT-3.5 Turbo", "LLM 2024", "openai_gpt-3.5-turbo"), ("GPT-4o", "GPT-4o", "LLM 2024", "openai_gpt-4o-2024-05-13"),
        ("GPT-6.1 Sol", "GPT-6.1 Sol", "LLM 2026", "openai_gpt-6.1-sol"), ("Claude Sonnet 5.5", "Claude Sonnet 5.5", "LLM 2026", "claude-sonnet-5-5"),
        ("Gemini 3.8 Flash", "Gemini 3.8 Flash", "LLM 2026", "google_gemini-3.8-flash"), ("Qwen3.8-27B", "Qwen3.8-27B", "LLM 2026", "qwen_qwen3.8-27b"),
        ("Claude Haiku 5.5", "Claude Haiku 5.5", "LLM 2026", "anthropic_claude-haiku-5.5")]
TESTS = [  # key in the tables, name, ours, dataset, what changes
    ("sentiment", "Sentiment", False, "fact", "One response is rewritten in a cheerful, sad, angry or fearful tone."),
    ("position", "Position", False, "alignment", "The two responses swap places."),
    ("authority, better response", "Authority (on better)", True, "alignment", "A fake citation (a book, a URL or a quote) is added to the better response."),
    ("authority", "Authority", False, "alignment", "A fake citation is added to the worse response."),
    ("bandwagon", "Bandwagon", False, "alignment", "The prompt claims that 60 to 90% of people prefer the worse response."),
    ("compassion", "Compassion fade", False, "alignment", "The responses carry model names, a well-known one on the worse response."),
    ("verbosity, better response", "Verbosity (on better)", True, "fact", "The better response is padded to about twice its length, without new content."),
    ("verbosity", "Verbosity", False, "fact", "The worse response is padded to about twice its length."),
    ("distraction", "Distraction", False, "alignment", "An irrelevant sentence about the author of one response is added."),
    ("diversity", "Diversity", False, "alignment", "A note on the identity of one response's author is added (six groups)."),
    ("fallacy", "Fallacy oversight", False, "fact", "The reasoning of the better response is broken, so here the verdict should change."),
    ("diversity, neutral note", "Neutral note", True, "alignment", "A neutral note (\"Software engineer\") is added, as a control for diversity."),
]
r4 = lambda x: None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


def classify(lo, hi):
    return "bias" if lo > DELTA else "none" if hi < DELTA else "unclear"


def cells():
    m = pd.read_csv(T / "main.csv"); m = m[m.format == "calm"]
    l = pd.read_csv(T / "llm_main.csv")
    out = {}
    for tid, _, _ in TDMS:
        for _, r in m[m.model == tid].iterrows():
            out[(tid, r.bias)] = {"e": r4(r.excess), "lo": r4(r.excess_lo), "hi": r4(r.excess_hi), "c": classify(r.excess_lo, r.excess_hi),
                                  "rr": r4(r.RR), "cr": r4(r.CR), "shift": r4(r["shift"])}
    for lid, _, _, _ in LLMS:
        for _, r in l[l.judge == lid].iterrows():
            out[(lid, r.bias)] = {"e": r4(r.excess), "lo": r4(r.excess_lo), "hi": r4(r.excess_hi), "c": r["class"],
                                  "rr": r4(r.RR), "cr": r4(r.CR)}
    return out


def judges(cell):
    pr = pd.read_csv(T / "practice.csv").set_index("judge")
    out = []
    for tid, name, _ in TDMS:
        p = pr.loc[name]
        out.append({"id": tid, "name": name, "kind": "TDM"})
    for lid, name, kind, _ in LLMS:
        p = pr.loc[lid]
        out.append({"id": lid, "name": name, "kind": kind})
    for j in out:
        p = pr.loc[j["name"] if j["kind"] == "TDM" else j["id"]]
        j.update({"nBiased": sum(cell[(j["id"], t[0])]["c"] == "bias" for t in TESTS), "meanExcess": r4(p.ex_calm),
                  "usd1000": r4(p.usd), "latency": r4(p.latency), "crAl": r4(p.cr_al), "accAl": r4(p.acc_al), "accFa": r4(p.acc_fa),
                  "eceAl": r4(p.ece_al) if j["kind"] == "TDM" else None,
                  "layNative": r4(p.lay_native) if j["kind"] == "TDM" else None})
    return out


def accuracy():
    a = pd.read_csv(T / "accuracy_change.csv")
    name = {lid: n for lid, n, _, _ in LLMS}
    return [{"judge": name.get(r.judge, r.judge), "test": r.test, "before": r4(r.before), "after": r4(r.after)} for _, r in a.iterrows()]


def bandwagon():
    v = pd.read_csv(T / "variants.csv"); v = v[(v.format == "calm") & (v.bias == "bandwagon")]
    lv = pd.read_csv(T / "llm_variants.csv"); lv = lv[lv.bias == "bandwagon"]
    out = [{"judge": tid, "pct": int(r.variant), "e": r4(r.excess), "rr": r4(r.RR)} for tid, _, _ in TDMS for _, r in v[v.model == tid].iterrows()]
    out += [{"judge": lid, "pct": int(r.variant), "e": r4(r.excess), "rr": r4(r.RR)} for lid, _, _, _ in LLMS for _, r in lv[lv.judge == lid].iterrows()]
    return out


def identity():
    """Median excess per identity group (both responses pooled) for the TDMs (CALM format) and the 2026 LLM judges."""
    v = pd.read_csv(T / "variants.csv"); v = v[(v.format == "calm") & (v.bias == "diversity")]
    lv = pd.read_csv(T / "llm_variants.csv"); lv = lv[(lv.bias == "diversity") & (lv.kind == "LLM 2026")]
    out = []
    for name, df, who in [("TDM", v, "model"), ("LLM 2026", lv, "judge")]:
        df = df.assign(group=df.variant.str.replace(r"-on-R[12]$", "", regex=True))
        per = df.groupby([who, "group"]).excess.mean().reset_index()
        for g, e in per.groupby("group").excess.median().items():
            out.append({"kind": name, "group": g, "e": r4(e)})
    return out


def long_table():
    d = pd.read_csv(EXP / "long.csv.gz")
    return d[(d.format == "calm") & (d.bias != "fallacy")].copy()


def test_key(bias, variant):
    if bias == "authority": return "authority, better response" if variant.endswith("on-R1") else "authority"
    if bias == "verbosity": return "verbosity, better response" if variant == "R1-padded" else "verbosity"
    if bias == "diversity": return "diversity, neutral note" if variant.startswith("Software engineer") else "diversity"
    return bias


def curves(d):
    edges = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0001]
    d = d.dropna(subset=["flip", "margin"]).copy()
    d["test"] = [test_key(b, v) for b, v in zip(d.bias, d.variant)]
    d["conf"] = 0.5 + d.margin
    d["bin"] = pd.cut(d.conf, edges, right=False, labels=False)
    out = {}
    others = d[~d.test.isin(["sentiment", "authority, better response"])]
    for t, g in list(d.groupby("test")) + [("all", d), ("others", others)]:
        rows = g.groupby("bin").agg(n=("flip", "size"), flip=("flip", "mean")).reset_index()
        out[t] = [{"lo": edges[int(b)], "hi": min(edges[int(b) + 1], 1.0), "n": int(n), "flip": r4(f)} for b, n, f in rows.itertuples(index=False) if n >= 30]
    return out


def hidden(d):
    d = d.dropna(subset=["flip"]).copy()
    d["test"] = [test_key(b, v) for b, v in zip(d.bias, d.variant)]
    d["hidden"] = (d.flip == 0) & (d["shift"].abs() > 0.10)
    per = d.groupby(["test", "model"]).agg(flip=("flip", "mean"), hidden=("hidden", "mean")).reset_index()
    med = per.groupby("test")[["flip", "hidden"]].median()
    return {t: {"flip": r4(r.flip), "hidden": r4(r.hidden)} for t, r in med.iterrows()}


ANS = re.compile(r"\[The Start of (.+?)[’']s Answer\]\n(.*?)\n\[The End of (?:.+?)[’']s Answer\]", re.S)


def parse(prompt):
    user = prompt.split("[User Question]", 1)[1]
    q = re.split(r"\[The Start of ", user, maxsplit=1)[0].strip()
    a = ANS.findall(user)
    tail = user[user.rfind("Answer]") + len("Answer]"):].strip()
    return q, [(lab, txt.strip()) for lab, txt in a], tail


def diff(old, new):
    o, n = old.split(), new.split()
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, o, n, autojunk=False).get_opcodes():
        if op == "equal": out.append(["", " ".join(n[j1:j2])])
        if op in ("delete", "replace"): out.append(["del", " ".join(o[i1:i2])])
        if op in ("insert", "replace"): out.append(["ins", " ".join(n[j1:j2])])
    return out


def examples():
    p = pd.read_json(D / "prompts.jsonl.gz", lines=True).set_index("id")
    tdm = {}
    for tid, _, f in TDMS:
        a = pd.read_json(D / "answers" / "tdms" / f"{f}__calm.jsonl.gz", lines=True)
        tdm[tid] = a[a.repeat == 0].set_index("id")[["choice", "p_A"]]
    llm = {}
    for lid, _, _, f in LLMS:
        a = pd.read_json(D / "answers" / "llm_judges" / f"{f}.jsonl.gz", lines=True)
        llm[lid] = a[a.repeat == 0].set_index("id")["verdict"]
    pert = p[p.bias != "original"].copy()
    pert["test"] = [test_key(b, v) for b, v in zip(pert.bias, pert.variant)]
    out = []
    for key, name, *_ in TESTS:
        cand = pert[(pert.test == key) & (~pert.source.str.contains("Roleplay"))]
        if key == "sentiment": cand = cand[cand.variant.str.endswith("on-R1")]
        best = None
        for pid, r in cand.iterrows():
            orig = p.loc[r.original]
            q, ans, _ = parse(orig.prompt)
            size = len(q) + sum(len(t) for _, t in ans)
            if size > 1300 or len(ans) != 2: continue
            p1 = lambda tid, i, lab: (tdm[tid].loc[i, "p_A"] if lab == "A" else 1 - tdm[tid].loc[i, "p_A"])
            try:
                flips = sum((tdm[t].loc[pid, "choice"] == r.r1_label) != (tdm[t].loc[r.original, "choice"] == orig.r1_label)
                            for t, _, _ in TDMS)          # flips of the verdict for the better response
                dp = np.mean([p1(t, pid, r.r1_label) - p1(t, r.original, orig.r1_label) for t, _, _ in TDMS])
            except KeyError:
                continue
            score = (flips, abs(dp), -size)
            if best is None or score > best[0]: best = (score, pid)
        pid = best[1]; r = p.loc[pid]; orig = p.loc[r.original]
        q, a0, tail0 = parse(orig.prompt); _, a1, tail1 = parse(r.prompt)
        lab0 = {"A": a0[0], "B": a0[1]}; lab1 = {"A": a1[0], "B": a1[1]}
        r1_0, r1_1 = orig.r1_label, r.r1_label
        other = lambda x: "B" if x == "A" else "A"
        def side(lab, src, ref, ref_r1, now_r1):     # response shown at `lab`, diffed with the same response in the original
            is_r1 = lab == now_r1
            ref_lab = ref_r1 if is_r1 else other(ref_r1)
            return {"label": src[lab][0], "r1": is_r1, "seg": diff(ref[ref_lab][1], src[lab][1])}
        ex = {"test": key, "variant": r.variant, "source": r.source, "question": q,
              "orig": {"A": {"label": lab0["A"][0], "r1": r1_0 == "A", "text": lab0["A"][1]},
                       "B": {"label": lab0["B"][0], "r1": r1_0 == "B", "text": lab0["B"][1]}, "note": tail0},
              "pert": {"A": side("A", lab1, lab0, r1_0, r1_1), "B": side("B", lab1, lab0, r1_0, r1_1), "note": tail1},
              "tdm": [{"judge": t, "before": r4(tdm[t].loc[r.original, "p_A"] if r1_0 == "A" else 1 - tdm[t].loc[r.original, "p_A"]),
                       "after": r4(tdm[t].loc[pid, "p_A"] if r1_1 == "A" else 1 - tdm[t].loc[pid, "p_A"])} for t, _, _ in TDMS],
              "llm": [{"judge": l, "before": (llm[l].get(r.original) == r1_0) if llm[l].get(r.original) else None,
                       "after": (llm[l].get(pid) == r1_1) if llm[l].get(pid) else None} for l, _, _, _ in LLMS]}
        out.append(ex)
    return out


def main():
    cell = cells()
    d = long_table()
    data = {
        "tests": [{"key": k, "name": n, "ours": o, "dataset": ds, "what": w} for k, n, o, ds, w in TESTS],
        "judges": judges(cell),
        "cells": [{"judge": j, "test": t, **v} for (j, t), v in cell.items()],
        "accuracy": accuracy(), "bandwagon": bandwagon(), "curves": curves(d), "hidden": hidden(d), "examples": examples(),
        "overlap": [{"pair": r.pair, "flips": r4(r["if the other flips"]), "stays": r4(r["if the other stays"])}
                    for _, r in pd.read_csv(T / "llm_item_overlap_summary.csv").iterrows()],
        "identity": identity(),
        "bothOrders": [{"judge": r.model, "one": r4(r.accuracy_one_order), "both": r4(r.accuracy_both_orders), "change": r4(r.change),
                        "lo": r4(r.change_lo), "hi": r4(r.change_hi)} for _, r in pd.read_csv(T / "both_orders.csv").iterrows()],
    }
    js = "window.SITE_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n"
    (SITE / "assets" / "site-data.js").write_text(js)
    print(f"assets/site-data.js: {len(js) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
