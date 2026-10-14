"""The experimental design: every prompt we send, built from CALM's data with CALM's templates.

`build_design()` returns one row per prompt (it needs CALM's repository in external/calm). `load_prompts()` returns
the same rows from the released file data/prompts.jsonl.gz, without CALM. A row is a dict with
    id        unique id, e.g. "3f2a91c0b7d4/bandwagon/90"
    item      id of the question-response pair (shared by all tests on that pair)
    dataset   "alignment" or "fact"
    source    CALM's data_resource (e.g. "GSM8K", "jondurbin/truthy-dpo-v0.1")
    bias      "original" or one of the nine biases
    variant   e.g. "90" for 90% bandwagon, "angry-on-R1" for sentiment
    original  id of the original prompt this row is compared with (None for originals)
    r1_label  "A" or "B": where the better response R1 is shown in this prompt
    target    "R1" or "R2": the response whose probability we follow (None for originals)
    prompt    CALM's prompt string, as CALM's own template function returns it

Every pair gets one response order, drawn with a fixed seed so that R1 comes first in
half of the pairs of each dataset. All prompts on that pair use this order, except the
position test, which swaps it.
"""
import hashlib
import json
import random

from . import authority, files, padding, paths
from .calm import T, load

SEED = 20261004
PERCENTAGES = [60, 70, 80, 90]                                     # bandwagon, as in CALM
GROUPS = ["Homosexual", "Black", "Female", "HIV Positive", "Refugees", "Muslim believers"]   # diversity, as in CALM
NEUTRAL_GROUP = "Software engineer"                                 # our control note
TONES = ["cheerful", "sad", "angry", "fear"]                        # sentiment, as in CALM
STRONG_NAMES = ["GPT-5.6", "Claude Opus 5.5", "Gemini 3.1 Pro"]     # compassion fade: on the worse response
WEAK_NAMES = ["Llama-2-7B", "GPT-3.5", "Mistral-7B"]                # compassion fade: on the better response

EXPECTED = {"original": 972, "position": 439, "verbosity": 1000, "fallacy": 500, "authority": 900,
            "sentiment": 1193, "bandwagon": 1756, "distraction": 878, "diversity": 6146, "compassion": 439}


def pair_id(item):
    """A stable id for a question with its two responses."""
    text = item["question"] + "\x00" + item["answer1"] + "\x00" + item["answer2"]
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def draw_orders(pair_ids, rng):
    """{pair id: True if R1 is shown first}. Exactly half of the pairs (rounded down) get True."""
    ids = sorted(pair_ids)
    rng.shuffle(ids)
    return {pid: i < len(ids) // 2 for i, pid in enumerate(ids)}


class Pair:
    """One question with its better (R1) and worse (R2) response, shown in a fixed order."""

    def __init__(self, item, dataset, r1_first):
        self.item, self.dataset, self.r1_first = item, dataset, r1_first
        self.id = pair_id(item)
        self.q = item["question"]

    def label(self, response):
        """Where response "R1" or "R2" is shown: "A" or "B"."""
        first = "R1" if self.r1_first else "R2"
        return "A" if response == first else "B"

    def shown(self, r1, r2):
        """The two texts in display order (A, B), given what stands in for R1 and R2."""
        return (r1, r2) if self.r1_first else (r2, r1)

    def row(self, bias, variant, prompt, target, r1_label=None):
        return {"id": f"{self.id}/{bias}/{variant}", "item": self.id, "dataset": self.dataset,
                "source": self.item["data_resource"], "bias": bias, "variant": variant,
                "original": f"{self.id}/original/-", "r1_label": r1_label or self.label("R1"),
                "target": target, "prompt": prompt}


def build_design(citations=None, seed=SEED):
    """All rows of the experiment. `citations` = authority.load_citations() (our regenerated citations)."""
    rng = random.Random(seed)
    alignment = load("alignment_dataset")
    authority_items = load("authority_bias_standardized")
    verbosity = load("verbosity_bias_standardized")
    fallacy = {pair_id(x): x for x in load("fallacy_oversight_bias_standardized")}
    sentiment = load("sentiment_bias_standardized")
    padded_better = padding.load_padded()        # our padded better responses (verbosity on the better response)

    # one order per pair, balanced within each dataset
    orders = draw_orders({pair_id(x) for x in alignment}, rng)
    orders |= draw_orders({pair_id(x) for x in verbosity} | {pair_id(x) for x in sentiment}, rng)
    pairs = {}

    def pair(item, dataset):
        pid = pair_id(item)
        if pid not in pairs:
            pairs[pid] = Pair(item, dataset, orders[pid])
        return pairs[pid]

    rows = []
    # -- alignment set: position, bandwagon, distraction, diversity, compassion fade ------------
    for item in alignment:
        p = pair(item, "alignment")
        a1, a2 = item["answer1"], item["answer2"]
        A, B = p.shown(a1, a2)
        rows.append(p.row("position", "swapped", T.evaluate_ai_responses(p.q, B, A),
                          target="R1" if p.r1_first else "R2", r1_label="B" if p.r1_first else "A"))
        worse = f"Assistant {p.label('R2')}"
        for pct in PERCENTAGES:
            rows.append(p.row("bandwagon", str(pct),
                              T.Bandwagon_effect_prompt(p.q, A, B, number=pct, chosen_model=worse), target="R2"))
        for r in ("R1", "R2"):
            template = T.Dstraction_Bias_A if p.label(r) == "A" else T.Dstraction_Bias_B
            rows.append(p.row("distraction", f"about-{r}", template(p.q, A, B), target=r))
        for r in ("R1", "R2"):
            for group in GROUPS + [NEUTRAL_GROUP]:
                rows.append(p.row("diversity", f"{group}-on-{r}",
                                  T.Diversity_bias_prompt(p.q, A, B, chosen_model=f"Assistant {p.label(r)}", group=group),
                                  target=r))
        strong, weak = rng.choice(STRONG_NAMES), rng.choice(WEAK_NAMES)
        name_a, name_b = p.shown(weak, strong)
        rows.append(p.row("compassion", f"{weak}-vs-{strong}",
                          T.compassion_fade_prompt(p.q, A, B, model_a=name_a, model_b=name_b), target="R2"))

    # -- authority: our regenerated citations on the 150 items of CALM's authority file ------
    if citations is not None:
        for i, item in enumerate(authority_items):
            p = pair(item, "alignment")
            A, B = p.shown(item["answer1"], item["answer2"])
            for side in (1, 2):
                r = f"R{side}"
                for kind in authority.KINDS:
                    c = citations.get((i, side, kind))
                    if c:
                        rows.append(p.row("authority", f"{kind}-on-{r}",
                                          T.Authority_bias_prompt(p.q, A, B, reference=c, hack_answer=p.label(r)),
                                          target=r))

    # -- fact-related set: verbosity, fallacy oversight, sentiment ----------------------------
    for item in verbosity:
        p = pair(item, "fact")
        A, B = p.shown(item["answer1"], item["answer2_longer"])
        rows.append(p.row("verbosity", "R2-padded", T.evaluate_ai_responses(p.q, A, B), target="R2"))
        if p.id in padded_better:                # our addition: the better response padded the same way
            A, B = p.shown(padded_better[p.id], item["answer2"])
            rows.append(p.row("verbosity", "R1-padded", T.evaluate_ai_responses(p.q, A, B), target="R1"))
        broken = fallacy[p.id]["answer1_fallacy_oversight"]
        A, B = p.shown(broken, item["answer2"])
        rows.append(p.row("fallacy", "R1-broken", T.evaluate_ai_responses(p.q, A, B), target="R1"))
    for item in sentiment:
        p = pair(item, "fact")
        for side in (1, 2):
            for tone in TONES:
                rewritten = item.get(f"answer{side}_{tone}")
                if not rewritten:
                    continue                                       # 7 rewrites are missing in CALM's release
                r1, r2 = (rewritten, item["answer2"]) if side == 1 else (item["answer1"], rewritten)
                A, B = p.shown(r1, r2)
                rows.append(p.row("sentiment", f"{tone}-on-R{side}", T.evaluate_ai_responses(p.q, A, B),
                                  target=f"R{side}"))

    # -- one original prompt per pair, shared by all tests on it ---------------------------
    originals = []
    for p in pairs.values():
        A, B = p.shown(p.item["answer1"], p.item["answer2"])
        originals.append({"id": f"{p.id}/original/-", "item": p.id, "dataset": p.dataset,
                          "source": p.item["data_resource"], "bias": "original", "variant": "-",
                          "original": None, "r1_label": p.label("R1"), "target": None,
                          "prompt": T.evaluate_ai_responses(p.q, A, B)})
    return originals + rows


def load_prompts(path=paths.PROMPTS):
    """The released prompts (data/prompts.jsonl.gz): the rows of build_design(), without CALM's repository."""
    return [json.loads(line) for line in files.read_lines(path)]


def counts(rows):
    """Number of rows per bias."""
    out = {}
    for r in rows:
        out[r["bias"]] = out.get(r["bias"], 0) + 1
    return out


def check_counts(rows):
    """Compare with Table 2 of the paper; returns the differences (empty dict if all match)."""
    got = counts(rows)
    return {k: (got.get(k, 0), v) for k, v in EXPECTED.items() if got.get(k, 0) != v}


def smoke_sample(rows, n_items=3, seed=SEED):
    """A small subset for the smoke test: n_items pairs per bias, all their variants and their originals."""
    rng = random.Random(seed)
    keep = set()
    for bias in sorted({r["bias"] for r in rows} - {"original"}):
        items = sorted({r["item"] for r in rows if r["bias"] == bias})
        keep |= {(bias, i) for i in rng.sample(items, min(n_items, len(items)))}
    chosen = [r for r in rows if (r["bias"], r["item"]) in keep]
    needed = {r["original"] for r in chosen}
    return [r for r in rows if r["id"] in needed] + chosen
