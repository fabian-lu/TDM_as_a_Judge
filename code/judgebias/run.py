"""Sending the design to the typed decision models and storing their answers.

    from judgebias import run
    records = run.run_model("typesafe/jev-1.13", "native", rows)       # rows from design.build_design()

Each design row becomes one request in the chosen format. Original prompts are sent a
second time (repeat=1), which gives the consistency rate and the noise floor. Results are
written to data/answers/tdms/<model>__<format>.jsonl, one line per request, with the latency,
cost and tokens of that request. Each run also adds one line to data/answers/tdms/run_log.jsonl
(start, wall-clock time, number of requests, concurrency, total cost), so the paper can
report how long and how much a full bias test takes per model.
"""
import json
import time
from pathlib import Path

from tdm import DecisionClient

from . import files, paths
from .calm import combine_decomposed, to_decomposed, to_native, to_typed

RESULTS = paths.TDM_ANSWERS

FORMATS = {"calm": to_typed, "native": to_native, "decomposed": to_decomposed}

# Typed decision models on OpenRouter's Decisions API, list frozen on 2026-10-04.
# Span-01 is left out: it only answers yes/no questions about conversations.
MODELS = [
    "typesafe/jev-1.13",
    "jaredpalmer/kev-4b",
    "togethercomputer/tev1-4b-experimental",
    "inception/mercury-decide:free",
    "upstage/solar-decide",
    "liquid/d1",
    "perplexity/pplx-decider-v1.1-27b",      # v1 (pplx-decider-v1-27b) left OpenRouter on 2026-10-07; v1.1 answers every test
    "cloudflare/clef",
    "cloudflare/clef-flash",
]


def requests(rows, fmt):
    """[(row, repeat)] for every request: each row once, original prompts a second time."""
    out = [(r, 0) for r in rows]
    out += [(r, 1) for r in rows if r["bias"] == "original"]
    return out


def to_request(row, fmt, repeat):
    state, questions = FORMATS[fmt](row["prompt"])
    return {"state": state, "questions": questions, "repeat": repeat}


def read_answer(result, fmt):
    """(choice, P(A), per-question P(A) or None) from one result. Choice is None if it failed or tied."""
    if not result["ok"]:
        return None, None, None
    answers = result["answers"]
    if fmt == "decomposed":
        p_a, choice = combine_decomposed(answers)
        return choice, p_a, {k: v["probabilities"]["A"] for k, v in answers.items()}
    verdict = answers["verdict"]
    return verdict["choice"], verdict["probabilities"]["A"], None


def cached_answers(client, row, fmt, repeat=0):
    """The answers of one request exactly as the API returned them, from the cache (no API call)."""
    hit = client._from_cache(to_request(row, fmt, repeat))
    return hit["answers"] if hit else None


def results_file(model, fmt, smoke=False):
    name = model.replace("/", "_").replace(":", "_") + f"__{fmt}" + ("__smoke" if smoke else "") + ".jsonl"
    return RESULTS / name


def run_model(model, fmt, rows, smoke=False, **client_kwargs):
    """Send all requests of `rows` to one model in one format, save and return the records."""
    client = DecisionClient(model, keep_exchanges="failed", **client_kwargs)
    todo = requests(rows, fmt)
    started = time.time()
    results = client.ask_many([to_request(r, fmt, rep) for r, rep in todo], desc=f"{model} [{fmt}]")
    finished = time.time()
    records = []
    for (row, rep), res in zip(todo, results):
        choice, p_a, per_question = read_answer(res, fmt)
        records.append({"id": row["id"], "repeat": rep, "model": model, "format": fmt, "ok": res["ok"],
                        "status": res["status"], "error": res["error"], "choice": choice, "p_A": p_a,
                        "p_A_questions": per_question, "snapshot": res["snapshot"],
                        "latency_s": res["latency_s"], "cost": (res["usage"] or {}).get("cost"),
                        "input_tokens": (res["usage"] or {}).get("input_tokens"),
                        "output_tokens": (res["usage"] or {}).get("output_tokens"), "cached": res["cached"]})
    path = results_file(model, fmt, smoke)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log = {"model": model, "format": fmt, "requests": len(records), "ok": sum(r["ok"] for r in records),
           "sent_now": len(records) - sum(r["cached"] for r in records), "concurrency": client.concurrency,
           "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
           "wall_s": round(finished - started, 1), "usd": round(sum(r["cost"] or 0 for r in records), 6)}
    with open(RESULTS / ("run_log__smoke.jsonl" if smoke else "run_log.jsonl"), "a") as f:
        f.write(json.dumps(log) + "\n")
    return records, client


def load_records(model, fmt, smoke=False):
    path = results_file(model, fmt, smoke)
    return [json.loads(line) for line in files.read_lines(path)]
