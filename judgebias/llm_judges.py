"""LLM judges on the same prompts as the typed decision models (notebook 08_llm_judges).

    from judgebias import llm_judges as L
    rows = design.build_design(citations=authority.load_citations())
    picked = L.smoke_sample(rows)                       # a few prompts per test, to estimate time and cost
    records = L.run("openai/gpt-4o-2024-05-13", picked, smoke=True)
    L.estimate(records)                                  # cost and time of the full design from the sample

Every judge receives CALM's prompt as one user message, as in CALM, but at temperature 0, so that it answers the
same prompt the same way, like the typed decision models. Its reply ends with "[[A]]" or "[[B]]" (`verdict`).
Hidden reasoning ("thinking") is switched off where the provider allows it: CALM's prompt already asks for a written
explanation before the verdict, and thinking tokens multiply cost and time. The
full design sends every perturbed prompt once and every original prompt twice (repeat 0 and 1), 14,695 requests.
"""
import json
import os
import re
import time
import urllib.request

import numpy as np

from tdm import ChatClient

from . import files, llm_cost
from . import run as design_run
from .run import DATA

# label: (OpenRouter id, generation, reasoning setting). The 2024 judges are those of CALM that are still served (llm_cost.JUDGES);
# the 2026 judges are current models of the same providers, the base LLM of two typed models (Qwen3.8-27B) and a
# cheap current model. GLM-5.3 was in the first smoke run and dropped (slow, expensive with its long thinking).
# Reasoning is switched off where the endpoint allows it; three endpoints make it mandatory ("Reasoning is
# mandatory for this endpoint and cannot be disabled", smoke run of 2026-10-08), so they get the lowest effort.
OFF, LOW = {"enabled": False}, {"effort": "low"}
MODELS = {
    "ChatGPT": ("openai/gpt-3.5-turbo", "2024", OFF),
    "GPT-4o": ("openai/gpt-4o-2024-05-13", "2024", OFF),
    "GPT-6.1 Sol": ("openai/gpt-6.1-sol", "2026", LOW),
    "Claude Sonnet 5.5": ("anthropic/claude-sonnet-5.5", "2026", LOW),
    "Gemini 3.8 Flash": ("google/gemini-3.8-flash", "2026", LOW),
    "Qwen3.8-27B": ("qwen/qwen3.8-27b", "2026", OFF),
    "Claude Haiku 5.5": ("anthropic/claude-haiku-5.5", "2026", OFF),
}
TEMPERATURE = 0
# Whether a judge really ran at TEMPERATURE. OpenRouter drops parameters an endpoint does not support, and the
# endpoints of GPT-6.1 Sol and Claude Haiku 5.5 list no temperature; Anthropic's API rejects it for Claude Sonnet 5.5
# ("`temperature` is deprecated for this model"). These three ran at their provider's default.
TEMPERATURE_APPLIED = {"ChatGPT": True, "GPT-4o": True, "GPT-6.1 Sol": False, "Claude Sonnet 5.5": False,
                       "Gemini 3.8 Flash": True, "Qwen3.8-27B": True, "Claude Haiku 5.5": False}
# Upper limit on the reply. OpenRouter reserves credit for the longest possible reply of every request in flight, which
# stopped Gemini at 753 of 14,695 requests ("would exceed your available credits", 2026-10-09). Replies have 200 to 800
# tokens, so 4,000 never cuts one. Set for the judges run after that day; the first three ran without a limit, and
# changing it for them would only send their requests again.
MAX_TOKENS = {"Gemini 3.8 Flash": 4000, "Claude Haiku 5.5": 4000, "GPT-4o": 4000, "Claude Sonnet 5.5": 4000}
KEY_ENV = "OPENROUTER_API_KEY"     # the OpenRouter key for the LLM judges (in .env)
FULL_DESIGN = 15_195              # requests per judge: 13,251 perturbed prompts + 972 originals sent twice
RESULTS = DATA / "llm_judges"


def smoke_sample(rows, per_test=3):
    """`per_test` prompts of every test and of the originals, spread over the versions (as in llm_cost)."""
    return llm_cost.sample(rows, per_test=per_test)


def verdict(text):
    """'A' or 'B' from the last [[A]] or [[B]] in the reply, None if there is none."""
    found = re.findall(r"\[\[\s*([AB])\s*\]\]", text or "")
    return found[-1] if found else None


def results_file(model, smoke=False):
    return RESULTS / (model.replace("/", "_") + ("__smoke" if smoke else "") + ".jsonl")


def run(model, rows, repeats=None, smoke=False, concurrency=8, budget_usd=5, reasoning=OFF, key_env=None,
        max_tokens=None):
    """Send the prompts to one LLM judge and save one record per request. `repeats` gives the repeat number of
    each row (0, or 1 for the second sending of an original); default 0 for all."""
    repeats = repeats or [0] * len(rows)
    client = ChatClient(model, temperature=TEMPERATURE, max_tokens=max_tokens, keep_exchanges="failed",
                        concurrency=concurrency, budget_usd=budget_usd)
    if reasoning is not None:
        client.params["reasoning"] = reasoning
    if key_env:                                  # pay with another OpenRouter key than OPENROUTER_API_KEY
        client.headers["Authorization"] = "Bearer " + os.environ[key_env]
    started = time.time()
    results = client.ask_many([{"messages": [{"role": "user", "content": r["prompt"]}], "repeat": k}
                               for r, k in zip(rows, repeats)], desc=model)
    wall = time.time() - started
    records = []
    for r, k, res in zip(rows, repeats, results):
        usage = res["usage"] or {}
        details = usage.get("completion_tokens_details") or {}
        records.append({"id": r["id"], "repeat": k, "bias": r["bias"], "variant": r["variant"], "model": model,
                        "ok": res["ok"], "error": res["error"], "latency_s": res["latency_s"], "cached": res["cached"],
                        "input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens"),
                        "reasoning_tokens": details.get("reasoning_tokens"), "cost": usage.get("cost"),
                        "snapshot": res["snapshot"], "verdict": verdict(res["text"]), "reply": res["text"]})
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(results_file(model, smoke), "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(RESULTS / ("run_log__smoke.jsonl" if smoke else "run_log.jsonl"), "a") as f:
        f.write(json.dumps({"model": model, "requests": len(records), "ok": sum(x["ok"] for x in records),
                            "sent_now": sum(not x["cached"] for x in records), "concurrency": concurrency,
                            "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
                            "wall_s": round(wall, 1), "usd": round(sum(x["cost"] or 0 for x in records), 6)}) + "\n")
    return records


def run_full(label, rows, budget_usd, key_env=KEY_ENV, concurrency=32):
    """The full design for one judge of MODELS: every perturbed prompt once, every original twice. Answers already
    in the cache are not sent again, so a run that stopped (budget, key limit, crash) continues where it ended."""
    model, _, reasoning = MODELS[label]
    reqs = design_run.requests(rows, None)
    return run(model, [r for r, _ in reqs], repeats=[k for _, k in reqs], concurrency=concurrency,
               budget_usd=budget_usd, reasoning=reasoning, key_env=key_env, max_tokens=MAX_TOKENS.get(label))


def key_status(key_env=KEY_ENV):
    """Limit, remaining limit and usage of an OpenRouter key (no key material is returned)."""
    req = urllib.request.Request("https://openrouter.ai/api/v1/key",
                                 headers={"Authorization": "Bearer " + os.environ[key_env]})
    d = json.load(urllib.request.urlopen(req, timeout=30))["data"]
    return {k: d.get(k) for k in ["limit", "limit_remaining", "usage"]}


def load_records(model, smoke=False):
    path = results_file(model, smoke)
    return [json.loads(line) for line in files.read_lines(path)]


def estimate(records, n=FULL_DESIGN, concurrency=(16, 32)):
    """Cost and time of `n` requests, extrapolated from a sample of records of one judge. Time assumes that
    `concurrency` requests run in parallel without rate limits, so it is a lower bound."""
    ok = [r for r in records if r["ok"]]
    cost = [r["cost"] for r in ok if r["cost"] is not None]
    lat = [r["latency_s"] for r in ok if r["latency_s"] is not None]
    out = {"sample": len(records), "ok": len(ok), "verdict_found": sum(r["verdict"] is not None for r in ok),
           "input_tokens": np.mean([r["input_tokens"] for r in ok if r["input_tokens"]]) if ok else np.nan,
           "output_tokens": np.mean([r["output_tokens"] for r in ok if r["output_tokens"]]) if ok else np.nan,
           "reasoning_tokens": np.mean([r["reasoning_tokens"] or 0 for r in ok]) if ok else np.nan,
           "usd_sample": sum(cost), "usd_per_1000": 1000 * np.mean(cost) if cost else np.nan,
           "usd_full_design": n * np.mean(cost) if cost else np.nan,
           "median_latency_s": float(np.median(lat)) if lat else np.nan}
    for c in concurrency:
        out[f"hours_full_at_{c}"] = n * np.mean(lat) / c / 3600 if lat else np.nan
    return out


# -- Claude Sonnet 5.5 directly on Anthropic's API -------------------------------------------------------------------
# Thinking is off (`between_tools`); the model accepts no temperature
# ("`temperature` is deprecated for this model"), as OpenRouter's list also shows for GPT-6.1 Sol and Claude Haiku 5.5.
# `run_anthropic` sends the requests directly (about an hour for the full design); the batch functions below run them
# at half price but within up to 24 hours, which was too slow for the deadline (a 30-request test batch sat unprocessed
# for 7 minutes on 2026-10-09 and was cancelled).
ANTHROPIC_KEY_ENV = "ANTHROPIC_API_KEY"
BATCH_MODELS = {"Claude Sonnet 5.5": "claude-sonnet-5-5"}
PRICE = {"claude-sonnet-5-5": (2.0, 10.0)}           # US$ per million tokens (input, output), list price
BATCH_PRICE = {"claude-sonnet-5-5": (1.0, 5.0)}      # batch = half the list price
BATCHES = RESULTS / "batches"


def run_anthropic(label, rows, repeats=None, smoke=False, concurrency=16):
    """Send the prompts directly to Anthropic's API. Every answer is appended to the results file as it arrives, and a
    re-run skips the requests already answered, so an interrupted run continues where it stopped."""
    import asyncio
    import concurrent.futures
    import anthropic
    model = BATCH_MODELS[label]
    repeats = repeats or [0] * len(rows)
    path = results_file(model, smoke)
    done = {}
    if path.exists():
        for line in open(path):
            rec = json.loads(line)
            if rec["ok"]:
                done[(rec["id"], rec["repeat"])] = rec
    todo = [(r, k) for r, k in zip(rows, repeats) if (r["id"], k) not in done]
    price_in, price_out = PRICE[model]
    RESULTS.mkdir(parents=True, exist_ok=True)
    started = time.time()

    async def one(client, sem, out, r, k):
        rec = {"id": r["id"], "repeat": k, "bias": r["bias"], "variant": r["variant"], "model": model, "ok": False,
               "error": None, "latency_s": None, "cached": False, "input_tokens": None, "output_tokens": None,
               "reasoning_tokens": None, "cost": None, "snapshot": None, "verdict": None, "reply": None}
        async with sem:
            t0 = time.monotonic()
            try:
                msg = await client.messages.create(model=model, max_tokens=4000, thinking={"type": "between_tools"},
                                                   messages=[{"role": "user", "content": r["prompt"]}])
                text = "".join(b.text for b in msg.content if b.type == "text")
                refusal = msg.stop_reason == "refusal"
                rec.update(ok=True, error=f"refusal: {msg.stop_details.category}" if refusal else None,
                           input_tokens=msg.usage.input_tokens, output_tokens=msg.usage.output_tokens,
                           cost=(msg.usage.input_tokens * price_in + msg.usage.output_tokens * price_out) / 1e6,
                           snapshot=msg.model, verdict=None if refusal else verdict(text), reply=text)
            except anthropic.APIStatusError as e:
                rec["error"] = f"HTTP {e.status_code}: {str(e.message)[:300]}"
            except anthropic.APIConnectionError as e:
                rec["error"] = f"connection: {e}"
            rec["latency_s"] = time.monotonic() - t0
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out.flush()
        return rec

    async def main():
        sem = asyncio.Semaphore(concurrency)
        async with anthropic.AsyncAnthropic(api_key=os.environ[ANTHROPIC_KEY_ENV], max_retries=8) as client:
            with open(path, "a") as out:
                return await asyncio.gather(*(one(client, sem, out, r, k) for r, k in todo))

    with concurrent.futures.ThreadPoolExecutor(1) as ex:       # works inside a notebook's running event loop
        new = ex.submit(asyncio.run, main()).result()
    records = {(x["id"], x["repeat"]): x for x in [json.loads(l) for l in open(path)] if x["ok"]}
    records = [records.get((r["id"], k)) or next(x for x in new if x["id"] == r["id"] and x["repeat"] == k)
               for r, k in zip(rows, repeats)]
    with open(path, "w") as f:                                  # one line per request, in design order
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(RESULTS / ("run_log__smoke.jsonl" if smoke else "run_log.jsonl"), "a") as f:
        f.write(json.dumps({"model": model, "requests": len(records), "ok": sum(x["ok"] for x in records),
                            "sent_now": len(todo), "concurrency": concurrency,
                            "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
                            "wall_s": round(time.time() - started, 1),
                            "usd": round(sum(x["cost"] or 0 for x in new if x["ok"]), 6)}) + "\n")
    return records


def _anthropic():
    import anthropic
    return anthropic.Anthropic(api_key=os.environ[ANTHROPIC_KEY_ENV])


def batch_submit(label, rows, repeats=None, smoke=False):
    """Send the prompts as one batch and save which request (custom_id) belongs to which prompt. Returns the batch id."""
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    model = BATCH_MODELS[label]
    repeats = repeats or [0] * len(rows)
    requests = [Request(custom_id=f"q{i}", params=MessageCreateParamsNonStreaming(
                    model=model, max_tokens=4000, thinking={"type": "between_tools"},
                    messages=[{"role": "user", "content": r["prompt"]}]))
                for i, r in enumerate(rows)]
    batch = _anthropic().messages.batches.create(requests=requests)
    BATCHES.mkdir(parents=True, exist_ok=True)
    meta = {"batch_id": batch.id, "label": label, "model": model, "smoke": smoke,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "requests": [{"custom_id": f"q{i}", "id": r["id"], "repeat": k, "bias": r["bias"], "variant": r["variant"]}
                         for i, (r, k) in enumerate(zip(rows, repeats))]}
    (BATCHES / f"{batch.id}.json").write_text(json.dumps(meta))
    return batch.id


def batch_status(batch_id):
    """Processing status and request counts of a batch."""
    b = _anthropic().messages.batches.retrieve(batch_id)
    c = b.request_counts
    return {"status": b.processing_status, "processing": c.processing, "succeeded": c.succeeded,
            "errored": c.errored, "canceled": c.canceled, "expired": c.expired}


def batch_collect(batch_id):
    """Download the results of an ended batch and save them in the format of `run`. A refusal counts as answered
    without a verdict; another model never stands in for it."""
    meta = json.loads((BATCHES / f"{batch_id}.json").read_text())
    by_id = {m["custom_id"]: m for m in meta["requests"]}
    price_in, price_out = BATCH_PRICE[meta["model"]]
    records = []
    for res in _anthropic().messages.batches.results(batch_id):
        m = by_id[res.custom_id]
        rec = {"id": m["id"], "repeat": m["repeat"], "bias": m["bias"], "variant": m["variant"], "model": meta["model"],
               "ok": False, "error": res.result.type, "latency_s": None, "cached": False, "input_tokens": None,
               "output_tokens": None, "reasoning_tokens": None, "cost": None, "snapshot": None, "verdict": None,
               "reply": None, "batch_id": batch_id, "custom_id": res.custom_id}
        if res.result.type == "succeeded":
            msg = res.result.message
            text = "".join(b.text for b in msg.content if b.type == "text")
            refusal = msg.stop_reason == "refusal"
            rec.update(ok=True, error=f"refusal: {msg.stop_details.category}" if refusal else None,
                       input_tokens=msg.usage.input_tokens, output_tokens=msg.usage.output_tokens,
                       cost=(msg.usage.input_tokens * price_in + msg.usage.output_tokens * price_out) / 1e6,
                       snapshot=msg.model, verdict=None if refusal else verdict(text), reply=text)
        records.append(rec)
    records.sort(key=lambda r: int(r["custom_id"][1:]))
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(results_file(meta["model"], meta["smoke"]), "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(RESULTS / ("run_log__smoke.jsonl" if meta["smoke"] else "run_log.jsonl"), "a") as f:
        f.write(json.dumps({"model": meta["model"], "requests": len(records), "ok": sum(x["ok"] for x in records),
                            "batch_id": batch_id, "submitted": meta["created"],
                            "collected": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "usd": round(sum(x["cost"] or 0 for x in records), 6)}) + "\n")
    return records


# -- Analysis --------------------------------------------------------------------------------------------------------
def file_model(label):
    """The model name under which a judge's answers are saved (Sonnet ran on Anthropic's API, the others on OpenRouter)."""
    return BATCH_MODELS.get(label) or MODELS[label][0]


def as_answers(records):
    """LLM records in the form that `metrics.table` reads for the typed models: the letter the judge picked as `choice`
    and P(A) as 1 or 0. A request without a verdict (failed, refused, or no [[A]] or [[B]] in the reply) has no choice,
    like a 50/50 answer of a typed model."""
    return [{"id": x["id"], "repeat": x["repeat"], "choice": x["verdict"] if x["ok"] else None,
             "p_A": float(x["verdict"] == "A") if x["ok"] and x["verdict"] else None} for x in records]


def table(label, rows):
    """`metrics.table` for one LLM judge, with our two additions as tests of their own. An LLM judge gives a verdict,
    not a probability, so the probability columns (and with them the shifts) stay empty."""
    from . import metrics
    df = metrics.separate_additions(metrics.table(rows, as_answers(load_records(file_model(label)))))
    df[["p", "p_rep", "p_hat"]] = np.nan
    return df
