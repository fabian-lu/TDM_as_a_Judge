"""Time and cost of CALM's LLM judges, for comparison with the typed decision models.

    from judgebias import llm_cost
    picked = llm_cost.sample(rows)                         # 10 prompts per test, spread over its versions
    records = llm_cost.run_judge("openai/gpt-4o-2024-05-13", picked)

Three of CALM's six judges are still served on OpenRouter (October 2026): ChatGPT
(gpt-3.5-turbo-0125), GPT-4-Turbo (2024-04-09) and GPT-4o (2024-05-13). Claude-3.5 Sonnet,
Qwen2-72B-Instruct and GLM-4 are not. Each judge receives a prompt exactly as in CALM: the
prompt string as one user message, temperature 0.7, no limit on the length of the reply. We
record tokens, cost and latency only, not the verdicts.

The input tokens of the whole design can be counted exactly with OpenAI's tokenizer
(`input_tokens`), so only the length of the replies and the latency come from the sample.
"""
import json
import random
import time

import tiktoken

from tdm import ChatClient

from . import files, paths

JUDGES = {"ChatGPT": "openai/gpt-3.5-turbo", "GPT-4-Turbo": "openai/gpt-4-turbo",
          "GPT-4o": "openai/gpt-4o-2024-05-13"}
ENCODINGS = {"openai/gpt-3.5-turbo": "cl100k_base", "openai/gpt-4-turbo": "cl100k_base",
             "openai/gpt-4o-2024-05-13": "o200k_base"}
PRICES = {   # US$ per million tokens (input, output), OpenRouter list prices on 2026-10-05
    "openai/gpt-3.5-turbo": (0.50, 1.50), "openai/gpt-4-turbo": (10.0, 30.0),
    "openai/gpt-4o-2024-05-13": (5.0, 15.0)}
TEMPERATURE = 0.7
PER_TEST = 10
SEED = 20261005
RESULTS = paths.COST_SAMPLE


def sample(rows, per_test=PER_TEST, seed=SEED):
    """`per_test` prompts of every test (the originals count as one test), taken in turn from its versions."""
    rng = random.Random(seed)
    picked = []
    for bias in sorted({r["bias"] for r in rows}):
        by_version = {}
        for r in rows:
            if r["bias"] == bias:
                by_version.setdefault(r["variant"], []).append(r)
        versions = sorted(by_version)
        for v in versions:
            rng.shuffle(by_version[v])
        i = 0
        while sum(1 for p in picked if p["bias"] == bias) < min(per_test, sum(map(len, by_version.values()))):
            pool = by_version[versions[i % len(versions)]]
            if pool:
                picked.append(pool.pop())
            i += 1
    return picked


def input_tokens(rows, model):
    """Tokens of each prompt with the judge's tokenizer (without the few tokens the chat format adds)."""
    enc = tiktoken.get_encoding(ENCODINGS[model])
    return [len(enc.encode(r["prompt"])) for r in rows]


def results_file(model, smoke=False):
    return RESULTS / (model.replace("/", "_") + ("__smoke" if smoke else "") + ".jsonl")


def run_judge(model, rows, smoke=False, concurrency=8):
    """Send the prompts to one LLM judge, save and return one record per prompt."""
    client = ChatClient(model, temperature=TEMPERATURE, keep_exchanges="failed", concurrency=concurrency,
                        budget_usd=5)
    started = time.time()
    results = client.ask_many([{"messages": [{"role": "user", "content": r["prompt"]}]} for r in rows], desc=model)
    wall = time.time() - started
    records = []
    for r, res in zip(rows, results):
        usage = res["usage"] or {}
        records.append({"id": r["id"], "bias": r["bias"], "variant": r["variant"], "model": model, "ok": res["ok"],
                        "error": res["error"], "latency_s": res["latency_s"], "cached": res["cached"],
                        "input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens"),
                        "cost": usage.get("cost"), "snapshot": res["snapshot"], "reply": res["text"]})
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


def load_records(model, smoke=False):
    path = results_file(model, smoke)
    return [json.loads(line) for line in files.read_lines(path)]
