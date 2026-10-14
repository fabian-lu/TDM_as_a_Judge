# tdm: access to typed decision models

A small client for the typed decision models on OpenRouter, used by `judgebias`. Needs Python 3.13 and an OpenRouter key in `../.env` (`OPENROUTER_API_KEY=...`).

```python
import sys; sys.path.insert(0, "..")   # the repository root
from tdm import DecisionClient, OPENROUTER, list_openrouter

jev = DecisionClient("typesafe/jev-1.13", budget_usd=2.0, concurrency=16)
q = {"q": {"type": "noul", "instructions": "Is the ticket about a payment problem?"}}

r = jev.ask({"ticket": "I was charged twice."}, q)          # one request
r["answers"]["q"]["noul"], r["snapshot"], r["cached"]

results = jev.ask_many([{"state": {...}, "questions": q}, ...])   # bulk, results in input order
# in a notebook cell you can also write:  results = await jev.ask_many_async(items)
jev.stats()                                                  # requests, cached, ok, retries, failed, spent_usd
```

## Seeing exactly what was sent and received

```python
jev.preview(state, q)          # the exact request ask() would send, nothing is sent
r = jev.ask(state, q)
jev.show(r)                    # request and response of this result, pretty-printed (API key redacted)
jev.show(r, raw=True)          # the same, byte for byte as sent and received
jev.show(r, all_attempts=True) # every attempt, if keep_exchanges="all"
jev.show_last()                # the most recent HTTP exchange of this client
```

`show()` works for fresh, cached and failed results. Pretty-printing only changes indentation; keys, order and
numbers stay exactly as sent and received. How much is kept in memory is set by `keep_exchanges`: `"last"`
(default), `"all"`, `"failed"` (recommended for very large runs; the cache keeps the raw bytes of every success
anyway) or `"none"`.

## Cleaning the cache

```python
jev.clear_cache()                          # dry run: how many rows of this model would be deleted
jev.clear_cache(dry_run=False)             # really delete this model's cached calls
jev.clear_cache(repeats_only=True, dry_run=False)   # only the deliberate repeats
from tdm.cache import Cache
Cache().summary()                          # rows, cost and dates per model
Cache().clear(before="2026-10-01", dry_run=False)   # anything older than a date, all models
```

Deleting cached calls means paying for them again, so everything is a dry run unless `dry_run=False`.

Each result is a dict: `ok`, `status` (`ok`, `transient`, `invalid`, `permanent`, `stopped`), `error`, `answers`, `snapshot` (exact dated model version), `usage`, `latency_s`, `attempts`, `cached`, `key` (cache key), `exchanges` (what `show()` prints).

## What happens under the hood

- **Cache** (`cache.py`): every successful response is stored in `../cache/decisions.sqlite` (not part of the repository), keyed by backend, model, state, questions and `repeat`. Asking again is free and instant. Pass `repeat=1, 2, ...` in an item to force a fresh identical call (e.g. to measure noise). Failures are never cached.
- **Validation** (`validate.py`): an HTTP 200 is only accepted if every question is answered with the right type, probabilities lie in [0, 1] and sum to 1 (+/- 0.03), and the chosen option exists.
- **Retries** (`client.py`): requests run concurrently (asyncio + httpx). Temporary problems (429, 5xx, timeouts, network) and invalid answers are retried with exponential backoff, respecting `Retry-After`. After `max_attempts`, the still-failing requests go into another round, and only those are re-sent, up to `max_rounds`. Permanent errors (400, 404) are reported at once. A 402 (no credits) or the budget guard stops all remaining work.
- **Progress bar**: moves as answers arrive and shows ok / failed / retries / dollars spent.

## Backends and models (`models.py`)

- `backend="openrouter"`: all OpenRouter decision models. `list_openrouter()` fetches the current list. Checked 2026-10-01: Jev 1.13, Kev-4B, Tev1-4B, Mercury Decide (free), Solar Decide, Span-01 (+ Lite, noul-only, conversations).
- `backend="ollama"`: a local Ollama >= 0.35 server (`tev1:0.8b`, `tev1`, `nimble`). Not used in the paper.

Note: `confidence` is not comparable across backends. Jev's equals (N * p_max - 1) / (N - 1); Ollama's Tev1 reports 1 minus the normalised entropy. Use the probabilities.

## ChatClient

`ChatClient(model, temperature=None, seed=None, max_tokens=None, accept=None, **kwargs)` sends chat
completions through OpenRouter with the same machinery as `DecisionClient`: concurrency, retries in
rounds, the SQLite cache (the key includes model, messages and sampling settings), the budget guard,
and `show()` / `show_last()` / `preview()` for the exact request and response. Items are
`{"messages": [...]}`; each result has the reply in `result["text"]`. `accept(text)` may return a
problem string to reject a reply, which is then retried like a malformed decision. Used to generate
the authority citations with GPT-4-Turbo and to query the LLM judges.
