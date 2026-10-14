"""One client for every typed decision model we use.

    from tdm import DecisionClient
    jev = DecisionClient("typesafe/jev-1.13")
    r = jev.ask(state={"ticket": "..."}, questions={"q": {"type": "noul", "instructions": "..."}})
    results = jev.ask_many([{"state": ..., "questions": ...}, ...])      # bulk, in input order

In a notebook, `await jev.ask_many_async(items)` does the same without a helper thread.

Backends share the TypeSafe request format (state + typed questions):
  * "openrouter": OpenRouter Decisions API (Jev, Kev, Tev1, Mercury Decide, ...)
  * "ollama":     a local Ollama >= 0.35 server (/v1/systemone)

How bulk requests are handled
  * Requests run concurrently (asyncio + httpx), at most `concurrency` at a time.
  * Every response is checked (validate.py). Outcomes fall into four classes:
      ok         valid answer, cached
      transient  rate limit, server error, timeout, network problem  -> retried
      invalid    HTTP 200 but answers missing or malformed             -> retried
      permanent  e.g. 400 bad request, 404 unknown model               -> reported, not retried
    A 402 (out of credits) or the budget guard stops all remaining work.
  * Each request gets `max_attempts` tries with exponential backoff (respecting
    Retry-After). Whatever still fails goes into the next *round*, which only
    re-sends the failures after a pause, until everything is answered or
    `max_rounds` is reached.
  * After a 429 all workers pause together for a moment (shared cooldown).
  * Failed answers are never cached; cached answers are free and instant.

Seeing exactly what was sent and received
  * Each request body is serialised once and those exact bytes are sent.
  * Every result carries its exchange(s): request bytes, response text, status,
    timing (also for failures and for calls made with use_cache=False).
    `keep_exchanges` controls how much is kept in memory: "last" (default, last
    attempt), "all" (every attempt), "failed" (only for failed results; use this
    for very large runs, the cache keeps the raw bytes of successes anyway), "none".
  * client.show(result)        pretty-print request and response of a result
    client.show_last()         the most recent HTTP exchange of this client
    client.preview(state, q)   what would be sent, without sending
    client.clear_cache(...)    delete this model's cached calls (dry run by default)
"""
import asyncio
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
from tqdm.auto import tqdm

from .cache import Cache, request_key
from .config import OLLAMA_URL, OPENROUTER_URL, openrouter_key
from .show import redact, show_exchange
from .validate import check

TRANSIENT_HTTP = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}
FATAL_HTTP = {402}


class BudgetExceeded(RuntimeError):
    pass


def _run_sync(coro):
    """Run a coroutine from normal code, also inside Jupyter where a loop is already running."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()


class DecisionClient:
    def __init__(self, model, backend="openrouter", budget_usd=5.0, concurrency=8, use_cache=True,
                 timeout=120.0, max_attempts=4, max_rounds=5, round_pause_s=5.0, keep_exchanges="last"):
        self.model = model
        self.backend = backend
        self.budget_usd = budget_usd
        self.concurrency = concurrency
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.max_rounds = max_rounds
        self.round_pause_s = round_pause_s
        self.keep_exchanges = keep_exchanges
        self.last = None                      # most recent HTTP exchange
        self.cache = Cache() if use_cache else None
        if backend == "openrouter":
            self.url = OPENROUTER_URL
            self.headers = {"Authorization": "Bearer " + openrouter_key(), "Content-Type": "application/json"}
        elif backend == "ollama":
            self.url = OLLAMA_URL
            self.headers = {"Content-Type": "application/json"}
        else:
            raise ValueError(f"unknown backend {backend!r}")
        self.spent = 0.0
        self.counts = {"requests": 0, "cached": 0, "ok": 0, "retries": 0, "failed": 0}
        self.stop_reason = None
        self._cooldown_until = 0.0

    # -- public API -----------------------------------------------------------------
    def ask(self, state, questions, repeat=0):
        """One request; same result format as ask_many."""
        return self.ask_many([{"state": state, "questions": questions, "repeat": repeat}], progress=False)[0]

    def ask_many(self, items, desc=None, progress=True):
        """items: list of dicts with state, questions and optionally repeat. Returns results in input order."""
        return _run_sync(self.ask_many_async(items, desc=desc, progress=progress))

    async def ask_many_async(self, items, desc=None, progress=True):
        results = [None] * len(items)
        pending = []
        for i, it in enumerate(items):                       # cache hits first: free and instant
            hit = self._from_cache(it)
            if hit is not None:
                results[i] = hit
            else:
                pending.append(i)

        bar = tqdm(total=len(items), initial=len(items) - len(pending), desc=desc or self.model,
                   disable=not progress, leave=True)
        sem = asyncio.Semaphore(self.concurrency)
        limits = httpx.Limits(max_connections=self.concurrency, max_keepalive_connections=self.concurrency)
        async with httpx.AsyncClient(timeout=self.timeout, limits=limits) as http:
            for rnd in range(1, self.max_rounds + 1):
                if not pending or self.stop_reason:
                    break
                if rnd > 1:
                    bar.set_description(f"{desc or self.model} (round {rnd}, {len(pending)} left)")
                    await asyncio.sleep(self.round_pause_s * 2 ** (rnd - 2))
                outcomes = await asyncio.gather(*(self._one(http, sem, items[i]) for i in pending))
                still = []
                for i, res in zip(pending, outcomes):
                    results[i] = res
                    if res["ok"] or res["status"] in ("permanent", "stopped"):
                        bar.update(1)
                    else:
                        still.append(i)
                    bar.set_postfix(ok=self.counts["ok"] + self.counts["cached"], failed=self.counts["failed"],
                                    retries=self.counts["retries"], usd=f"{self.spent:.4f}")
                pending = still
        bar.close()
        self.counts["failed"] += len(pending)
        if pending:
            print(f"[tdm] {len(pending)} of {len(items)} requests still failing after {self.max_rounds} rounds; "
                  f"last error: {results[pending[0]]['error']}")
        if self.stop_reason:
            print(f"[tdm] stopped early: {self.stop_reason}")
        return results

    def stats(self):
        return {"model": self.model, "backend": self.backend, **self.counts, "spent_usd": round(self.spent, 6),
                "stop_reason": self.stop_reason}

    def show(self, result, raw=False, all_attempts=False):
        """Print exactly what was sent and received for one result (fresh, cached or failed)."""
        exs = result.get("exchanges") or []
        if not exs:
            print("(no exchange kept for this result; see keep_exchanges)")
            return
        if result.get("cached"):
            print("[served from cache; this is the original exchange from "
                  + (exs[-1].get("time_str") or "an earlier run") + "]")
        for i, ex in enumerate(exs if all_attempts else exs[-1:]):
            title = f"REQUEST (attempt {ex['attempt']})" if ex.get("attempt") else "REQUEST"
            show_exchange(ex, raw=raw, title=title)

    def show_last(self, raw=False):
        show_exchange(self.last, raw=raw)

    def preview(self, state, questions, raw=False):
        """Print the exact request that ask() would send, without sending it."""
        body = {"model": self.model, "state": state, "questions": questions}
        show_exchange({"method": "POST", "url": self.url, "headers": redact(self.headers),
                       "request_raw": self._serialize(body), "sent": False},
                      raw=raw, title="REQUEST (preview, not sent)")

    def clear_cache(self, dry_run=True, **filters):
        """Delete this model's cached calls; filters as in Cache.clear (before=, repeats_only=)."""
        if self.cache is None:
            return 0
        return self.cache.clear(model=self.model, backend=self.backend, dry_run=dry_run, **filters)

    # -- internals (ChatClient overrides _body, _check, _key and _result) -------------
    def _body(self, it):
        """The JSON body sent for one item."""
        return {"model": self.model, "state": it["state"], "questions": it["questions"]}

    def _check(self, it, response):
        """None if the response is a valid answer to the item, else the reason."""
        return check(it["questions"], response)

    def _key(self, it):
        return request_key(self.backend, self.model, it["state"], it["questions"], it.get("repeat", 0))

    def _from_cache(self, it):
        if self.cache is None:
            return None
        key = self._key(it)
        rec = self.cache.record(key)
        if rec is None:
            return None
        self.counts["cached"] += 1
        reconstructed = not rec.get("request_raw")
        ex = {"method": "POST", "url": rec.get("url") or self.url, "headers": redact(self.headers),
              "request_raw": rec.get("request_raw") or rec["request"],
              "response_raw": rec.get("response_raw") or rec["response"],
              "status_code": rec.get("status_code") or 200, "latency_s": rec["latency_s"], "time": rec["created"],
              "time_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(rec["created"])),
              "reconstructed": reconstructed, "error": None}
        keep = [ex] if self.keep_exchanges in ("last", "all") else []
        return self._result("ok", json.loads(rec["response"]), rec["latency_s"], attempts=0, cached=True,
                            key=key, exchanges=keep)

    async def _one(self, http, sem, it):
        """Up to max_attempts tries for one request. Returns a result dict."""
        body = self._body(it)
        key = self._key(it)
        status, error, latency = "transient", None, None
        exchanges = []

        def done(status_, response_, attempts_, error_=None):
            if self.keep_exchanges == "all" or (self.keep_exchanges == "failed" and status_ != "ok"):
                kept = exchanges
            elif self.keep_exchanges == "last":
                kept = exchanges[-1:]
            else:
                kept = []
            return self._result(status_, response_, latency, attempts_, error=error_, key=key, exchanges=kept)

        for attempt in range(1, self.max_attempts + 1):
            if self.stop_reason:
                return done("stopped", None, attempt - 1, self.stop_reason)
            if self.spent >= self.budget_usd:
                self.stop_reason = f"budget of ${self.budget_usd} reached (spent ${self.spent:.4f})"
                return done("stopped", None, attempt - 1, self.stop_reason)
            wait = self._cooldown_until - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            async with sem:
                # check again once it is this request's turn: all requests are queued at the start, so the check
                # above alone let a whole run pass the budget (2026-10-09)
                if self.stop_reason or self.spent >= self.budget_usd:
                    self.stop_reason = self.stop_reason or f"budget of ${self.budget_usd} reached (spent ${self.spent:.4f})"
                    return done("stopped", None, attempt - 1, self.stop_reason)
                t0 = time.monotonic()
                status, response, error, retry_after, ex = await self._post(http, body, it)
                latency = time.monotonic() - t0
            ex["latency_s"], ex["attempt"] = latency, attempt
            exchanges.append(ex)
            self.last = ex
            self.counts["requests"] += 1
            if response is not None:
                self.spent += (response.get("usage") or {}).get("cost") or 0.0
            if status == "ok":
                self.counts["ok"] += 1
                if self.cache is not None:
                    self.cache.put(key, backend=self.backend, model=self.model, url=self.url,
                                   repeat=it.get("repeat", 0), request=body, response=response,
                                   request_raw=ex["request_raw"], response_raw=ex["response_raw"],
                                   status_code=ex["status_code"], latency_s=latency)
                return done("ok", response, attempt)
            if status == "fatal":
                self.stop_reason = error
                self.counts["failed"] += 1
                return done("stopped", None, attempt, error)
            if status == "permanent":
                self.counts["failed"] += 1
                return done("permanent", None, attempt, error)
            # transient or invalid: back off and try again
            self.counts["retries"] += 1
            delay = retry_after if retry_after else min(60.0, 2 ** (attempt - 1)) * (0.5 + random.random())
            if retry_after:
                self._cooldown_until = max(self._cooldown_until, time.monotonic() + retry_after)
            await asyncio.sleep(delay)
        return done(status, None, self.max_attempts, error)

    @staticmethod
    def _serialize(body):
        return json.dumps(body, ensure_ascii=False, separators=(",", ":"))

    async def _post(self, http, body, it):
        """Returns (status, response, error, retry_after_seconds, exchange)."""
        raw = self._serialize(body)
        ex = {"method": "POST", "url": self.url, "headers": redact(self.headers), "request_raw": raw,
              "status_code": None, "response_raw": None, "time": time.time(), "error": None}
        try:
            r = await http.post(self.url, content=raw.encode("utf-8"), headers=self.headers)
        except (httpx.TimeoutException, httpx.TransportError) as e:
            ex["error"] = f"{type(e).__name__}: {e}"
            return "transient", None, ex["error"], None, ex
        ex["status_code"], ex["response_raw"] = r.status_code, r.text
        if r.status_code == 200:
            try:
                response = r.json()
            except ValueError:
                ex["error"] = "response is not JSON"
                return "invalid", None, ex["error"], None, ex
            problem = self._check(it, response)
            ex["error"] = problem
            return ("ok", response, None, None, ex) if problem is None else ("invalid", response, problem, None, ex)
        error = f"HTTP {r.status_code}: {r.text[:300]}"
        ex["error"] = error
        if r.status_code == 402 and "in_flight_budget_exhausted" in r.text:
            # OpenRouter holds credit for every request in flight until its cost settles. With many fast requests the
            # holds can reach the key's limit although credit is left; wait like after a rate limit (2026-10-09).
            self._cooldown_until = max(self._cooldown_until, time.monotonic() + 20.0)
            return "transient", None, error, 20.0, ex
        if r.status_code in FATAL_HTTP:
            return "fatal", None, error, None, ex
        if r.status_code in TRANSIENT_HTTP:
            retry_after = None
            try:
                retry_after = float(r.headers.get("retry-after"))
            except (TypeError, ValueError):
                pass
            if r.status_code == 429:
                self._cooldown_until = max(self._cooldown_until, time.monotonic() + (retry_after or 2.0))
            return "transient", None, error, retry_after, ex
        return "permanent", None, error, None, ex

    @staticmethod
    def _result(status, response, latency, attempts, cached=False, error=None, key=None, exchanges=None):
        ok = status == "ok"
        return {"ok": ok, "status": status, "error": error,
                "answers": response.get("answers") if ok else None,
                "snapshot": response.get("model") if ok else None,
                "usage": response.get("usage") if ok else None,
                "latency_s": latency, "attempts": attempts, "cached": cached,
                "key": key, "exchanges": exchanges or []}
