"""A small SQLite cache for model calls.

Every successful request is stored together with the exact bytes that were sent
and received, so re-running a notebook never pays for (or waits on) the same
call twice, and `show()` can still display precisely what went over the wire.
The key covers everything that can change the answer: backend, model, state,
questions and a repeat index. The repeat index lets us ask the identical
question again on purpose, e.g. to measure run-to-run noise.

Cleaning: `Cache().clear(...)` deletes rows by model, backend, date or repeat
index. It is a dry run unless `dry_run=False` is passed, because deleted
answers have to be paid for again.
"""
import hashlib
import json
import sqlite3
import threading
import time

from .config import CACHE_DIR

COLUMNS = {   # name -> SQL type; columns missing in older cache files are added automatically
    "key": "TEXT PRIMARY KEY", "backend": "TEXT", "model": "TEXT", "request": "TEXT", "response": "TEXT",
    "snapshot": "TEXT", "latency_s": "REAL", "cost": "REAL", "created": "REAL",
    "url": "TEXT", "repeat": "INTEGER", "status_code": "INTEGER", "request_raw": "TEXT", "response_raw": "TEXT",
}


def request_key(backend, model, state, questions, repeat=0):
    payload = json.dumps([backend, model, state, questions, repeat], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


class Cache:
    def __init__(self, path=CACHE_DIR / "decisions.sqlite"):
        self.path = path
        self._lock = threading.Lock()
        # A long busy timeout and write-ahead logging let another process read the cache while a run writes to it
        # (a reading query once stopped a run with "database is locked").
        self._db = sqlite3.connect(str(path), check_same_thread=False, timeout=120)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.row_factory = sqlite3.Row
        cols = ", ".join(f'"{k}" {v}' for k, v in COLUMNS.items())
        self._db.execute(f"CREATE TABLE IF NOT EXISTS calls ({cols})")
        present = {r["name"] for r in self._db.execute("PRAGMA table_info(calls)")}
        for k, v in COLUMNS.items():
            if k not in present:
                self._db.execute(f'ALTER TABLE calls ADD COLUMN "{k}" {v}')
        self._db.commit()

    # -- reading ------------------------------------------------------------------
    def record(self, key):
        """The full stored row as a dict, or None."""
        with self._lock:
            row = self._db.execute("SELECT * FROM calls WHERE key = ?", (key,)).fetchone()
        return None if row is None else dict(row)

    def summary(self):
        """Rows, cost and date range per backend and model."""
        with self._lock:
            rows = self._db.execute(
                "SELECT backend, model, COUNT(*) AS n, ROUND(SUM(cost), 6) AS usd, "
                "MIN(created) AS first, MAX(created) AS last FROM calls GROUP BY backend, model").fetchall()
        fmt = lambda t: time.strftime("%Y-%m-%d %H:%M", time.localtime(t)) if t else None
        return [{**dict(r), "first": fmt(r["first"]), "last": fmt(r["last"])} for r in rows]

    # -- writing ------------------------------------------------------------------
    def put(self, key, *, backend, model, url, repeat, request, response, request_raw, response_raw,
            status_code, latency_s):
        usage = response.get("usage") or {}
        row = {"key": key, "backend": backend, "model": model, "request": json.dumps(request, ensure_ascii=False),
               "response": json.dumps(response, ensure_ascii=False), "snapshot": response.get("model"),
               "latency_s": latency_s, "cost": usage.get("cost") or 0.0, "created": time.time(), "url": url,
               "repeat": repeat, "status_code": status_code, "request_raw": request_raw, "response_raw": response_raw}
        names = ", ".join(f'"{k}"' for k in row)
        with self._lock:
            self._db.execute(f"INSERT OR REPLACE INTO calls ({names}) VALUES ({', '.join('?' * len(row))})",
                             list(row.values()))
            self._db.commit()

    def clear(self, model=None, backend=None, before=None, repeats_only=False, dry_run=True):
        """Delete cached calls. Filters combine with AND; no filter means everything.

        model / backend: exact match. before: a date 'YYYY-MM-DD' or a Unix time.
        repeats_only: only rows that were deliberate repeats (repeat > 0).
        Returns the number of rows that match (dry run) or were deleted.
        """
        where, args = [], []
        if model is not None:
            where.append("model = ?"); args.append(model)
        if backend is not None:
            where.append("backend = ?"); args.append(backend)
        if before is not None:
            t = time.mktime(time.strptime(before, "%Y-%m-%d")) if isinstance(before, str) else float(before)
            where.append("created < ?"); args.append(t)
        if repeats_only:
            where.append("COALESCE(repeat, 0) > 0")
        clause = (" WHERE " + " AND ".join(where)) if where else ""
        with self._lock:
            n = self._db.execute(f"SELECT COUNT(*) FROM calls{clause}", args).fetchone()[0]
            if not dry_run:
                self._db.execute(f"DELETE FROM calls{clause}", args)
                self._db.commit()
        verb = "would delete" if dry_run else "deleted"
        print(f"[tdm cache] {verb} {n} rows" + (" (pass dry_run=False to really delete)" if dry_run and n else ""))
        return n
