"""Print exactly what was sent to a model and what came back.

An *exchange* is one HTTP round trip, recorded as a dict:
    method, url, headers (API key redacted), request_raw (the exact body bytes as text),
    status_code, response_raw (the exact body text received), latency_s, time, error

`show_exchange()` pretty-prints both bodies: key order, values and number
formatting are kept as sent and received. Long text values (with line breaks or
over 80 characters) are printed as a text block below their key, with their real
line breaks, so prompts are readable. Every real line starts with `│`; lines longer
than 100 characters are wrapped for display, and the wrapped part starts with `┊`
(no line break is sent there). With
`raw=True` it prints the bodies byte for byte instead.
"""
import json
import textwrap
import time

RULE = "─" * 78


def redact(headers):
    out = {}
    for k, v in headers.items():
        if k.lower() == "authorization" and isinstance(v, str) and len(v) > 20:
            out[k] = v[:17] + "…" + v[-4:] + "  (redacted)"
        else:
            out[k] = v
    return out


def _pretty(text):
    """Re-indent a JSON text without changing keys, order or number formatting."""
    if text is None:
        return "(nothing)"
    try:
        obj = json.loads(text, parse_float=lambda s: _Raw(s), parse_int=lambda s: _Raw(s))
    except ValueError:
        return text
    return _dump(obj, 0)


class _Raw(str):
    """A number kept exactly as written in the original text."""


def _dump(o, ind):
    pad, pad2 = "  " * ind, "  " * (ind + 1)
    if isinstance(o, dict):
        if not o:
            return "{}"
        items = [f"{pad2}{json.dumps(k, ensure_ascii=False)}: {_dump(v, ind + 1)}" for k, v in o.items()]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(o, list):
        if not o:
            return "[]"
        return "[\n" + ",\n".join(pad2 + _dump(v, ind + 1) for v in o) + "\n" + pad + "]"
    if isinstance(o, _Raw):
        return str(o)
    if isinstance(o, str) and ("\n" in o or len(o) > 80):
        lines = []
        for para in o.split("\n"):
            parts = textwrap.wrap(para, width=100, replace_whitespace=False, drop_whitespace=True) or [""]
            lines += [pad + "  │ " + parts[0]] + [pad + "  ┊ " + p for p in parts[1:]]
        return "(text, %d characters)\n" % len(o) + "\n".join(lines) + "\n" + pad + "  └"
    return json.dumps(o, ensure_ascii=False)


def show_exchange(ex, raw=False, title=None):
    if ex is None:
        print("(no exchange recorded)")
        return
    when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ex["time"])) if ex.get("time") else "not sent"
    print(RULE)
    print(title or "REQUEST", f"   {ex.get('method', 'POST')} {ex.get('url')}   [{when}]")
    for k, v in (ex.get("headers") or {}).items():
        print(f"  {k}: {v}")
    print()
    print(ex.get("request_raw") if raw else _pretty(ex.get("request_raw")))
    if ex.get("sent") is False:
        print(RULE)
        return
    print(RULE)
    status = ex.get("status_code")
    lat = f"{ex['latency_s'] * 1000:.0f} ms" if ex.get("latency_s") is not None else "?"
    print("RESPONSE", f"   HTTP {status if status is not None else '-'}   {lat}"
          + (f"   ERROR: {ex['error']}" if ex.get("error") and status in (None, 200) else ""))
    print()
    print(ex.get("response_raw") if raw else _pretty(ex.get("response_raw")))
    print(RULE)
