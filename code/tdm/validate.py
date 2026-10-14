"""Check that a response really answers every question we asked.

An HTTP 200 is not enough: a response can lack a question, return the wrong
type, or contain probabilities that are not a distribution. `check()` returns
None for a valid response and a short reason otherwise. Invalid responses are
retried by the client and never cached.
"""

SUM_TOLERANCE = 0.03     # Jev rounds to 0.01, so its distributions can sum to 0.99 or 1.01


def _is_prob(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and -1e-9 <= x <= 1 + 1e-9


def _check_distribution(probs, allowed):
    if not isinstance(probs, dict) or not probs:
        return "probabilities missing"
    unknown = set(probs) - set(allowed)
    if unknown:
        return f"unknown options {sorted(unknown)[:3]}"
    if not all(_is_prob(v) for v in probs.values()):
        return "probability outside [0, 1]"
    total = sum(probs.values())
    if abs(total - 1) > SUM_TOLERANCE:
        return f"probabilities sum to {total:.3f}"
    return None


def check(questions, response):
    answers = response.get("answers") if isinstance(response, dict) else None
    if not isinstance(answers, dict):
        return "no answers in response"
    for qid, q in questions.items():
        a = answers.get(qid)
        if not isinstance(a, dict):
            return f"{qid}: not answered"
        qtype = q.get("type")
        if a.get("type") != qtype:
            return f"{qid}: type {a.get('type')!r} instead of {qtype!r}"
        if qtype == "noul":
            if not _is_prob(a.get("noul")):
                return f"{qid}: noul value invalid"
        elif qtype == "choice":
            options = list(q.get("criteria") or {})
            problem = _check_distribution(a.get("probabilities"), options)
            if problem:
                return f"{qid}: {problem}"
            if a.get("choice") not in options:
                return f"{qid}: choice {a.get('choice')!r} not among options"
        elif qtype == "score":
            levels = [str(i) for i in range(len(q.get("criteria") or []))]
            problem = _check_distribution(a.get("probabilities"), levels)
            if problem:
                return f"{qid}: {problem}"
            s = a.get("score")
            if not isinstance(s, (int, float)) or not (-1e-6 <= s <= len(levels) - 1 + 1e-6):
                return f"{qid}: score {s!r} outside the scale"
    return None
