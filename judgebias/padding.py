"""Padded better responses for the verbosity test on the better response (our third addition, review round 2).

CALM pads only the worse response (`answer2_longer` in its verbosity file). A judge that dislikes padding then keeps
its verdicts and looks robust, so we also pad the better response of the same 500 items, the way CALM padded the
worse one:

  * model        GPT-4-Turbo (`openai/gpt-4-turbo` on OpenRouter), as for our regenerated citations
  * temperature  0.7, as CALM used for all its generating models
  * prompt       CALM's `longSentence` prompt from promptTemplate.py ("Prompt to make worse answers longer"), with the
                 better response as the answer
  * padded text  the reply, as in CALM's file

    gpt = ChatClient(padding.MODEL, temperature=padding.TEMPERATURE, accept=padding.check_reply)
    padding.save(padding.generate(gpt, padding.jobs()))
    padded = padding.load_padded()              # {pair id: padded better response}, used by design.build_design
"""
import json
from pathlib import Path

from . import files
from .calm import T, load

MODEL = "openai/gpt-4-turbo"
TEMPERATURE = 0.7
DATA = Path(__file__).resolve().parent.parent / "data"
PADDED_FILE = DATA / "padded_better.jsonl"
MIN_GROWTH = 1.2          # a reply counts as padded when it is at least 20% longer than the response


def _pair_id(item):
    from .design import pair_id
    return pair_id(item)


def jobs():
    """One job per item of CALM's verbosity file: the better response (answer1) to be padded."""
    return [{"item": _pair_id(it), "index": i, "answer": it["answer1"], "prompt": T.longSentence(it["answer1"])}
            for i, it in enumerate(load("verbosity_bias_standardized"))]


def padded_text(reply):
    """The padded response in a reply: the reply without surrounding whitespace or a leading label."""
    text = (reply or "").strip()
    for label in ("Expanded Answer:", "Expanded answer:", "Answer:"):
        if text.startswith(label):
            text = text[len(label):].strip()
    return text


def check_reply(reply, answer=None):
    """For ChatClient(accept=...): None if the reply is a usable padded response, else the reason."""
    text = padded_text(reply)
    if not text:
        return "empty reply"
    if text.startswith("Prompt:") or "Original Answer:" in text:
        return "the reply echoes the prompt"
    return None


def generate(client, todo):
    """Run the jobs with a ChatClient and return one record per job (reply, padded response, reported model)."""
    results = client.ask_many([{"messages": [{"role": "user", "content": j["prompt"]}]} for j in todo],
                              desc="padding")
    records = []
    for j, r in zip(todo, results):
        text = padded_text(r["text"]) if r["ok"] else None
        grew = bool(text) and len(text) >= MIN_GROWTH * len(j["answer"])
        records.append({**j, "model": MODEL, "temperature": TEMPERATURE, "ok": r["ok"] and grew,
                        "model_reported": r["snapshot"], "reply": r["text"], "padded": text if grew else None,
                        "growth": len(text) / len(j["answer"]) if text else None, "error": r["error"]})
    return records


def save(records, path=PADDED_FILE):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_padded(path=PADDED_FILE):
    """{pair id: padded better response} from the saved generations, or {} if they do not exist yet."""
    if files.find(path) is None:
        return {}
    return {r["item"]: r["padded"] for r in map(json.loads, files.read_lines(path)) if r["padded"]}
