"""Fake citations for the authority test, generated anew with CALM's procedure.

In CALM's released authority file most citations are attached to the wrong items
(paper, Appendix C). We generate them again the way CALM describes:

  * model        GPT-4-Turbo (`openai/gpt-4-turbo` on OpenRouter, the April 2024 version CALM lists)
  * temperature  0.7, as CALM used for all its generating models
  * prompts      CALM's three generation prompts (book, URL, quote) from promptTemplate.py,
                 with the response as the "sentence"
  * citation     the text between [[ and ]] in the reply
  * insertion    CALM's own authority template appends it to the response: answer + "(citation)"

Our only change: the reply format of the quote prompt gets a slot for the quoted words
(see `quote_prompt`). Every generation is kept, so the citations can be released.
"""
import json
import re
from pathlib import Path

from . import files, paths
from .calm import T, load

MODEL = "openai/gpt-4-turbo"
TEMPERATURE = 0.7
KINDS = ("book", "url", "quote")
CITATIONS_FILE = paths.CITATIONS

PUBLISHED_QUOTE_LINE = ("[[Quote: ——[Random Politician or Scientist Name], [Random Year between 1900-2022], "
                        "in an interview on the topic of [keywords]]]")
OUR_QUOTE_LINE = ('[[Quote: "[quote]" ——[Random Politician or Scientist Name], [Random Year between 1900-2022], '
                  'in an interview on the topic of [keywords]]]')


def quote_prompt(answer):
    """CALM's quote prompt with one change: the reply format asks for the quoted words too."""
    prompt = T.generate_quote_prompt(answer)
    assert prompt.count(PUBLISHED_QUOTE_LINE) == 1, "CALM's quote prompt is not as expected"
    return prompt.replace(PUBLISHED_QUOTE_LINE, OUR_QUOTE_LINE)


def generation_prompt(kind, answer):
    """The prompt that asks GPT-4-Turbo for one citation of the given kind."""
    if kind == "book":
        return T.generate_book_prompt(answer)
    if kind == "url":
        return T.generate_url_prompt(answer)
    if kind == "quote":
        return quote_prompt(answer)
    raise ValueError(kind)


BRACKETS = re.compile(r"\[\[(.+?)\]\]", re.S)


def parse_citation(reply):
    """The citation in a reply: the last [[...]] block, or None if there is none or it is still a template."""
    found = BRACKETS.findall(reply or "")
    if not found:
        return None
    citation = found[-1].strip()
    if "[Random" in citation or "[keywords" in citation or "[topic" in citation:
        return None                                   # the model echoed the template
    return citation


def check_reply(reply):
    """For ChatClient(accept=...): None if the reply contains a usable citation, else the reason."""
    return None if parse_citation(reply) else "no usable [[...]] citation in the reply"


def jobs():
    """One generation job per item of CALM's authority file, response (1 or 2) and kind: 150 x 2 x 3 = 900."""
    items = load("authority_bias_standardized")
    out = []
    for i, item in enumerate(items):
        for side in (1, 2):
            for kind in KINDS:
                out.append({"item": i, "side": side, "kind": kind,
                            "prompt": generation_prompt(kind, item[f"answer{side}"])})
    return out


def generate(client, todo):
    """Run the jobs with a ChatClient and return one record per job (reply, citation, reported model)."""
    results = client.ask_many([{"messages": [{"role": "user", "content": j["prompt"]}]} for j in todo],
                              desc="citations")
    records = []
    for j, r in zip(todo, results):
        records.append({**j, "model": MODEL, "temperature": TEMPERATURE, "ok": r["ok"],
                        "model_reported": r["snapshot"], "reply": r["text"],
                        "citation": parse_citation(r["text"]) if r["ok"] else None, "error": r["error"]})
    return records


def fill_missing(client, records):
    """Generate the citations that are still missing and return all records, in the original order."""
    missing = [r for r in records if r["citation"] is None]
    if not missing:
        return records
    redone = generate(client, [{k: r[k] for k in ("item", "side", "kind", "prompt")} for r in missing])
    by_job = {(r["item"], r["side"], r["kind"]): r for r in redone}
    return [by_job.get((r["item"], r["side"], r["kind"]), r) for r in records]


def load_records(path=CITATIONS_FILE):
    """All saved generation records (one per job)."""
    return [json.loads(line) for line in files.read_lines(path)]


def save(records, path=CITATIONS_FILE):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_citations(path=CITATIONS_FILE):
    """{(item, side, kind): citation} from the saved generations, or None if they do not exist yet."""
    if files.find(path) is None:
        return None
    recs = [json.loads(line) for line in files.read_lines(path)]
    return {(r["item"], r["side"], r["kind"]): r["citation"] for r in recs if r["citation"]}


def with_citation(answer, citation):
    """The response with the citation added, exactly as CALM's authority template does it."""
    return answer + f"({citation})"


def regenerated_file(citations):
    """CALM's authority file with our citations, in CALM's own format (a drop-in replacement)."""
    out = []
    for i, item in enumerate(load("authority_bias_standardized")):
        row = {k: item[k] for k in ("question", "answer1", "answer2", "data_resource")}
        for side in (1, 2):
            for kind in KINDS:
                c = citations.get((i, side, kind))
                if c:
                    row[f"answer{side}_with_reference_{kind}"] = with_citation(item[f"answer{side}"], c)
        out.append(row)
    return out
