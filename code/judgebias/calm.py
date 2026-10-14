"""CALM (Ye et al., ICLR 2025) data and prompts, converted to typed decision requests.

Principle: we never rewrite CALM's prompts. Every request starts from the string
returned by one of CALM's own functions in `external/calm/promptTemplate.py`, called
with CALM's own data. That string is then split mechanically:

  * the `[System]` block becomes the typed question's `instructions`, minus the two
    sentences that only make sense for a model that writes text ("... provide a short
    explanation." and "After providing your explanation, output your final verdict by
    strictly following this format: ..."),
  * everything from `[User Question]` on (question, answer blocks with their headers,
    and any appended bias sentence) becomes the `state`, as CALM's exact text,
  * the verdict options are read from CALM's own format sentence ("[[A]]" if X is
    better, ...), so the option descriptions use CALM's wording, including the model
    names of the compassion-fade template.

The only deviation from CALM is therefore the missing explanation step, which typed
decision models cannot perform. LLM judges can be run with CALM's original string.
"""
import json
import re
import sys
from pathlib import Path

from .paths import CALM_DIR

sys.path.insert(0, str(CALM_DIR))
try:
    import promptTemplate as T  # noqa: E402  (CALM's code, unchanged)
except ImportError:             # CALM not cloned: sending the released prompts (data/prompts.jsonl.gz) still works
    T = None

DROP = [  # sentences that ask for written output; removed for typed decision models
    re.compile(r"Begin your evaluation by comparing the two\s+responses and provide a short explanation\.\s*"),
    re.compile(r"After providing your explanation, output your\s+final verdict by strictly following this format:.*?(?=\[User Question\])", re.S),
]
VERDICT = re.compile(r'"\[\[([ABC])\]\]"\s*(?:if|for)\s*(.+?)(?=,\s*"\[\[|,\s*and\s*"\[\[|\.\s*$|$)', re.S)


def load(name):
    """Load a CALM file by stem, e.g. 'alignment_dataset' or 'verbosity_bias_standardized'."""
    for sub in ("base_datasets", "bias_datasets"):
        p = CALM_DIR / sub / f"{name}.json"
        if p.exists():
            data = json.loads(p.read_text())
            return data if isinstance(data, list) else next(iter(data.values()))
    raise FileNotFoundError(f"{name}: clone CALM into {CALM_DIR} first (see code/external/README.md)")


def to_typed(prompt):
    """Split one CALM pairwise prompt string into (state, questions) for a typed decision model."""
    system, rest = prompt.split("[User Question]", 1)
    fmt = re.search(r"output your\s+final verdict by strictly following this format:(.*?)\[User Question\]",
                    prompt, re.S)
    options = {}
    for label, cond in VERDICT.findall(fmt.group(1)):
        cond = " ".join(cond.split()).rstrip(".")
        options[label] = ("It is a tie." if label == "C" else f"{cond[0].upper()}{cond[1:]}.")
    instructions = system.replace("[System]", "")
    for pat in DROP:
        instructions = pat.sub("", instructions + "[User Question]").replace("[User Question]", "")
    instructions = " ".join(instructions.split())
    state = ("[User Question]" + rest).strip()
    question = {"type": "choice", "instructions": instructions, "criteria": options}
    return state, {"verdict": question}


def verdict(result):
    """The chosen option ('A', 'B' or 'C') and the probability of each option, from a tdm result."""
    if not result["ok"]:
        return None, None
    a = result["answers"]["verdict"]
    return a["choice"], a["probabilities"]


# -- native and decomposed formats ----------------------------------------------------
# TypeSafe's docs recommend structured state and structured instructions. Both formats use
# exactly the instruction sentences of the CALM format (`to_typed`), split into named fields:
#   task      CALM's two opening sentences (role and what to choose)
#   consider  CALM's criteria sentence
#   guidance  CALM's debiasing sentences
# The native format differs from the CALM format only in layout. The decomposed format asks
# one question per criterion; its only change is that `consider` names that one criterion.
BLOCK = re.compile(r"\[The Start of (?P<name>.+?)[’']s Answer\]\n(?P<text>.*?)\n\[The End of (?P=name)[’']s Answer\]", re.S)
CRITERIA = ["helpfulness", "relevance", "accuracy", "depth", "creativity", "level of detail"]
FACTOR_IDS = ["helpfulness", "relevance", "accuracy", "depth", "creativity", "detail"]   # question ids, not sent to the model


def parse(prompt):
    """CALM prompt string -> dict(system, question, answers=[(label, text), ...], note)."""
    system, rest = prompt.split("[User Question]", 1)
    first = BLOCK.search(rest)
    question = rest[:first.start()].strip()
    answers = [(m["name"], m["text"]) for m in BLOCK.finditer(rest)]
    note = rest[list(BLOCK.finditer(rest))[-1].end():].strip()
    return {"system": system.replace("[System]", "").strip(), "question": question, "answers": answers, "note": note}


def split_instruction(text):
    """CALM-format instruction -> {task, consider, guidance}, the same sentences in the same order."""
    i = text.index("Your evaluation should consider")
    j = text.index("Avoid any position biases")
    parts = {"task": text[:i].strip(), "consider": text[i:j].strip(), "guidance": text[j:].strip()}
    assert " ".join(parts.values()) == text, "instruction split changed the text"
    return parts


def _native_state(prompt):
    p = parse(prompt)
    state = {"user_question": p["question"]}
    for label, text in p["answers"]:
        state[f"{label}'s answer"] = text
    if p["note"]:
        state["note"] = p["note"]
    return state


def to_native(prompt):
    """Same text as `to_typed`, as a structured state and a structured instruction."""
    _, typed_q = to_typed(prompt)
    q = typed_q["verdict"]
    return _native_state(prompt), {"verdict": {"type": "choice", "instructions": split_instruction(q["instructions"]),
                                               "criteria": q["criteria"]}}


def to_decomposed(prompt):
    """Native state, one question per CALM criterion. Only the `consider` sentence differs."""
    _, typed_q = to_typed(prompt)
    q = typed_q["verdict"]
    base = split_instruction(q["instructions"])
    assert base["consider"].startswith("Your evaluation should consider factors such as the helpfulness")
    questions = {}
    for qid, crit in zip(FACTOR_IDS, CRITERIA):
        instr = dict(base, consider=f"Your evaluation should consider the {crit} of their responses.")
        questions[qid] = {"type": "choice", "instructions": instr, "criteria": q["criteria"]}
    return _native_state(prompt), questions


def combine_decomposed(answers):
    """Mean probability of option A over the six questions, and the verdict (None on an exact tie)."""
    p_a = sum(answers[k]["probabilities"]["A"] for k in FACTOR_IDS) / len(FACTOR_IDS)
    return p_a, ("A" if p_a > 0.5 else "B" if p_a < 0.5 else None)
