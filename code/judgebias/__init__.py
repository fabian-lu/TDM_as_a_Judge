"""CALM's judge-bias tests on typed decision models (TDMs) and LLM judges.

    paths.py       where the data (data/) and the analysis results (code/outputs/) live
    files.py       reading the data files, compressed or not
    calm.py        CALM's data and prompt templates, and the three request formats for TDMs
    authority.py   the fake citations of the authority test, generated anew with CALM's procedure
    padding.py     the padded better responses of verbosity (on better)
    design.py      every prompt we send (one row per prompt), or the released prompts
    run.py         sending the prompts to a TDM and storing the answers
    llm_judges.py  sending the prompts to an LLM judge and reading its verdict
    llm_cost.py    time and cost of three of CALM's LLM judges
    metrics.py     the metrics and statistics of the paper
    explore.py     one long table of all TDM answers, for the exploratory analysis and the mixed models
    ensembles.py   ensembles of TDMs, from the stored answers
    reference.py   CALM's published numbers, for comparison
    figures.py     style, names and helpers for the paper's figures and tables
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]          # code/, home of the `tdm` package
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
