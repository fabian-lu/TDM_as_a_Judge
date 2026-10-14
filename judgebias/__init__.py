"""CALM's judge-bias tests on typed decision models (TDMs) and LLM judges.

    calm.py       CALM's data and prompt templates, and the three request formats
    authority.py  the fake citations of the authority test, generated anew with CALM's procedure
    design.py     every prompt we send (one row per prompt)
    run.py        sending the design to the models and storing the answers
    metrics.py    the metrics and statistics of Section 3
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]          # the repository root, home of the `tdm` package
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
