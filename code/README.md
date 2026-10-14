# Code

The code reproduces every number, table and figure of the paper from the answers in `data/`, and sends the prompts
to other judges.

## Layout

| Folder | What it holds |
|---|---|
| `judgebias/` | The Python package: CALM's prompts in three request formats (`calm.py`), the prompts of all tests (`design.py`), sending them to TDMs (`run.py`) and LLM judges (`llm_judges.py`), the metrics and statistics (`metrics.py`), and helpers for the figures (`figures.py`). All folder paths are set in `paths.py`. |
| `tdm/` | A small client for the models on OpenRouter, with retries, a budget guard and a cache of every call. |
| `notebooks/` | The analysis, step by step, with the outputs of our run. |
| `R/` | The mixed models (`lme4`), run from `notebooks/05_exploratory`. |
| `outputs/` | The results as CSV files (`tables/`) and the long table with the mixed models (`exploratory/`). `notebooks/07_figures` adds the paper's figures (`figures/`) and tables (`latex/`). |
| `external/` | Where CALM's repository is cloned to (see `external/README.md`). |

## Setup

From the `code/` folder, with Python 3.13:

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
git clone https://github.com/Y0oMu/LLM-Judge-Bias-Dataset external/calm
git -C external/calm checkout f6248e9f9a93c20c2059bbb46ce3096ca414ad19
```

The mixed models need R 4.5 with the packages `data.table` and `lme4`.

## Reproducing the paper

All answers of the judges are stored in `data/`, so the analysis sends no requests and needs no API key. Run the
notebooks `04`, `05`, `09` and then `07`. The others document how the data was made and send requests.

| Notebook | What it does |
|---|---|
| `01_formats` | CALM's prompt and the exact request in each of the three formats, for every test. |
| `02_citations` | Generates the 900 fake citations of the authority tests anew, as CALM describes. |
| `03_runs` | Sends every prompt to every TDM in every format. |
| `04_analysis` | The metrics for every TDM, format and test, and the comparison with CALM. |
| `05_exploratory` | Margins, items, versions, the mixed models (about 16 minutes) and asking in both orders. |
| `06_llm_cost` | Time and cost of three of CALM's LLM judges on 100 prompts. |
| `07_figures` | The paper's figures and results tables, written to `outputs/`. |
| `08_llm_judges` | Sends every prompt to the seven LLM judges. |
| `09_llm_analysis` | The metrics for the LLM judges and the comparison with the TDMs. |

## Testing other models

Sending the released prompts to another model needs an API key but not CALM's repository. Copy `.env.example` to
`.env` and fill in an OpenRouter key. An Anthropic key is only needed for Claude models called directly on
Anthropic's API, as Claude Sonnet 5.5 was in the paper. Every answer is stored in a cache (`cache/`), so a run that
stops continues where it ended, and sending the same request again costs nothing.

**A TDM.** Any decision model on OpenRouter can be tested in one of the three formats. A full test cost US$0.26
to 3.11 per model and format in October 2026.

```python
from judgebias import design, run

rows = design.load_prompts()                                    # the 14,223 prompts of data/prompts.jsonl.gz
records, client = run.run_model("typesafe/jev-1.13", "calm", rows, budget_usd=5)
```

The answers are saved in `data/answers/tdms/`. To include the model in the analysis, add its OpenRouter id to
`MODELS` in `judgebias/run.py` and run `04_analysis`, which then reports it in every table in `outputs/tables/`.

**An LLM judge.** Add a line to `MODELS` in `judgebias/llm_judges.py`, with a name, its OpenRouter id, a generation
(`"2024"` or `"2026"` in the paper) and whether it reasons (`OFF`, or `LOW` where reasoning cannot be switched off):

```python
    "My judge": ("provider/model", "2026", OFF),
```

Then send the prompts:

```python
from judgebias import design, llm_judges

records = llm_judges.run_full("My judge", design.load_prompts(), budget_usd=50)
```

The judge receives CALM's prompt unchanged, and its verdict is read from the `[[A]]` or `[[B]]` that CALM asks for.
The answers are saved in `data/answers/llm_judges/`, and `09_llm_analysis` includes every judge in `MODELS`. A full
run of the paper's judges cost between US$4.73 (Claude Haiku 5.5) and US$111 (GPT-4o).

The figures of `07_figures` are laid out for the paper's eight TDMs and seven LLM judges.
