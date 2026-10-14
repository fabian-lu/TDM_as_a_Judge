# Cheap, Consistent, Calibrated, but Biased? Typed Decision Models as Judges

Fabian Lukassen<sup>1</sup>, Christoph Weisser<sup>2</sup>, Alexander Silbersdorff<sup>1</sup>  
<sup>1</sup>University of Göttingen, <sup>2</sup>Bielefeld University of Applied Sciences and Arts (HSBI)

[Project page](https://fabian-lu.github.io/TDM_as_a_Judge/) | [Paper (PDF)](https://fabian-lu.github.io/TDM_as_a_Judge/paper.pdf) | arXiv (soon)

Typed decision models (TDMs) return a probability for each allowed answer instead of generating text, which makes
them candidates for cheaply replacing LLM judges that compare two answers. We run the bias tests of the CALM
benchmark ([Ye et al., 2025](https://llm-judge-bias.github.io/)) on eight TDMs and seven LLM judges of 2024 and 2026.
This repository contains our code, the prompts we sent and every answer of every judge, so that all numbers, tables
and figures of the paper can be rebuilt without a single API call, and a new TDM can be tested for a few dollars.

## Contents

| Folder | What it holds |
|---|---|
| `judgebias/` | The code: CALM's prompts in three request formats, the test design, sending requests, the metrics and statistics, figures. |
| `tdm/` | A small client for the TDMs (and chat models) on OpenRouter, with retries, a budget guard and a cache of every call. |
| `notebooks/` | The analysis, step by step, with the outputs of our run. |
| `R/` | The mixed models (`lme4`). |
| `data/` | The prompts, the generated citations and padded responses, and the answers of all judges (gzip-compressed). |
| `tables/` | The results as CSV files, from which the paper's tables and figures are made. |
| `external/` | Where CALM's repository is cloned to (not included, see below). |

## Setup

Python 3.13:

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

CALM's data and prompt templates are read from a clone of its repository at the commit we used:

```bash
git clone https://github.com/Y0oMu/LLM-Judge-Bias-Dataset external/calm
git -C external/calm checkout f6248e9f9a93c20c2059bbb46ce3096ca414ad19
```

API keys are needed only for the notebooks that send requests. Copy `.env.example` to `.env` and fill in an
OpenRouter key (TDMs and most LLM judges) and, for Claude Sonnet 5.5, an Anthropic key. Every answer is stored in a
local cache (`cache/`), so running a notebook again costs nothing.

The mixed models need R 4.5 with the packages `data.table` and `lme4`.

## Reproducing the paper

The notebooks are numbered in the order in which we ran them. Those marked *API* send requests, the others work
from the released data.

| Notebook | What it does |
|---|---|
| `01_formats` | Shows CALM's prompt and the exact request in each of the three formats for every test (nothing is sent, but the client needs an OpenRouter key). |
| `02_citations` | *API.* Generates the 900 fake citations of the authority test anew, as CALM describes (about US$5). |
| `03_runs` | *API.* Sends every prompt to every TDM in every format (US$38 for all eight TDMs in all three formats). |
| `04_analysis` | The metrics of the paper for every TDM, format and test, and the comparison with CALM. |
| `05_exploratory` | Margins, items, versions, the mixed models (runs the R scripts, about 16 minutes) and the effect of asking in both orders. |
| `06_llm_cost` | *API.* Time and cost of three of CALM's LLM judges on a sample of 100 prompts. |
| `07_figures` | Every figure and results table of the paper, written to `paper/`. |
| `08_llm_judges` | *API.* Sends every prompt to the seven LLM judges (US$321 for all seven). |
| `09_llm_analysis` | The metrics for the LLM judges and the comparison with the TDMs. |

To rebuild the paper's numbers from the released answers, run `04`, `05`, `09` and then `07`.

## Testing a new TDM

Any decision model on OpenRouter can be tested on the full design. A full test in the CALM or the native format
cost US$0.26 to 3.11 per model in October 2026.

```python
from judgebias import authority, design, run

rows = design.build_design(citations=authority.load_citations())        # the 14,223 prompts
records, client = run.run_model("typesafe/jev-1.13", "calm", rows, budget_usd=5)
```

The answers are written to `data/results/`. To include the model in the analysis, add it to `run.MODELS` and run
`04_analysis`.

## Data

All files with one JSON object per line are gzip-compressed (`.jsonl.gz`). The code reads them directly.

* `design.jsonl.gz`: the 14,223 prompts (13,251 perturbed and 972 original ones), with the test (`bias`), the
  version (`variant`), the item and its source dataset, where the better response is shown (`r1_label`), the response
  whose probability we follow (`target`) and CALM's prompt string. Each original prompt is sent twice, which gives
  15,195 requests per judge and format.
* `results/<model>__<format>.jsonl.gz`: the answers of the eight TDMs in the three formats, one line per request: the
  prompt id, `repeat` (1 for the second sending of an original prompt), the choice, P(A), for the decomposed format
  P(A) of each of its six questions, the model version the API reported, latency, cost and tokens.
  `results/run_log.jsonl` has one line per run.
* `llm_judges/<model>.jsonl.gz`: the answers of the seven LLM judges, with the full reply, the verdict read from it,
  tokens, cost and latency. `llm_judges/run_log.jsonl` has one line per run.
* `llm_cost/<model>.jsonl`: the time and cost sample of GPT-3.5 Turbo, GPT-4 Turbo and GPT-4o (100 prompts each).
* `authority_citations.jsonl.gz`: every generation of a fake citation (item, response, kind, the exact prompt, the
  full reply, the citation and the model the API reported). `authority_bias_regenerated.json` is CALM's authority
  file with these citations, in CALM's format.
* `padded_better.jsonl.gz`: the 500 padded better responses of our test verbosity (on better).
* `exploratory/`: one long table of all TDM answers with their features, and the results of the mixed models.

**Warnings.** All citations in the authority data are fabricated on purpose, since the test measures whether a judge
is swayed by fake references. The books and authors are invented, the URLs were never checked, and most quotes
attribute invented words to real public figures. Do not use them as sources or quote them outside the benchmark.
Every record in `authority_citations.jsonl.gz` carries `"fabricated": true`. CALM's alignment set, and therefore our
prompts, contains 44 items from a role-play dataset with sexual content.

## Licences

Our code is released under the MIT licence (`LICENSE`). Our data, that is the prompts, the answers of the judges and
our analysis in `data/` and `tables/`, are released under CC BY-NC-SA 4.0 (`LICENSE-DATA`), as required for material
adapted from ScienceQA. The prompts contain material from CALM and the datasets it builds on, which keep their own
terms. CALM itself is not included.

## Citation

```bibtex
@misc{lukassen2026tdmjudge,
  title  = {Cheap, Consistent, Calibrated, but Biased? Typed Decision Models as Judges},
  author = {Lukassen, Fabian and Weisser, Christoph and Silbersdorff, Alexander},
  year   = {2026},
  note   = {Preprint}
}
```

Please also cite CALM:

```bibtex
@inproceedings{ye2025justice,
  title     = {Justice or Prejudice? Quantifying Biases in {LLM}-as-a-Judge},
  author    = {Ye, Jiayi and Wang, Yanbo and Huang, Yue and Chen, Dongping and Zhang, Qihui and Moniz, Nuno and Gao, Tian and Geyer, Werner and Huang, Chao and Chen, Pin-Yu and Chawla, Nitesh V. and Zhang, Xiangliang},
  booktitle = {The Thirteenth International Conference on Learning Representations},
  year      = {2025}
}
```
