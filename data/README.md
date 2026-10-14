# The benchmark

The prompts of our bias tests and the answers of all fifteen judges of the paper. The prompts extend the CALM
benchmark ([Ye et al., 2025](https://llm-judge-bias.github.io/)) and can be sent to any judge that compares two
responses. No code from this repository is needed to use them.

## Quick start

```python
import pandas as pd

prompts = pd.read_json("data/prompts.jsonl.gz", lines=True)                              # 14,223 prompts
jev = pd.read_json("data/answers/tdms/typesafe_jev-1.13__calm.jsonl.gz", lines=True)    # one judge's answers
d = jev.query("repeat == 0").merge(prompts, on="id")
```

## How the tests work

Each test changes one thing that should not matter, such as the order of the two responses or a claimed majority
for one of them. Every perturbed prompt points to its unperturbed version (`original`), and a fair judge gives both
the same verdict. The better response is called R1 and the worse R2, and `r1_label` says whether R1 is shown as
answer A or B. The paper measures how much more often a judge changes its verdict under a perturbation than when the
unperturbed prompt is simply sent again (the excess). For this, each original prompt was sent twice (`repeat` 0 and 1).

| Test | `bias` | `variant` | Prompts |
|---|---|---|---|
| Position | `position` | `swapped` | 439 |
| Verbosity | `verbosity` | `R2-padded` | 500 |
| Verbosity (on better)\* | `verbosity` | `R1-padded` | 500 |
| Fallacy oversight | `fallacy` | `R1-broken` | 500 |
| Sentiment | `sentiment` | a tone (cheerful, sad, angry, fear) on R1 or R2 | 1,193 |
| Authority | `authority` | a citation (book, URL, quote) on R2 | 450 |
| Authority (on better)\* | `authority` | a citation on R1 | 450 |
| Bandwagon | `bandwagon` | a claimed majority for R2 (60 to 90%) | 1,756 |
| Distraction | `distraction` | an irrelevant note about R1 or R2 | 878 |
| Diversity | `diversity` | an identity note (six groups) on R1 or R2 | 5,268 |
| Neutral note\* | `diversity` | the note "Software engineer" on R1 or R2 | 878 |
| Compassion fade | `compassion` | model names for the two responses | 439 |
| Original prompts | `original` | `-` | 972 |

\*Our additions to CALM. In fallacy oversight the perturbation breaks the reasoning of R1, so there the verdict
should change.

## Files

| File | Content |
|---|---|
| `prompts.jsonl.gz` | The 14,223 prompts. |
| `answers/tdms/<model>__<format>.jsonl.gz` | The answers of the eight typed decision models (TDMs) in three request formats (`calm`, `native`, `decomposed`, see the paper). |
| `answers/llm_judges/<model>.jsonl.gz` | The answers of the seven LLM judges, with their full replies. |
| `answers/cost_sample/<model>.jsonl` | Time and cost of GPT-3.5 Turbo, GPT-4 Turbo and GPT-4o on 100 prompts. |
| `answers/*/run_log.jsonl` | One line per run: start, duration, requests, concurrency and cost. |
| `generated/authority_citations.jsonl.gz` | The 900 fake citations of the authority tests, with the exact prompt and reply of each generation. |
| `generated/authority_bias_regenerated.json` | CALM's authority file with these citations, in CALM's format. |
| `generated/padded_better.jsonl.gz` | The 500 padded better responses of verbosity (on better). |

### Fields

**Prompts.** `id`, `item` (the question and its two responses, shared by all tests on it), `dataset` (`alignment` or
`fact`), `source` (the dataset the item comes from), `bias`, `variant`, `original` (the id of the unperturbed prompt,
empty for originals), `r1_label` (where R1 is shown, `A` or `B`), `target` (the response the perturbation is about,
`R1` or `R2`) and `prompt` (CALM's judge prompt, as CALM's template returns it).

**TDM answers.** `id`, `repeat`, `choice` (`A` or `B`), `p_A` (the probability of A), `p_A_questions` (the decomposed
format only: P(A) for each of the six criteria), `snapshot` (the model version the API reported), `ok`, `status`,
`error`, `latency_s`, `cost` (US$), `input_tokens`, `output_tokens`.

**LLM judge answers.** `id`, `repeat`, `verdict` (`A`, `B` or empty if the reply has none), `reply` (the full reply),
`snapshot`, `ok`, `error`, `latency_s`, `cost`, `input_tokens`, `output_tokens`, `reasoning_tokens`.

A verdict is correct if it equals `r1_label`. It flips if it differs from the verdict on the original prompt.

## Warnings

All citations in the authority tests are fabricated on purpose, since these tests measure whether a judge is swayed
by fake references. The books and authors are invented, the URLs were never checked, and most quotes attribute
invented words to real public figures. Do not use them as sources or quote them outside the benchmark. Every
record in `generated/authority_citations.jsonl.gz` carries `"fabricated": true`. CALM's alignment set, and therefore
our prompts, contains 44 items from a role-play dataset with sexual content.

## Licence

CC BY-NC-SA 4.0 (`LICENSE`), as required for material adapted from ScienceQA. The prompts contain material from
CALM and the datasets it builds on, which keep their own terms (listed in `LICENSE`).
