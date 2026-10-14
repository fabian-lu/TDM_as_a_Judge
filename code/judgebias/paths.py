"""Where everything lives: the benchmark in data/ at the repository root, the analysis results in code/outputs/."""
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]           # code/
ROOT = CODE.parent                                   # the repository root
DATA = ROOT / "data"

PROMPTS = DATA / "prompts.jsonl"                     # stored as prompts.jsonl.gz, read by files.read_lines
TDM_ANSWERS = DATA / "answers" / "tdms"              # <model>__<format>.jsonl(.gz)
LLM_ANSWERS = DATA / "answers" / "llm_judges"        # <model>.jsonl(.gz)
COST_SAMPLE = DATA / "answers" / "cost_sample"       # time and cost of three of CALM's LLM judges
CITATIONS = DATA / "generated" / "authority_citations.jsonl"
PADDED = DATA / "generated" / "padded_better.jsonl"

OUTPUTS = CODE / "outputs"
TABLES = OUTPUTS / "tables"                          # the results as CSV files
EXPLORATORY = OUTPUTS / "exploratory"                # the long table and the mixed models
FIGURES = OUTPUTS / "figures"                        # the paper's figures (PDF, with a PNG preview)
LATEX = OUTPUTS / "latex"                            # the paper's results tables (LaTeX)

CALM_DIR = CODE / "external" / "calm"                # CALM's repository (see external/README.md)
