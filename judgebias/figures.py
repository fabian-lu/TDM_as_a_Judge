"""Shared style, colours and small data helpers for the paper's figures and tables (notebook 07_figures).

    from judgebias import figures as F
    F.style()                                  # matplotlib defaults for the paper
    fig, ax = plt.subplots(figsize=(F.COLUMN, 2.4))
    F.save(fig, "shifts")                      # paper/figures/shifts.pdf (+ a PNG preview next to it)

Colours come from a palette validated for colour-vision deficiency (dataviz skill, light mode):
models use the categorical slots in a fixed order, typed decision models vs LLM judges use the
first two slots, and the direction of a shift uses the diverging pair (red rewards, blue punishes).
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent.parent
TABLES = CODE / "tables"
PAPER = CODE / "paper"            # figures and LaTeX tables are written to paper/ in the repository
FIGURES = PAPER / "figures"
TEX = PAPER / "tables"

COLUMN, TEXTWIDTH = 3.03, 6.3          # inches, ACL two-column layout

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
TYPED, LLM = "#2a78d6", "#eb6834"      # categorical slots 1 and 2
REWARD, PUNISH, NEUTRAL = "#e34948", "#2a78d6", "#f0efec"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

MODELS = ["jev-1.13", "kev-4b", "tev1-4b", "solar-decide", "d1", "pplx-decider-v1.1-27b", "clef", "clef-flash"]
MODEL_NAMES = {"jev-1.13": "Jev 1.13", "kev-4b": "Kev 4B", "tev1-4b": "Tev1 4B", "solar-decide": "Solar Decide",
               "d1": "D1", "pplx-decider-v1.1-27b": "Perplexity Decider", "clef": "Clef", "clef-flash": "Clef Flash"}
MODEL_COLOURS = dict(zip(MODELS, CATEGORICAL))

# LLM judges on the same prompts (notebooks 08 and 09): CALM's two 2024 judges that are still served, then five 2026 LLMs
LLMS = ["ChatGPT", "GPT-4o", "GPT-6.1 Sol", "Claude Sonnet 5.5", "Gemini 3.8 Flash", "Qwen3.8-27B", "Claude Haiku 5.5"]
LLMS_2024, LLMS_2026 = LLMS[:2], LLMS[2:]
LLM_SHORT = {"ChatGPT": "ChatGPT", "GPT-4o": "GPT-4o", "GPT-6.1 Sol": "GPT-6.1 Sol", "Claude Sonnet 5.5": "Sonnet 5.5",
             "Gemini 3.8 Flash": "Gemini 3.8 Fl.", "Qwen3.8-27B": "Qwen3.8-27B", "Claude Haiku 5.5": "Haiku 5.5"}
# requests in flight in the main pass of each LLM judge (data/llm_judges/run_log.jsonl)
LLM_PARALLEL = {"ChatGPT": "32", "GPT-4o": "16", "GPT-6.1 Sol": "32", "Claude Sonnet 5.5": "32",
                "Gemini 3.8 Flash": "32, then 16", "Qwen3.8-27B": "32", "Claude Haiku 5.5": "16"}
LLM_OLD = "#1baf7a"                    # categorical slot 3: the 2024 LLM judges (LLM, slot 2, for the 2026 ones)

TEST_NAMES = {"position": "Position", "verbosity": "Verbosity", "compassion": "Compassion fade",
              "bandwagon": "Bandwagon", "authority": "Authority", "authority, better response": "Authority (on better)",
              "distraction": "Distraction", "diversity": "Diversity", "diversity, neutral note": "Neutral note",
              "sentiment": "Sentiment", "fallacy": "Fallacy oversight", "verbosity, better response": "Verbosity (on better)"}
DELTA = 0.05                           # smallest excess we call a bias (Section 4.1)


def style():
    """Matplotlib defaults: a Times-like serif to match the paper, small sizes, recessive axes."""
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 0, "axes.grid": False, "grid.color": GRID,
        "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300, "pdf.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "lines.solid_capstyle": "round"})


def save(fig, name, preview_dir=None):
    """Save as vector PDF for the paper and as PNG for a quick look."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig((Path(preview_dir) if preview_dir else FIGURES) / f"{name}.png", dpi=200)
    return FIGURES / f"{name}.pdf"


def write_tex(name, text):
    """Write a LaTeX table snippet to paper/tables/<name>.tex (input by the paper). The judge that CALM calls ChatGPT
    keeps that label in the data (llm_judges.MODELS, the CSV files) but is shown as GPT-3.5 Turbo in the paper."""
    text = text.replace("ChatGPT", "GPT-3.5 Turbo")
    TEX.mkdir(parents=True, exist_ok=True)
    (TEX / f"{name}.tex").write_text(text)
    return TEX / f"{name}.tex"


def classify(excess_lo, excess_hi, delta=DELTA):
    """'bias' if the whole 95% interval of the excess lies above delta, 'none' if below, else 'unclear'."""
    return "bias" if excess_lo > delta else "none" if excess_hi < delta else "unclear"


def calibration(model, fmt="calm", dataset="alignment", bins=10):
    """Calibration of P(better response) on the original prompts: Brier score, ECE (top label, equal-width bins
    on [0.5, 1]), log loss, accuracy, mean confidence, and the reliability curve (bin confidence vs accuracy).

    A 50/50 answer enters every measure with its probability 0.5. It has no verdict, so it is left out of the
    accuracy (as everywhere in the paper), and in the ECE and the reliability curve it counts as half right, the
    accuracy of a choice between two options at random."""
    from . import authority, design, metrics, run
    rows = design.build_design(citations=authority.load_citations())
    orig = {r["id"]: r for r in rows if r["bias"] == "original" and r["dataset"] == dataset}
    recs = [x for x in run.load_records(model, fmt) if x["repeat"] == 0 and x["id"] in orig and x["ok"]]
    p1 = np.array([x["p_A"] if orig[x["id"]]["r1_label"] == "A" else 1 - x["p_A"] for x in recs])
    p1 = np.clip(p1, 1e-6, 1 - 1e-6)
    tie = np.abs(p1 - 0.5) < metrics.TIE
    conf, correct = np.maximum(p1, 1 - p1), np.where(tie, 0.5, (p1 > 0.5).astype(float))
    b = np.clip(np.digitize(conf, np.linspace(0.5, 1, bins + 1)) - 1, 0, bins - 1)
    curve = pd.DataFrame({"bin": b, "conf": conf, "correct": correct}).groupby("bin").agg(
        conf=("conf", "mean"), accuracy=("correct", "mean"), n=("correct", "size")).reset_index()
    ece = float(sum(abs(r.accuracy - r.conf) * r.n for r in curve.itertuples()) / len(conf))
    return {"brier": float(np.mean((1 - p1) ** 2)), "ece": ece, "log_loss": float(-np.mean(np.log(p1))),
            "accuracy": float(correct[~tie].mean()), "confidence": float(conf.mean()), "n": len(conf),
            "ties": int(tie.sum()), "curve": curve}
