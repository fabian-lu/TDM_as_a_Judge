"""CALM's published numbers for its six LLM judges (Ye et al., ICLR 2025), for comparison.

Copied from the ICLR 2025 version of the paper: Table 4 (one number per bias) and Table 8
(per variant). Judges: ChatGPT (gpt-3.5-turbo-0125), GPT-4-Turbo (0409), GPT-4o (0513),
GLM-4 (0520), Claude-3.5 (Sonnet 0620), Qwen2-72B-Instruct, all at temperature 0.7.

Notes on reading them:
  * All values are robustness rates (RR), except fallacy oversight, which we read as the share
    of changed verdicts, i.e. correct reactions (CALM counts a changed verdict as correct).
  * CR_fact and CR_alignment are CALM's consistency rates on the two datasets.
  * Table 8 lists distraction as "h.c" and "h.r"; we read these as the distraction on the
    better (chosen) and on the worse (rejected) response.
"""

JUDGES = ["ChatGPT", "GPT-4-Turbo", "GPT-4o", "GLM-4", "Claude-3.5", "Qwen2"]

TABLE4 = {   # bias: values for JUDGES in order
    "verbosity":   [0.900, 0.915, 0.977, 0.887, 0.952, 0.884],
    "fallacy":     [0.917, 0.969, 0.984, 0.979, 0.985, 0.935],
    "sentiment":   [0.804, 0.653, 0.699, 0.679, 0.660, 0.651],
    "CR_fact":     [0.998, 0.990, 0.998, 0.970, 0.999, 0.994],
    "position":    [0.566, 0.818, 0.776, 0.781, 0.832, 0.760],
    "compassion":  [0.862, 0.858, 0.868, 0.835, 0.875, 0.877],
    "bandwagon":   [0.688, 0.638, 0.791, 0.690, 0.610, 0.710],
    "authority":   [0.662, 0.846, 0.787, 0.796, 0.865, 0.779],
    "distraction": [0.713, 0.729, 0.790, 0.814, 0.878, 0.785],
    "diversity":   [0.679, 0.855, 0.814, 0.788, 0.914, 0.826],
    "CR_alignment": [0.906, 0.856, 0.925, 0.884, 0.915, 0.904],
}

TABLE8 = {   # (bias, CALM's variant name, our variant name): values for JUDGES in order
    ("bandwagon", "60%", "60"): [0.680, 0.635, 0.773, 0.703, 0.563, 0.711],
    ("bandwagon", "70%", "70"): [0.667, 0.630, 0.787, 0.676, 0.613, 0.711],
    ("bandwagon", "80%", "80"): [0.707, 0.662, 0.805, 0.664, 0.638, 0.698],
    ("bandwagon", "90%", "90"): [0.699, 0.623, 0.800, 0.716, 0.627, 0.718],
    ("distraction", "h.c", "about-R1"): [0.716, 0.718, 0.749, 0.806, 0.904, 0.749],
    ("distraction", "h.r", "about-R2"): [0.710, 0.740, 0.830, 0.822, 0.851, 0.821],
    ("authority", "Book", "book"): [0.628, 0.841, 0.800, 0.765, 0.856, 0.785],
    ("authority", "Quote", "quote"): [0.660, 0.841, 0.747, 0.758, 0.856, 0.745],
    ("authority", "URL", "url"): [0.700, 0.855, 0.813, 0.866, 0.884, 0.805],
    ("sentiment", "Che.(bet.)", "cheerful-on-R1"): [0.803, 0.682, 0.727, 0.770, 0.609, 0.726],
    ("sentiment", "Che.(wor.)", "cheerful-on-R2"): [0.910, 0.888, 0.970, 0.905, 0.976, 0.871],
    ("sentiment", "Sad(bet.)", "sad-on-R1"): [0.659, 0.271, 0.343, 0.306, 0.259, 0.307],
    ("sentiment", "Sad(wor.)", "sad-on-R2"): [0.916, 0.920, 0.983, 0.907, 0.970, 0.929],
    ("sentiment", "Ang.(bet.)", "angry-on-R1"): [0.639, 0.366, 0.243, 0.380, 0.256, 0.283],
    ("sentiment", "Ang.(wor.)", "angry-on-R2"): [0.946, 0.921, 0.987, 0.950, 0.973, 0.926],
    ("sentiment", "Fea.(bet.)", "fear-on-R1"): [0.639, 0.254, 0.355, 0.271, 0.260, 0.238],
    ("sentiment", "Fea.(wor.)", "fear-on-R2"): [0.923, 0.921, 0.987, 0.943, 0.973, 0.926],
    ("diversity", "Homosexual", "Homosexual"): [0.697, 0.830, 0.819, 0.779, 0.945, 0.839],
    ("diversity", "Black", "Black"): [0.660, 0.843, 0.820, 0.784, 0.897, 0.819],
    ("diversity", "Female", "Female"): [0.646, 0.825, 0.826, 0.765, 0.924, 0.805],
    ("diversity", "HIV Pos.", "HIV Positive"): [0.692, 0.856, 0.820, 0.832, 0.942, 0.826],
    ("diversity", "Refugees", "Refugees"): [0.667, 0.896, 0.799, 0.785, 0.862, 0.826],
    ("diversity", "Muslim", "Muslim believers"): [0.710, 0.881, 0.800, 0.785, 0.913, 0.845],
}

FACT_BIASES = {"verbosity", "fallacy", "sentiment"}      # the others use the alignment set


def table4_long():
    """CALM's Table 4 as rows: judge, bias, value, and the judge's consistency rate on that dataset."""
    rows = []
    for bias, values in TABLE4.items():
        if bias.startswith("CR_"):
            continue
        cr = TABLE4["CR_fact" if bias in FACT_BIASES else "CR_alignment"]
        for judge, v, c in zip(JUDGES, values, cr):
            rows.append({"judge": judge, "bias": bias, "value": v, "CR": c})
    return rows
