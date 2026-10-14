# CALM (Ye et al., ICLR 2025)

The code reads CALM's data and prompt templates from `external/calm/`, a clone of
https://github.com/Y0oMu/LLM-Judge-Bias-Dataset at commit f6248e9f9a93c20c2059bbb46ce3096ca414ad19, the version we
used (taken on 2026-10-02). Create it from the repository root with

    git clone https://github.com/Y0oMu/LLM-Judge-Bias-Dataset external/calm
    git -C external/calm checkout f6248e9f9a93c20c2059bbb46ce3096ca414ad19

Our code imports `calm/promptTemplate.py` and reads CALM's JSON files without changes. CALM's repository has no
licence file, so we do not copy it here.
