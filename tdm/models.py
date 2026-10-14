"""Which typed decision models exist and how we reach them (checked 2026-10-01, updated 2026-10-04).

`OPENROUTER` lists the entries of OpenRouter's "decisions" output modality.
`OLLAMA` lists decision models that Ollama >= 0.35 can run locally.
`list_openrouter()` fetches the current list, since new models appear weekly.
"""
import json
import urllib.request

from .config import OPENROUTER_MODELS_URL, openrouter_key

OPENROUTER = {
    "typesafe/jev-1.13":                     {"maker": "TypeSafe AI", "types": "noul, choice, score", "context": 32000,
                                              "note": "closed; the reference model"},
    "jaredpalmer/kev-4b":                    {"maker": "Jared Palmer", "types": "noul, choice, score", "context": 8192,
                                              "note": "open weights, Qwen3.5-4B + LoRA + pointer head"},
    "togethercomputer/tev1-4b-experimental": {"maker": "Together AI", "types": "noul, choice, score", "context": 32768,
                                              "note": "open weights, fine-tuned Qwen3.5-4B"},
    "inception/mercury-decide:free":         {"maker": "Inception", "types": "noul, choice, score", "context": 32768,
                                              "note": "free tier"},
    "upstage/solar-decide":                  {"maker": "Upstage", "types": "noul, choice, score", "context": 524288,
                                              "note": "built on Solar Mini 4"},
    "liquid/d1":                             {"maker": "Liquid AI", "types": "noul, choice, score", "context": 65536,
                                              "note": "listed 2026-10-01"},
    "perplexity/pplx-decider-v1-27b":        {"maker": "Perplexity", "types": "noul, choice, score", "context": 262144,
                                              "note": "listed 2026-10-01, 27B"},
    "cloudflare/clef":                       {"maker": "Cloudflare", "types": "noul, choice, score", "context": 65536,
                                              "note": "listed 2026-10-01, open weights, fine-tuned Qwen3.8-27B, multimodal"},
    "cloudflare/clef-flash":                 {"maker": "Cloudflare", "types": "noul, choice, score", "context": 65536,
                                              "note": "listed 2026-10-01, open weights, fine-tuned Qwen3.5-9B"},
    "respan/span-01":                        {"maker": "Respan", "types": "noul only", "context": None,
                                              "note": "scores behaviours in conversations; state = text or {input, output}"},
    "respan/span-01-lite":                   {"maker": "Respan", "types": "noul only", "context": None,
                                              "note": "lighter Span-01"},
    "respan/span-01-lite:free":              {"maker": "Respan", "types": "noul only", "context": None,
                                              "note": "free tier of Span-01 Lite"},
}

OLLAMA = {   # local, CPU only on this laptop (no GPU); Ollama 0.35.0 installed 2026-10-01
    "tev1:0.8b": {"maker": "Together AI", "size": "0.8B", "note": "811 MB download, ~0.75 s per call on CPU"},
    "tev1":      {"maker": "Together AI", "size": "4B"},
    "nimble":    {"maker": "Bespoke Labs", "size": "9B"},
}


def list_openrouter():
    """Current OpenRouter decision models as a list of dicts (id, name, context_length, pricing, ...)."""
    req = urllib.request.Request(OPENROUTER_MODELS_URL, headers={"Authorization": "Bearer " + openrouter_key()})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())["data"]
