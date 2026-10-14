"""Paths and secrets: the cache folder and the API keys in `.env`, both in the repository root."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent      # the repository root
CACHE_DIR = ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def _load_env(path):
    """Minimal .env reader (KEY=VALUE lines), so tdm needs no extra package."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env(ROOT / ".env")

OPENROUTER_URL = "https://openrouter.ai/api/alpha/decisions"
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models?output_modalities=decisions"
OLLAMA_URL = "http://localhost:11434/v1/systemone"


def openrouter_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(f"OPENROUTER_API_KEY not set; expected it in {ROOT / '.env'}")
    return key
