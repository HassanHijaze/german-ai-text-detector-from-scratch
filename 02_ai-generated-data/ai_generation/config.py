"""Settings for generating the AI texts.

Override from the notebook, e.g.:
    from ai_generation import config as cfg
    cfg.MAX_RETRIES = 3
"""

from pathlib import Path

# Repo root (this file is in <repo>/02_ai_generation/ai_generation/)
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = REPO_ROOT / ".env"

# Input: the human data from step 01
HUMAN_META = REPO_ROOT / "data" / "raw" / "human" / "meta.json"
HUMAN_DIR = REPO_ROOT / "data" / "raw" / "human" / "human"

# Output: one text file per human sample, named <seed_sample_id>.txt
ASSISTED_DIR = REPO_ROOT / "data" / "raw" / "ai_assisted"
GENERATED_DIR = REPO_ROOT / "data" / "raw" / "ai_generated"
LOG_FILE = REPO_ROOT / "data" / "raw" / "generation_log.jsonl"

MAX_RETRIES = 5

# Reject outputs shorter than this share of the human text and retry
# (catches cut-off or empty answers).
MIN_LENGTH_RATIO = 0.5

# Each model handled one range of human sample IDs.
#   provider:  "azure", "deepseek" or "gemini"
#   model:     model name (or the .env variable holding the Azure deployment name)
#   strategy:  prompt strategy for the AI-generated class (see prompts.py)
MODELS = {
    "gpt-5.6-luna": {
        "provider": "azure",
        "company": "OpenAI",
        "model_env": "AZURE_LUNA56_DEPLOYMENT",
        "seed_ids": (1, 2499),
        "strategy": "source_based",
        "workers": 50,
    },
    "gemini-3.8-flash": {
        "provider": "gemini",
        "company": "Google",
        "model": "gemini-3.8-flash",
        "seed_ids": (2500, 4999),
        "strategy": "source_based",
        "workers": 5,
    },
    "deepseek-flash": {
        "provider": "deepseek",
        "company": "DeepSeek",
        "model": "deepseek-flash",
        "seed_ids": (5000, 7499),
        "strategy": "topics",
        "workers": 20,
    },
    "gpt-6-luna": {
        "provider": "azure",
        "company": "OpenAI",
        "model_env": "AZURE_LUNA6_DEPLOYMENT",
        "seed_ids": (7500, 10227),
        "strategy": "source_based",
        "workers": 50,
    },
}
