"""One function per provider: complete(instructions, text) -> answer text.

Keys come from the single .env file in the repo root (see .env.example).
"""

import os

import requests
from dotenv import load_dotenv
from openai import OpenAI

from . import config as cfg
from .prompts import GEMINI_INPUT_TEMPLATE

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"


def _env(name, default=None):
    value = os.getenv(name, default)
    if not value:
        raise RuntimeError(f"{name} is missing in {cfg.ENV_FILE}")
    return value


def _openai_style(client, model):
    """Azure OpenAI and DeepSeek both use the OpenAI Responses API."""

    def complete(instructions, text):
        response = client.responses.create(model=model, instructions=instructions, input=text)
        return response.output_text.strip()

    return complete


def _gemini(model, api_key):
    """Gemini Interactions API. Instructions and input go in as one text."""

    def complete(instructions, text):
        response = requests.post(
            GEMINI_URL,
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            json={
                "model": model,
                "input": GEMINI_INPUT_TEMPLATE.format(instructions=instructions, text=text),
                "store": False,
            },
            timeout=180,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Gemini HTTP {response.status_code}: {response.text[:1000]}")

        for step in response.json().get("steps", []):
            if step.get("type") != "model_output":
                continue
            for item in step.get("content", []):
                if item.get("type") == "text" and item.get("text", "").strip():
                    return item["text"].strip()
        return ""

    return complete


def get_client(model_name):
    """Return (complete_function, model_id) for an entry in cfg.MODELS."""
    load_dotenv(cfg.ENV_FILE, override=True)
    spec = cfg.MODELS[model_name]
    provider = spec["provider"]

    if provider == "azure":
        model = _env(spec["model_env"])
        client = OpenAI(api_key=_env("AZURE_API_KEY"), base_url=_env("AZURE_BASE_URL"))
        return _openai_style(client, model), model

    if provider == "deepseek":
        model = spec["model"]
        client = OpenAI(
            api_key=_env("DEEPSEEK_API_KEY"),
            base_url=_env("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
        return _openai_style(client, model), model

    if provider == "gemini":
        model = spec["model"]
        return _gemini(model, _env("GEMINI_API_KEY")), model

    raise ValueError(f"Unknown provider: {provider}")
