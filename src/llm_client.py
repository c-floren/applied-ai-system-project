"""
Gemini API wrapper with disk-based response caching.

Used by the naive LLM and RAG recommendation modes in recommender.py.
Caches keep demos and screenshots reproducible without burning quota.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / ".cache"
DEFAULT_MODEL = "gemini-2.5-flash"


class LLMError(RuntimeError):
    """Raised when the Gemini call fails or the response cannot be parsed."""


def _cache_key(system: str, prompt: str, model: str) -> str:
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(system.encode("utf-8"))
    h.update(b"\x00")
    h.update(prompt.encode("utf-8"))
    return h.hexdigest()


def _load_api_key() -> str:
    load_dotenv(PROJECT_ROOT / ".env")
    key = os.getenv("GEMINI_API_KEY")
    if not key or key == "your_key_here":
        raise LLMError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key "
            "from https://aistudio.google.com/app/apikey"
        )
    return key


def generate_json(
    prompt: str,
    system: str,
    model: str = DEFAULT_MODEL,
    use_cache: bool = True,
) -> dict:
    """
    Send `prompt` to Gemini and parse the JSON response.

    Cached on disk by sha256(model + system + prompt) so re-runs are free.
    Raises LLMError on missing key, API failure, or unparseable output.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    key_path = CACHE_DIR / f"{_cache_key(system, prompt, model)}.json"

    if use_cache and key_path.exists():
        try:
            return json.loads(key_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            key_path.unlink(missing_ok=True)

    api_key = _load_api_key()

    try:
        from google import genai
        from google.genai import types
    except ImportError as e:
        raise LLMError(
            "google-genai is not installed. Run: pip install -r requirements.txt"
        ) from e

    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.3,
                response_mime_type="application/json",
                max_output_tokens=4096,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
    except Exception as e:
        raise LLMError(f"Gemini API call failed: {e}") from e

    text: Optional[str] = getattr(response, "text", None)
    if not text:
        raise LLMError("Gemini returned an empty response.")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise LLMError(f"Gemini response was not valid JSON: {text[:200]}") from e

    if use_cache:
        key_path.write_text(json.dumps(parsed, indent=2), encoding="utf-8")

    return parsed
