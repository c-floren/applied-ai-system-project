"""
Reliability tests for the AI modes.

These hit the live Gemini API (or the disk cache if a previous run populated
it). Skipped automatically if GEMINI_API_KEY is not set, so contributors
without a key can still run `pytest` cleanly.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "songs.csv"

from src.recommender import load_songs, recommend_rag
from src.llm_client import LLMError
from src.main import BUILTIN_PROFILES


@pytest.fixture(scope="module", autouse=True)
def _require_api_key():
    load_dotenv(PROJECT_ROOT / ".env")
    if not os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY") == "your_key_here":
        pytest.skip("GEMINI_API_KEY not set; skipping live-LLM grounding tests")


@pytest.fixture(scope="module")
def songs():
    return load_songs(str(DATA_PATH))


@pytest.fixture(scope="module")
def catalog_titles(songs):
    return {s["title"] for s in songs}


@pytest.mark.parametrize("profile_name,prefs", list(BUILTIN_PROFILES.items()))
def test_rag_only_returns_catalog_songs(profile_name, prefs, songs, catalog_titles):
    """
    The RAG system prompt forbids inventing titles. This locks that contract:
    every returned title must exist in the catalog. If this fails, the LLM
    violated the constraint and the recommender's [INVALID: ...] marker
    should have caught it — the test reports both.

    Skipped (not failed) on transient API errors / rate limits, since the
    contract being tested is the LLM's adherence to the system prompt — not
    Gemini's free-tier uptime.
    """
    try:
        results = recommend_rag(prefs, songs, k=5, candidates=10)
    except LLMError as e:
        pytest.skip(f"LLM unavailable for {profile_name}: {str(e)[:80]}")

    assert len(results) > 0, f"RAG returned no items for {profile_name}"

    out_of_catalog = [
        (song["title"], song.get("artist", ""))
        for song, _, _ in results
        if song.get("title", "").startswith("[INVALID: ") or song.get("title") not in catalog_titles
    ]
    assert not out_of_catalog, (
        f"RAG returned out-of-catalog titles for profile '{profile_name}': "
        f"{out_of_catalog}"
    )


def test_rag_returns_requested_count(songs):
    """RAG should return exactly k=5 items for a typical profile."""
    prefs = BUILTIN_PROFILES["High-Energy Pop"]
    try:
        results = recommend_rag(prefs, songs, k=5, candidates=10)
    except LLMError as e:
        pytest.skip(f"LLM unavailable: {str(e)[:80]}")
    assert len(results) == 5, f"Expected 5 items, got {len(results)}"
