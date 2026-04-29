"""
Command line runner for the Music Recommender Simulation.

Pick a user profile and a recommendation mode:
  1) Naive LLM  - Gemini guesses songs from its training data, no catalog
  2) RAG        - rule-based retriever pulls catalog candidates, Gemini re-ranks

The legacy rule-based scorer still exists in recommender.py — it now powers
the RAG retriever. The user-facing answer comes from Gemini in both modes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "songs.csv"

try:
    from src.recommender import (
        load_songs,
        recommend_naive_llm,
        recommend_rag,
    )
    from src.llm_client import LLMError
except ImportError:
    from recommender import (
        load_songs,
        recommend_naive_llm,
        recommend_rag,
    )
    from llm_client import LLMError


USER_PROFILES: Dict[str, Dict] = {
    "High-Energy Pop": {
        "genre": "pop",
        "mood": "happy",
        "energy": 0.9,
        "likes_acoustic": False,
    },
    "Chill Lofi": {
        "genre": "lofi",
        "mood": "chill",
        "energy": 0.3,
        "likes_acoustic": True,
    },
    "Deep Intense Rock": {
        "genre": "rock",
        "mood": "intense",
        "energy": 0.85,
        "likes_acoustic": False,
    },
    "Nostalgic Jazz": {
        "genre": "jazz",
        "mood": "nostalgic",
        "energy": 0.4,
        "likes_acoustic": False,
    },
    # --- Adversarial / Edge Case Profiles ---
    "Sad but Hyper (Contradictory)": {
        "genre": "pop",
        "mood": "melancholic",
        "energy": 0.95,
        "likes_acoustic": False,
    },
    "Ghost Genre (Nonexistent)": {
        "genre": "k-pop",
        "mood": "happy",
        "energy": 0.7,
        "likes_acoustic": False,
    },
    "Zero Energy Rocker (Conflicting)": {
        "genre": "rock",
        "mood": "intense",
        "energy": 0.05,
        "likes_acoustic": True,
    },
    "Wants Everything (Greedy)": {
        "genre": "folk",
        "mood": "happy",
        "energy": 0.95,
        "likes_acoustic": True,
    },
}

MODES = {
    "1": ("Naive LLM", "naive"),
    "2": ("RAG (retrieval-augmented)", "rag"),
}


def _print_results(
    profile_name: str,
    user_prefs: Dict,
    mode_label: str,
    results: List[Tuple[Dict, float, str]],
) -> None:
    print("\n" + "=" * 60)
    print(f"  MODE: {mode_label.upper()}")
    print(f"  PROFILE: {profile_name}")
    print(
        f"  Genre: {user_prefs['genre']} | Mood: {user_prefs['mood']} | "
        f"Energy: {user_prefs['energy']}"
    )
    print("=" * 60 + "\n")

    if not results:
        print("  (no recommendations returned)\n")
        return

    for i, (song, score, reasons) in enumerate(results, 1):
        print(f"  #{i}")
        print(f"  Title:   {song.get('title', 'Unknown')}")
        print(f"  Artist:  {song.get('artist', 'Unknown')}")
        print(f"  Score:   {score:.2f}")
        print(f"  Reasons: {reasons}")
        print()


def _run_mode(mode: str, user_prefs: Dict, songs: List[Dict]) -> List[Tuple[Dict, float, str]]:
    if mode == "naive":
        return recommend_naive_llm(user_prefs, k=5)
    if mode == "rag":
        return recommend_rag(user_prefs, songs, k=5, candidates=10)
    raise ValueError(f"Unknown mode: {mode}")


def _prompt_choice(label: str, choices: Dict[str, str]) -> str:
    print(f"\n{label}")
    for key, value in choices.items():
        print(f"  [{key}] {value}")
    while True:
        choice = input("> ").strip()
        if choice in choices:
            return choice
        print(f"  Invalid choice. Pick one of: {', '.join(choices.keys())}")


def _interactive_loop(songs: List[Dict]) -> None:
    profile_choices = {str(i): name for i, name in enumerate(USER_PROFILES.keys(), 1)}
    mode_choices = {key: label for key, (label, _) in MODES.items()}

    while True:
        profile_key = _prompt_choice("Pick a user profile:", profile_choices)
        profile_name = profile_choices[profile_key]
        user_prefs = USER_PROFILES[profile_name]

        mode_key = _prompt_choice("Pick a mode:", mode_choices)
        mode_label, mode = MODES[mode_key]

        try:
            results = _run_mode(mode, user_prefs, songs)
        except LLMError as e:
            print(f"\n[error] {e}\n")
            continue

        _print_results(profile_name, user_prefs, mode_label, results)

        again = input("Run another? (y/n) > ").strip().lower()
        if again != "y":
            break


def _run_all(songs: List[Dict]) -> None:
    """Sweep every profile through every mode. Useful for the model card writeup."""
    for profile_name, user_prefs in USER_PROFILES.items():
        for _, (mode_label, mode) in MODES.items():
            try:
                results = _run_mode(mode, user_prefs, songs)
            except LLMError as e:
                print(f"\n[error] {profile_name} / {mode_label}: {e}\n")
                continue
            _print_results(profile_name, user_prefs, mode_label, results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Music recommender (AI-powered).")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run every profile through every mode (no menu). For comparison writeups.",
    )
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA_PATH),
        help=f"Path to songs CSV (default: {DEFAULT_DATA_PATH}).",
    )
    args = parser.parse_args()

    songs = load_songs(args.data)

    if args.all:
        _run_all(songs)
    else:
        _interactive_loop(songs)


if __name__ == "__main__":
    main()
