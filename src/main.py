"""
Command line runner for the Music Recommender Simulation.

Top-level menu lets you:
  - Get recommendations (Naive LLM or RAG mode)
  - Add a new song to the catalog (persisted to data/songs.csv)
  - Create a new user profile (persisted to data/user_profiles.json)

Custom songs and profiles persist across runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "songs.csv"
DEFAULT_PROFILES_PATH = PROJECT_ROOT / "data" / "user_profiles.json"

try:
    from src.recommender import (
        load_songs,
        append_song,
        load_custom_profiles,
        save_custom_profile,
        recommend_naive_llm,
        recommend_rag,
    )
    from src.llm_client import LLMError
except ImportError:
    from recommender import (
        load_songs,
        append_song,
        load_custom_profiles,
        save_custom_profile,
        recommend_naive_llm,
        recommend_rag,
    )
    from llm_client import LLMError


BUILTIN_PROFILES: Dict[str, Dict] = {
    "High-Energy Pop": {
        "genre": "pop", "mood": "happy", "energy": 0.9, "likes_acoustic": False,
    },
    "Chill Lofi": {
        "genre": "lofi", "mood": "chill", "energy": 0.3, "likes_acoustic": True,
    },
    "Deep Intense Rock": {
        "genre": "rock", "mood": "intense", "energy": 0.85, "likes_acoustic": False,
    },
    "Nostalgic Jazz": {
        "genre": "jazz", "mood": "nostalgic", "energy": 0.4, "likes_acoustic": False,
    },
    # --- Adversarial / Edge Case Profiles ---
    "Sad but Hyper (Contradictory)": {
        "genre": "pop", "mood": "melancholic", "energy": 0.95, "likes_acoustic": False,
    },
    "Ghost Genre (Nonexistent)": {
        "genre": "k-pop", "mood": "happy", "energy": 0.7, "likes_acoustic": False,
    },
    "Zero Energy Rocker (Conflicting)": {
        "genre": "rock", "mood": "intense", "energy": 0.05, "likes_acoustic": True,
    },
    "Wants Everything (Greedy)": {
        "genre": "folk", "mood": "happy", "energy": 0.95, "likes_acoustic": True,
    },
}

MODES = {
    "1": ("Naive LLM", "naive"),
    "2": ("RAG (retrieval-augmented)", "rag"),
}


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def _prompt_choice(label: str, choices: Dict[str, str]) -> str:
    print(f"\n{label}")
    for key, value in choices.items():
        print(f"  [{key}] {value}")
    while True:
        choice = input("> ").strip()
        if choice in choices:
            return choice
        print(f"  Invalid choice. Pick one of: {', '.join(choices.keys())}")


def _prompt_nonempty(label: str) -> str:
    while True:
        value = input(f"{label}: ").strip()
        if value:
            return value
        print("  Required. Please enter a value.")


def _prompt_float(label: str, lo: float, hi: float) -> float:
    while True:
        raw = input(f"{label} ({lo}-{hi}): ").strip()
        try:
            value = float(raw)
        except ValueError:
            print(f"  Not a number. Try again.")
            continue
        if not (lo <= value <= hi):
            print(f"  Out of range. Must be between {lo} and {hi}.")
            continue
        return value


def _prompt_yesno(label: str) -> bool:
    while True:
        raw = input(f"{label} (y/n): ").strip().lower()
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  Please answer y or n.")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Flows
# ---------------------------------------------------------------------------

def _recommend_flow(songs: List[Dict], profiles: Dict[str, Dict]) -> None:
    profile_choices = {str(i): name for i, name in enumerate(profiles.keys(), 1)}
    mode_choices = {key: label for key, (label, _) in MODES.items()}

    profile_key = _prompt_choice("Pick a user profile:", profile_choices)
    profile_name = profile_choices[profile_key]
    user_prefs = profiles[profile_name]

    mode_key = _prompt_choice("Pick a mode:", mode_choices)
    mode_label, mode = MODES[mode_key]

    try:
        results = _run_mode(mode, user_prefs, songs)
    except LLMError as e:
        print(f"\n[error] {e}\n")
        return

    _print_results(profile_name, user_prefs, mode_label, results)


def _add_song_flow(data_path: Path, songs: List[Dict]) -> None:
    print("\n--- Add a new song to the catalog ---")
    title = _prompt_nonempty("Title")
    artist = _prompt_nonempty("Artist")
    genre = _prompt_nonempty("Genre (e.g. pop, lofi, rock)")
    mood = _prompt_nonempty("Mood (e.g. happy, chill, intense)")
    energy = _prompt_float("Energy", 0.0, 1.0)
    tempo = _prompt_float("Tempo (BPM)", 30.0, 300.0)
    valence = _prompt_float("Valence", 0.0, 1.0)
    danceability = _prompt_float("Danceability", 0.0, 1.0)
    acousticness = _prompt_float("Acousticness", 0.0, 1.0)

    new_song = {
        "title": title,
        "artist": artist,
        "genre": genre,
        "mood": mood,
        "energy": energy,
        "tempo_bpm": tempo,
        "valence": valence,
        "danceability": danceability,
        "acousticness": acousticness,
    }
    record = append_song(str(data_path), new_song)
    songs.append(record)  # keep in-memory catalog in sync
    print(f"\nAdded: {record['title']} by {record['artist']} (id={record['id']})")
    print(f"Catalog now has {len(songs)} songs.\n")


def _create_profile_flow(profiles_path: Path, profiles: Dict[str, Dict]) -> None:
    print("\n--- Create a new user profile ---")
    while True:
        name = _prompt_nonempty("Profile name")
        if name in profiles:
            overwrite = _prompt_yesno(f"'{name}' already exists. Overwrite?")
            if overwrite:
                break
            continue
        break

    genre = _prompt_nonempty("Favorite genre")
    mood = _prompt_nonempty("Favorite mood")
    energy = _prompt_float("Target energy", 0.0, 1.0)
    likes_acoustic = _prompt_yesno("Like acoustic music?")

    prefs = {
        "genre": genre,
        "mood": mood,
        "energy": energy,
        "likes_acoustic": likes_acoustic,
    }
    save_custom_profile(str(profiles_path), name, prefs)
    profiles[name] = prefs
    print(f"\nSaved profile '{name}'. It will appear in the picker on next run too.\n")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

TOP_MENU = {
    "1": "Get recommendations",
    "2": "Add a song to the catalog",
    "3": "Create a new user profile",
    "q": "Quit",
}


def _interactive_loop(data_path: Path, profiles_path: Path) -> None:
    songs = load_songs(str(data_path))
    profiles: Dict[str, Dict] = {**BUILTIN_PROFILES, **load_custom_profiles(str(profiles_path))}

    while True:
        choice = _prompt_choice("What would you like to do?", TOP_MENU)
        if choice == "q":
            break
        if choice == "1":
            _recommend_flow(songs, profiles)
        elif choice == "2":
            _add_song_flow(data_path, songs)
        elif choice == "3":
            _create_profile_flow(profiles_path, profiles)


def _run_all(songs: List[Dict]) -> None:
    """Sweep every built-in profile through every mode. Useful for the model card writeup."""
    for profile_name, user_prefs in BUILTIN_PROFILES.items():
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
        help="Run every built-in profile through every mode (no menu). For comparison writeups.",
    )
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA_PATH),
        help=f"Path to songs CSV (default: {DEFAULT_DATA_PATH}).",
    )
    parser.add_argument(
        "--profiles",
        default=str(DEFAULT_PROFILES_PATH),
        help=f"Path to custom profiles JSON (default: {DEFAULT_PROFILES_PATH}).",
    )
    args = parser.parse_args()

    if args.all:
        songs = load_songs(args.data)
        _run_all(songs)
    else:
        _interactive_loop(Path(args.data), Path(args.profiles))


if __name__ == "__main__":
    main()
