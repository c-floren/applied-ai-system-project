from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import csv
import json

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

def _profile_to_prefs(user: UserProfile) -> Dict:
    return {
        "genre": user.favorite_genre,
        "mood": user.favorite_mood,
        "energy": user.target_energy,
        "likes_acoustic": user.likes_acoustic,
    }


class Recommender:
    """
    OOP implementation of the rule-based recommender.
    Required by tests/test_recommender.py.
    Wraps the functional `score_song` / `recommend_songs` so the same scoring
    logic powers both the legacy interface and the RAG retriever.
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        prefs = _profile_to_prefs(user)
        scored = [(song, score_song(prefs, asdict(song))[0]) for song in self.songs]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [song for song, _ in scored[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        prefs = _profile_to_prefs(user)
        _, reasons = score_song(prefs, asdict(song))
        return ", ".join(reasons)

def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
    songs = []
    with open(csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Convert numeric fields to appropriate types
            song = {
                'id': int(row['id']),
                'title': row['title'],
                'artist': row['artist'],
                'genre': row['genre'],
                'mood': row['mood'],
                'energy': float(row['energy']),
                'tempo_bpm': float(row['tempo_bpm']),
                'valence': float(row['valence']),
                'danceability': float(row['danceability']),
                'acousticness': float(row['acousticness'])
            }
            songs.append(song)
    return songs

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """
    Scores a single song against user preferences.
    Required by recommend_songs() and src/main.py
    """
        
    # Feature weights (percentage values sum to 100)
    weights = {
        "genre":          0.40,
        "mood":           0.30,
        "energy":         0.15,
        "danceability":   0.10,
        "acousticness":   0.05,
    }
    
    total_score = 0.0
    reasons = []
    
    # CATEGORICAL SCORES (0 or 1)
    genre_match = 1.0 if song["genre"] == user_prefs["genre"] else 0.0
    mood_match = 1.0 if song["mood"] == user_prefs["mood"] else 0.0
    
    if genre_match:
        reasons.append(f"Matches favorite genre: {song['genre']}")
    
    if mood_match:
        reasons.append(f"Matches favorite mood: {song['mood']}")
    
    # NUMERICAL SCORES (distance-based, 0.0-1.0)
    energy_score = 1.0 - abs(song["energy"] - user_prefs["energy"]) / 1.0
    energy_diff = abs(song["energy"] - user_prefs["energy"])
    if energy_diff < 0.2:
        reasons.append(f"Energy level close to preference ({song['energy']:.2f})")
    
    danceability_score = 1.0 - abs(song["danceability"] - 0.7) / 1.0  # Assume neutral target = 0.7
    if song["danceability"] > 0.75:
        reasons.append(f"High danceability ({song['danceability']:.2f})")
    
    acousticness_score = 1.0 if user_prefs.get("likes_acoustic", False) else 1.0 - song["acousticness"]
    if user_prefs.get("likes_acoustic", False) and song["acousticness"] > 0.6:
        reasons.append(f"Good acousticness match ({song['acousticness']:.2f})")

    # WEIGHTED TOTAL
    total_score = (
        genre_match * weights["genre"] * 100 +
        mood_match * weights["mood"] * 100 +
        energy_score * weights["energy"] * 100 +
        danceability_score * weights["danceability"] * 100 +
        acousticness_score * weights["acousticness"] * 100
    )
    
    # Add fallback reason if no specific matches
    if not reasons:
        reasons.append("General match based on musical attributes")
    
    # Expected return format: (score, reasons)
    return (total_score, reasons)

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Return the top k songs ranked by score.
    Required by src/main.py
    """
    # Calculate scores for all songs using list comprehension
    scored_songs = [
        (song, score, ", ".join(reasons))
        for song in songs
        for score, reasons in [score_song(user_prefs, song)]
    ]
    
    # Return top k sorted by score descending
    return sorted(scored_songs, key=lambda x: x[1], reverse=True)[:k]


# ---------------------------------------------------------------------------
# AI-powered recommendation modes
# ---------------------------------------------------------------------------

_NAIVE_SYSTEM = (
    "You are a music recommender. The user describes their taste with a "
    "genre, mood, target energy (0-1), and acoustic preference. Recommend "
    "real songs from your training data that fit the profile. You have NO "
    "access to any catalog — pick from songs you already know.\n\n"
    "Return STRICT JSON of the form:\n"
    "{\"recommendations\": [{\"title\": str, \"artist\": str, "
    "\"score\": int (0-100), \"explanation\": str}]}\n"
    "Provide exactly k items. Score reflects how well it fits the profile. "
    "Explanations should be 1-2 sentences and reference the user's stated "
    "preferences."
)

_RAG_SYSTEM = (
    "You are a music recommender for a small fixed catalog. You will be "
    "given a user taste profile and a CANDIDATE list of songs from the "
    "catalog (with attributes). Re-rank and pick the best k for this user. "
    "Hard rules:\n"
    "1. You MUST pick songs ONLY from the candidate list. Never invent a "
    "title or artist.\n"
    "2. Each explanation MUST cite concrete attributes of the chosen song "
    "(genre, mood, energy value, etc.) and connect them to the user's "
    "profile.\n"
    "3. If the user's stated genre is not present in the candidates, say so "
    "honestly in the explanation and pick the closest available match by "
    "mood and energy.\n\n"
    "Return STRICT JSON of the form:\n"
    "{\"recommendations\": [{\"title\": str, \"artist\": str, "
    "\"score\": int (0-100), \"explanation\": str}]}"
)


def _format_profile(user_prefs: Dict) -> str:
    return (
        f"- favorite_genre: {user_prefs.get('genre')}\n"
        f"- favorite_mood: {user_prefs.get('mood')}\n"
        f"- target_energy (0-1): {user_prefs.get('energy')}\n"
        f"- likes_acoustic: {user_prefs.get('likes_acoustic', False)}"
    )


def recommend_naive_llm(user_prefs: Dict, k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Naive LLM mode: ask Gemini to recommend songs from its training data.
    No catalog access. Returns the same (song_dict, score, reasons_str)
    tuple shape used by the rule-based path so main.py can render either.
    """
    try:
        from src.llm_client import generate_json
    except ImportError:
        from llm_client import generate_json

    prompt = (
        f"User taste profile:\n{_format_profile(user_prefs)}\n\n"
        f"Recommend exactly {k} songs."
    )
    parsed = generate_json(prompt=prompt, system=_NAIVE_SYSTEM)
    items = parsed.get("recommendations", [])[:k]

    results: List[Tuple[Dict, float, str]] = []
    for item in items:
        song_dict = {
            "title": item.get("title", "Unknown"),
            "artist": item.get("artist", "Unknown"),
        }
        score = float(item.get("score", 0))
        reasons = item.get("explanation", "")
        results.append((song_dict, score, reasons))
    return results


def recommend_rag(
    user_prefs: Dict,
    songs: List[Dict],
    k: int = 5,
    candidates: int = 10,
) -> List[Tuple[Dict, float, str]]:
    """
    RAG mode: retrieve top `candidates` from the catalog using the existing
    rule-based scorer, then ask Gemini to re-rank to k and write personalized
    explanations grounded in the candidate attributes.
    """
    try:
        from src.llm_client import generate_json
    except ImportError:
        from llm_client import generate_json

    retrieved = recommend_songs(user_prefs, songs, k=candidates)
    candidate_rows = [
        {
            "title": s["title"],
            "artist": s["artist"],
            "genre": s["genre"],
            "mood": s["mood"],
            "energy": s["energy"],
            "tempo_bpm": s["tempo_bpm"],
            "valence": s["valence"],
            "danceability": s["danceability"],
            "acousticness": s["acousticness"],
            "rule_based_score": round(score, 2),
        }
        for s, score, _ in retrieved
    ]
    catalog_titles = {row["title"] for row in candidate_rows}

    prompt = (
        f"User taste profile:\n{_format_profile(user_prefs)}\n\n"
        f"CANDIDATE songs (you must pick exactly {k} from this list, "
        f"and only this list):\n"
        f"{json.dumps(candidate_rows, indent=2)}\n\n"
        f"Re-rank for this user and explain each pick using the listed attributes."
    )
    parsed = generate_json(prompt=prompt, system=_RAG_SYSTEM)
    items = parsed.get("recommendations", [])[:k]

    results: List[Tuple[Dict, float, str]] = []
    title_to_song = {s["title"]: s for s, _, _ in retrieved}
    for item in items:
        title = item.get("title", "")
        if title not in catalog_titles:
            # LLM violated the constraint; surface it instead of silently dropping.
            song_dict = {"title": f"[INVALID: {title}]", "artist": item.get("artist", "")}
        else:
            song_dict = title_to_song[title]
        score = float(item.get("score", 0))
        reasons = item.get("explanation", "")
        results.append((song_dict, score, reasons))
    return results
