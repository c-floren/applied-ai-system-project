"""
Evaluate Naive LLM vs RAG modes against the catalog.

Measures the grounding rate (the fraction of returned titles that actually
exist in data/songs.csv) for each mode across every built-in user profile.

Run: python -m src.evaluate
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "songs.csv"

try:
    from src.recommender import load_songs, recommend_naive_llm, recommend_rag
    from src.llm_client import LLMError
    from src.main import BUILTIN_PROFILES
except ImportError:
    from recommender import load_songs, recommend_naive_llm, recommend_rag
    from llm_client import LLMError
    from main import BUILTIN_PROFILES


@dataclass
class ModeResult:
    mode: str
    calls_attempted: int = 0
    calls_succeeded: int = 0
    total_items: int = 0
    in_catalog: int = 0
    hallucinations: List[Tuple[str, str, str]] = field(default_factory=list)  # (profile, title, artist)
    failures: List[Tuple[str, str]] = field(default_factory=list)  # (profile, error_msg)

    @property
    def grounding_rate(self) -> float:
        return self.in_catalog / self.total_items if self.total_items else 0.0

    @property
    def success_rate(self) -> float:
        return self.calls_succeeded / self.calls_attempted if self.calls_attempted else 0.0


def evaluate() -> Dict[str, ModeResult]:
    songs = load_songs(str(DATA_PATH))
    catalog_titles = {s["title"] for s in songs}
    results = {"naive": ModeResult("naive"), "rag": ModeResult("rag")}

    for profile_name, prefs in BUILTIN_PROFILES.items():
        for mode_key, recommend_fn in (
            ("naive", lambda p: recommend_naive_llm(p, k=5)),
            ("rag", lambda p: recommend_rag(p, songs, k=5, candidates=10)),
        ):
            r = results[mode_key]
            r.calls_attempted += 1
            try:
                items = recommend_fn(prefs)
            except LLMError as e:
                r.failures.append((profile_name, str(e)))
                continue
            r.calls_succeeded += 1
            r.total_items += len(items)
            for song, _, _ in items:
                title = song.get("title", "")
                # The recommender marks RAG constraint violations with [INVALID: ...].
                cleaned = title[len("[INVALID: "):-1] if title.startswith("[INVALID: ") else title
                if cleaned in catalog_titles:
                    r.in_catalog += 1
                else:
                    r.hallucinations.append((profile_name, title, song.get("artist", "")))
    return results


def _format_section(r: ModeResult) -> str:
    lines = [f"\nMode: {r.mode.upper()}"]
    lines.append(f"  Calls succeeded:     {r.calls_succeeded}/{r.calls_attempted} ({r.success_rate:.0%})")
    lines.append(f"  Items returned:      {r.total_items}")
    lines.append(f"  Titles in catalog:   {r.in_catalog}/{r.total_items} ({r.grounding_rate:.0%})  <- grounding rate")
    lines.append(f"  Hallucinated titles: {len(r.hallucinations)}")
    if r.hallucinations:
        sample = r.hallucinations[: min(4, len(r.hallucinations))]
        sample_str = ", ".join(f"{title} ({artist})" for _, title, artist in sample)
        more = "" if len(r.hallucinations) <= 4 else f" (+{len(r.hallucinations) - 4} more)"
        lines.append(f"  Sample hallucinations: {sample_str}{more}")
    if r.failures:
        lines.append(f"  Failures: {len(r.failures)}")
        for profile, err in r.failures:
            lines.append(f"    - {profile}: {err[:100]}")
    return "\n".join(lines)


def print_report(results: Dict[str, ModeResult]) -> None:
    print("=" * 60)
    print("  EVALUATION REPORT")
    print("=" * 60)
    print(f"Profiles tested: {len(BUILTIN_PROFILES)}")
    print(f"Modes tested:    2 (naive, rag)")
    print(f"Total calls:     {sum(r.calls_attempted for r in results.values())}")

    for mode_key in ("naive", "rag"):
        print(_format_section(results[mode_key]))

    print("\n" + "-" * 60)
    naive = results["naive"]
    rag = results["rag"]
    print(
        f"Conclusion: RAG grounding rate {rag.grounding_rate:.0%}, "
        f"Naive LLM grounding rate {naive.grounding_rate:.0%}. "
        f"RAG produced {len(naive.hallucinations) - len(rag.hallucinations)} fewer hallucinations."
    )
    print("-" * 60)


def main() -> None:
    results = evaluate()
    print_report(results)


if __name__ == "__main__":
    main()
