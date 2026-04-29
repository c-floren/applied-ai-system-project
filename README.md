# SoundMatch 2.0 — AI-Augmented Music Recommender

A music recommender that compares a deterministic rule-based scorer against two LLM-driven approaches — a naive LLM call and a retrieval-augmented (RAG) call — over the same 19-song catalog. The goal is to make the differences between these approaches concrete and visible: where each one wins, where each one fails, and how grounding an LLM in real catalog data changes its behavior.

This project matters because most public discussion of "AI recommenders" treats the LLM as a black box. SoundMatch 2.0 puts a transparent rule-based system, a naively prompted LLM, and a properly retrieval-augmented LLM side by side on the same inputs, so the trade-offs can be inspected rather than asserted.

---

## Original Project: SoundMatch 1.0 (Modules 1-3)

The starting point for this project was **SoundMatch 1.0**, a content-based music recommender built across CodePath AI110 Modules 1-3. It represented songs and a user "taste profile" as structured data and scored each song on a weighted formula — Genre 40%, Mood 30%, Energy closeness 15%, Danceability 10%, Acousticness 5% — to return the top 5 picks with a bullet-list of reasons. It was deliberately rule-based so the scoring logic could be inspected and explained, but the model card surfaced its central weakness: 70% of the score came from binary string-match on genre and mood, so any user whose preferred genre wasn't in the catalog (the "Ghost Genre" adversarial test) silently fell through to weak results.

---

## What's New in 2.0

| | SoundMatch 1.0 | SoundMatch 2.0 |
|---|---|---|
| Recommender output | Rule-based scorer | Gemini 2.5-flash (Naive LLM **or** RAG) |
| Catalog | Read-only `songs.csv` | User can add songs through the CLI |
| User profiles | 8 hardcoded | 8 built-ins + user-created (persisted) |
| Explanations | Joined-string list of matched features | LLM-generated, attribute-grounded sentences |
| Failure on unknown genre ("k-pop") | Silent fallback to weak matches | RAG: explicitly says "k-pop is not in the catalog" and picks closest match |
| Tests | None implemented (stubs only) | 2 pytest tests pinning the rule-based scorer (now reused as the RAG retriever) |

The rule-based scorer **was not deleted**. It now serves as the **retriever** inside RAG mode — the same scoring logic that used to be the user-facing answer is now the candidate-selection step under the LLM. This makes the comparison between modes honest: same retriever, different reasoners.

---

## Architecture Overview

```mermaid
flowchart LR
    User([User]) -->|"picks profile + mode"| CLI["CLI Menu<br/>src/main.py"]

    Catalog[("data/songs.csv<br/>18+ songs")]

    CLI -->|"profile only"| Naive["Gemini 2.5-flash<br/>system: recommend from<br/>training data (no catalog)"]

    CLI -->|"profile"| Retriever["Rule-based Scorer<br/>recommend_songs() in<br/>recommender.py · top 10"]
    Catalog --> Retriever
    Retriever -->|"top 10 candidates<br/>with attributes"| RAG["Gemini 2.5-flash<br/>system: pick ONLY from<br/>candidate list"]

    Cache[(".cache/<br/>sha256-keyed JSON")]
    Naive <-.->|"cache lookup"| Cache
    RAG <-.->|"cache lookup"| Cache

    Naive --> Output["Top-5 songs<br/>+ explanations"]
    RAG --> Output
    Output --> Display["CLI output<br/>shown to user"]

    Tests["pytest<br/>tests/test_recommender.py"] -.->|"locks rule-based<br/>scoring logic"| Retriever
    Adversarial["Adversarial profiles<br/>Ghost Genre, Sad but Hyper, ..."] -.->|"stress-test inputs"| CLI
    Display -.->|"compare Naive vs RAG<br/>across profiles"| Human([Human Reviewer])
    Human -.->|"writes findings into<br/>model_card.md"| ModelCard["model_card.md"]

    classDef agent fill:#fde68a,stroke:#d97706,color:#000
    classDef data fill:#dbeafe,stroke:#2563eb,color:#000
    classDef eval fill:#dcfce7,stroke:#16a34a,color:#000
    class Naive,RAG agent
    class Catalog,Cache,ModelCard data
    class Tests,Adversarial,Human eval
```

**Three layers:**

1. **Data layer (blue)** — the song catalog (`data/songs.csv`), a disk cache for LLM responses (`.cache/`), and the human-written model card.
2. **Agent layer (yellow)** — two distinct Gemini calls. **Naive LLM** receives only the user profile and recommends from its training data (no catalog access). **RAG** receives the user profile *plus* the top 10 catalog candidates with full attributes, and is constrained by its system prompt to pick only from that candidate list and to cite concrete attributes in every explanation.
3. **Evaluation layer (green)** — pytest pins the rule-based scoring logic so the RAG retriever can't silently regress; eight adversarial profiles (Ghost Genre, Sad but Hyper, Zero Energy Rocker, Wants Everything, etc.) feed deliberately weird inputs to expose where each mode fails; a human reviewer compares the Naive and RAG outputs and writes findings into `model_card.md`.

LLM responses are cached on disk by `sha256(model + system_prompt + user_prompt)`, so repeat runs of the same input are free, deterministic, and survive across sessions. This is what makes the side-by-side comparisons reproducible.

---

## Setup

**Requirements:** Python 3.10+, a Google Gemini API key (free tier is fine), and a terminal.

```bash
# 1. Clone and enter the repo
git clone <your-fork-url>
cd applied-ai-system-final

# 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate         # Windows
# source .venv/bin/activate    # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Gemini API key
cp .env.example .env
# then edit .env and paste your key from https://aistudio.google.com/app/apikey

# 5. Run the app
python -m src.main
```

The top-level menu lets you:
- **Get recommendations** — pick a user profile (built-in or one you've created), then a mode (Naive LLM or RAG).
- **Add a song to the catalog** — fill in attributes (title, artist, genre, mood, energy, tempo, valence, danceability, acousticness). The new song is appended to `data/songs.csv` and is available immediately.
- **Create a new user profile** — name your profile and fill in genre, mood, target energy, and acoustic preference. Saved to `data/user_profiles.json` and merged into the picker on every future run.

**Other useful invocations:**

```bash
python -m src.main --all                            # Sweep every built-in profile through every mode
python -m src.main --data path/to/custom_songs.csv  # Use a different catalog
pytest                                              # Run the test suite (2 tests, < 1 second)
```

---

## Sample Interactions

The three examples below all use the catalog as shipped in `data/songs.csv`. Outputs are direct transcripts (lightly trimmed for length).

### Example 1: Standard profile, RAG mode (the well-behaved case)

**Input:** profile `High-Energy Pop` (genre=pop, mood=happy, energy=0.9, likes_acoustic=False), mode `RAG`.

**Output (top 3 of 5):**
```
#1  Sunrise City — Neon Echo                                 Score: 98
    This song is an excellent match for your profile. It's squarely in
    the 'pop' genre, features a 'happy' mood, and its energy of 0.82 is
    very close to your target of 0.9. Additionally, its low acousticness
    (0.18) aligns with your preference against acoustic sounds.

#2  Gym Hero — Max Pulse                                     Score: 90
    While the mood is 'intense' rather than 'happy', this song is still a
    strong fit due to its 'pop' genre and very high energy (0.93), which
    is almost exactly your target of 0.9.

#3  Rooftop Lights — Indigo Parade                           Score: 85
    A great choice because it has a 'happy' mood. While its genre is
    'indie pop' rather than pure 'pop', it's a close match.
```

**What it shows:** every pick is a real song in the catalog, every explanation cites concrete attribute values (energy 0.82, acousticness 0.18, genre 'indie pop'), and the LLM acknowledges trade-offs ("the mood is 'intense' rather than 'happy'") instead of pretending every pick is perfect. This is the behavior the RAG system prompt was designed to produce.

### Example 2: Adversarial profile, Naive LLM mode (the hallucination case)

**Input:** profile `Ghost Genre (Nonexistent)` (genre=k-pop, mood=happy, energy=0.7, likes_acoustic=False), mode `Naive LLM`.

**Output:**
```
#1  Red Flavor — Red Velvet                                  Score: 93
#2  Dynamite — BTS                                           Score: 90
#3  DALLA DALLA — ITZY                                       Score: 88
#4  Aju Nice (Very Nice) — SEVENTEEN                         Score: 85
```

**What it shows:** Gemini happily returns four real K-pop songs from its training data. They are *plausible* recommendations for a real human, but **none of them exist in our 18-song catalog**. If this output were piped into a downstream "play this song" feature, every pick would 404. This is the textbook hallucination failure mode that motivates RAG.

### Example 3: Adversarial profile, RAG mode (the headline result)

**Input:** same `Ghost Genre` profile (genre=k-pop), mode `RAG`.

**Output (top 3 of 5):**
```
#1  Sunrise City — Neon Echo                                 Score: 95
    While 'k-pop' is not available in the catalog, 'Sunrise City' is an
    excellent match due to its 'pop' genre, which is a close stylistic
    relative. It perfectly aligns with your 'happy' mood and has a high
    energy of 0.82, very close to your target of 0.7.

#2  Rooftop Lights — Indigo Parade                           Score: 90
    This song, though 'indie pop' instead of k-pop, is a strong contender
    because it matches your 'happy' mood and has an energy level of 0.76,
    which is very close to your target of 0.7.

#3  Electric Dreams — SynthWave Peak                         Score: 85
    Although 'k-pop' is not available, 'Electric Dreams' from the
    'electronic' genre offers a high-energy experience (0.88).
```

**What it shows:** every recommended title exists in the catalog, and the LLM honestly states up front that "k-pop is not available in the catalog" before justifying each substitute. This is the RAG advantage made explicit. The original rule-based scorer in SoundMatch 1.0 *also* returned only catalog songs in this case, but it gave no signal that anything was wrong — it silently weighted on mood and energy and reported scores like 56 with no explanation of why.

---

## Design Decisions

**Why keep the rule-based scorer instead of replacing it.** The cleanest move would have been to delete `recommend_songs` and let Gemini do everything. Instead it now serves as the retriever inside RAG mode. Two reasons: (1) on a 18-song catalog, sending the entire catalog to the LLM every call would work, but on a realistic catalog (millions of tracks) it wouldn't — the retrieval step is the part that actually scales. Building it on day one, even with a fake-tiny catalog, made the architecture honest. (2) Reusing the rule-based scorer means the `pytest` tests on it directly protect the RAG retriever; if someone later "improves" the scoring weights and breaks the retrieval, the tests catch it.

**Why Gemini and not Claude or GPT-4.** Gemini 2.5-flash has a generous free tier, native JSON-mode output (`response_mime_type="application/json"`), and a configurable thinking budget that can be set to 0 to keep response latency and token cost predictable. The wrapper in `src/llm_client.py` is intentionally thin so swapping providers would be straightforward; the Gemini-specific bits are isolated to the `client.models.generate_content` call.

**Why disk-cache LLM responses by prompt hash.** Two motivations: (1) developing an LLM-driven feature requires running the same prompt many times, and burning quota on identical inputs is wasteful. (2) A class project benefits enormously from deterministic output for screenshots and the writeup — caching by `sha256(model + system + prompt)` means the same input always produces the same output across sessions, even though the underlying model is non-deterministic.

**Why a CLI menu instead of a web UI.** Streamlit is in `requirements.txt` from the original project but unused. The CLI keeps friction low for graders and reviewers — no port management, no browser state, just `python -m src.main`. The menu structure (top-level → sub-flow) is verbose to type but reads cleanly in screenshots, which matters for the model card.

**Why persist user-created songs and profiles to disk.** The natural alternative was session-only storage. Persistence (`data/songs.csv` for songs, `data/user_profiles.json` for profiles) is more useful in practice — a user who creates a profile once shouldn't have to recreate it on every run — and the implementation cost was small (one new file, two helper functions, an extra `--profiles` flag).

**The trade-off I'd flag.** Naive LLM mode is *strictly worse* than RAG mode for this task. I kept it because the comparison is what the project is *about* — without Naive mode, there's no concrete demonstration of why RAG matters. But in a production system, I'd never ship Naive mode as a user-facing option; it would only ever exist as an evaluation harness.

---

## Testing Summary

**What's tested automatically.** `tests/test_recommender.py` has two pytest tests on the rule-based core: one that confirms `Recommender.recommend()` returns songs sorted by score, and one that confirms `Recommender.explain_recommendation()` returns a non-empty string. They run in under a second and lock the scoring logic that the RAG retriever depends on.

**What's tested manually.** Eight user profiles are encoded into `src/main.py`'s built-in profile picker — four "standard" profiles and four "adversarial" ones (Ghost Genre, Sad but Hyper, Zero Energy Rocker, Wants Everything) explicitly designed to break the system. Each profile was run through both Naive LLM and RAG modes, and the outputs were compared by hand. The findings are written up in `model_card.md`.

**What worked.** RAG mode does what it was designed to do — every output title is a real catalog song, and explanations consistently cite concrete attributes. The Ghost Genre comparison (Example 2 vs Example 3 above) is a clean demonstration of grounding mattering. The disk cache makes reruns instant and reproducible, which made the writeup itself much easier to do.

**What didn't work the first time.** The first version used `gemini-1.5-flash`, which has been retired from the v1beta API — every call returned a 404 until I switched to `gemini-2.5-flash` and migrated from the deprecated `google-generativeai` SDK to the current `google-genai` SDK. The 2.5 model also defaults to a non-zero thinking budget, which silently truncated the JSON response on longer prompts; setting `thinking_config=ThinkingConfig(thinking_budget=0)` and `max_output_tokens=4096` fixed it. Both bugs surfaced as parse failures rather than obvious errors, which is a reminder that LLM-facing code needs more defensive error handling than typical glue code.

**What I learned.** Two things stand out. First, the *system prompt* in RAG mode is doing most of the work — without "you MUST pick songs ONLY from the candidate list", Gemini freely mixes catalog songs with hallucinated ones. The constraint has to be in the prompt, not just in the calling code. Second, looking at adversarial profiles in both modes side-by-side made bias and failure modes far more obvious than evaluating either mode alone — the contrast is what's informative.

---

## Reflection

The biggest realization is that "use AI" isn't a feature — it's an architectural choice with downstream consequences. The interesting question for this project wasn't "should I add an LLM?" but "where should the LLM sit in the pipeline?" Putting it at the *output* (Naive LLM) makes the system more articulate but unmoored from the data. Putting it at the *output, with a retriever in front* (RAG) makes it both articulate and grounded — but requires building and maintaining a retriever, which is essentially the same problem the rule-based system was already solving. The rule-based scorer didn't go away; it changed jobs.

The other thing I learned is how much the model card matters once an LLM is in the picture. With a deterministic rule-based system, you can audit it by reading the code. With an LLM, the only way to know how it actually behaves is to run it on a deliberate set of adversarial inputs and write down what you saw. That's not a one-time evaluation — it's something you'd want to repeat every time the model version changes (and in this project, that already happened once when 1.5-flash was retired).

If I had more time, the next steps would be: (1) replace the binary genre/mood matching in the retriever with a similarity score so "rock" can return "metal" candidates and the filter-bubble problem the original SoundMatch 1.0 model card identified is actually fixed; (2) add an evaluator agent that automatically diff's Naive vs RAG output across all profiles and flags hallucinations (titles not in the catalog) without a human in the loop; (3) wire in the Streamlit UI that's already in `requirements.txt` so the side-by-side comparison can be screenshotted in one frame instead of two.

---

## Project Layout

```
applied-ai-system-final/
├── src/
│   ├── main.py            # CLI menu + flow dispatch
│   ├── recommender.py     # Rule-based scorer (now RAG retriever) + AI mode functions + persistence helpers
│   └── llm_client.py      # Thin Gemini wrapper with disk-cached generate_json()
├── data/
│   ├── songs.csv          # Catalog (19 shipped songs + any you add)
│   └── user_profiles.json # Profiles you create at runtime (created on first save)
├── tests/
│   └── test_recommender.py
├── assets/
│   └── music_recommender_flowchart.mmd  # Mermaid source for the architecture diagram
├── .cache/                # LLM response cache (gitignored)
├── .env                   # API key (gitignored — copy from .env.example)
├── model_card.md          # Bias, limitations, and adversarial findings
├── requirements.txt
└── README.md
```
