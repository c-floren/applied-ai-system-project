# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  

**SoundMatch 1.0**  

---

## 2. Intended Use  

SoundMatch recommends the top 5 songs from a small catalog based on a user's favorite genre, mood, energy level, and acoustic preference. It assumes each user has one favorite genre and one favorite mood. It also assumes the user can describe their energy preference as a number between 0 and 1. This is a classroom project, not a real product. It is meant to help us learn how recommender systems work and where they can go wrong.  

---

## 3. How the Model Works  

Each song has six attributes: genre, mood, energy, tempo, danceability, and acousticness. The user tells us their favorite genre, favorite mood, target energy level, and whether they like acoustic music.

The system scores every song on a scale of 0 to 100. Genre is worth the most — 40 points if the song's genre matches exactly, 0 if it does not. Mood is next at 30 points, same all-or-nothing rule. Energy closeness is worth up to 15 points — the closer the song's energy is to what the user wants, the more points it gets. Danceability adds up to 10 points based on how close a song is to a default target of 0.7. Acousticness adds up to 5 points — if the user likes acoustic music, all songs get full credit here; if not, acoustic songs lose points.

After scoring every song, we sort them from highest to lowest and return the top 5. We also list the reasons each song scored well, so the user can see why it was picked.

---

## 4. Data  

The catalog has 18 songs stored in a CSV file. There are 15 genres represented: pop, lofi, rock, jazz, folk, metal, electronic, hip-hop, classical, country, funk, indie pop, reggae, synthwave, and ambient. Moods include happy, chill, intense, relaxed, melancholic, nostalgic, confident, energetic, peaceful, angry, moody, playful, and focused.

We did not add or remove any songs from the original dataset. The catalog is heavily skewed — lofi has 3 songs and pop has 2, but most genres only have 1 song each. This means some users get variety in their results while others are stuck with a single match. Major genres like R&B, Latin, and hip-hop subgenres are underrepresented or missing entirely. The data mostly reflects a Western, English-language music perspective.  

---

## 5. Strengths  

The system works best when the user's preferences align with a genre that has multiple songs in the catalog. The Chill Lofi profile got excellent results — Library Rain and Midnight Coding both scored above 97 and felt like exactly what a lofi listener would want. The High-Energy Pop profile also worked well, with Sunrise City scoring 97 and matching on genre, mood, energy, and danceability all at once.

The scoring logic is fully transparent. For every recommendation, we can explain exactly why a song was picked — which features matched and how many points each one contributed. This makes it easy to trust or question the results, unlike a black-box algorithm where you just have to accept what it gives you.  

---

## 6. Limitations and Bias 

Where the system struggles or behaves unfairly. 

The most significant weakness is that genre and mood together account for 70% of the total score, and both are scored as binary all-or-nothing matches — a song either matches exactly or receives zero credit. This means numerical preferences like energy are largely cosmetic; our adversarial test with a "Zero Energy Rocker" profile (rock/intense but energy 0.05) still ranked the high-energy Storm Runner (energy 0.91) as the top pick at 86.70 points, completely ignoring the user's stated energy preference. This creates a filter bubble where users are locked into their declared genre and never exposed to adjacent styles — a rock fan will never discover metal or indie, even though those genres are musically similar. In a real product, this kind of rigid matching would narrow listeners' taste over time rather than helping them explore new music.  

---

## 7. Evaluation  

How you checked whether the recommender behaved as expected. 

We tested eight user profiles: four standard profiles (High-Energy Pop, Chill Lofi, Deep Intense Rock, Nostalgic Jazz) and four adversarial edge cases (Sad but Hyper, Ghost Genre, Zero Energy Rocker, Wants Everything). For the standard profiles, we verified that the top recommendations matched the expected genre and mood — for example, the Chill Lofi user received Library Rain and Midnight Coding as the top two results, both lofi/chill songs with high acousticness, which aligned with our intuition. The adversarial profiles revealed that the scoring logic can be "tricked": the Ghost Genre profile (k-pop, which doesn't exist in the dataset) still returned results with a top score of 56.40, falling back silently to mood and energy matching without any indication that the genre preference was ignored. The most surprising finding was how little energy mattered — the Zero Energy Rocker wanted energy at 0.05 but still got Storm Runner (energy 0.91) as the top pick because the 70% genre+mood weight completely overpowered the energy mismatch. We did not use numeric accuracy metrics, but the adversarial testing approach proved more informative than standard profiles alone for exposing weaknesses in the scoring logic.

---

## 8. What We Added (AI Augmentation)

We replaced the rule-based ranker as the user-facing answer with a Gemini-powered pipeline that runs in one of two modes selected from a CLI menu:

- **Naive LLM mode** — Gemini receives only the user profile and recommends 5 songs from its training data. It has no access to the catalog. We kept this mode specifically so the comparison with RAG is honest: it shows what an LLM does without grounding.
- **RAG mode** — The original `recommend_songs` scorer runs first as a *retriever* and returns the top 10 candidates from `data/songs.csv` with their full attributes. Those candidates are injected into Gemini's prompt, and the system instruction forbids inventing titles outside the candidate list and requires every explanation to cite concrete attributes (energy, mood, genre). The output is parsed back to records that exist in our catalog.

This setup is deliberately structured so the rule-based logic isn't thrown away — it becomes the retrieval layer underneath the AI reasoner. Same retriever, different reasoners, fair comparison.

**What changed in behavior:**
- The Ghost Genre (k-pop) profile, which used to fail silently with weak fallback results, now produces honest output in RAG mode: Gemini explains that k-pop is not in the catalog and picks the closest available match by mood and energy. Naive LLM, by contrast, hallucinates k-pop songs that don't exist in our data — a clean illustration of why retrieval matters.
- Adversarial profiles like Zero Energy Rocker and Sad but Hyper get more nuanced explanations from RAG that acknowledge the contradiction in the user's preferences, rather than the rule-based scorer's silent overweighting of genre+mood.

**Engineering notes:**
- Gemini calls are cached to disk (`.cache/`, keyed by `sha256(model + system + prompt)`) so reruns of the same profile+mode produce identical output and don't burn API quota. This keeps screenshots reproducible.
- `temperature=0.3` and `response_mime_type="application/json"` give us low-variance, parseable output.
- The `Recommender` class still wraps the rule-based scorer so the existing pytest suite continues to pass.

**Still future work:**
- Use similar genres instead of exact match in the retriever (rock ↔ metal).
- Let users set their own danceability target.
- Add a diversity rule so the top 5 isn't dominated by one cluster.
- Use tempo and valence in the retriever scoring.

---

## 9. Personal Reflection  

Building this recommender taught me that the weights you choose matter more than the features themselves. We had energy, danceability, and acousticness in the data, but because genre and mood carried 70% of the score, those numerical features barely made a difference. The system looked sophisticated on paper but was really just a genre filter in disguise.

The most surprising thing was how one song — Gym Hero — kept appearing across completely different user profiles. A happy pop fan, a sad pop fan, a rock fan, and even a zero-energy listener all got Gym Hero in their top 5. It made me realize how real apps like Spotify probably have the same problem: certain songs with popular tags get pushed to everyone, not because they truly fit, but because the algorithm rewards broad labels over specific taste.

This project changed how I think about recommendations I get in real life. Now when Spotify suggests something, I wonder: did it pick this because it actually matches what I want, or because it matches one broad category and the rest of my preferences were quietly ignored?  
