"""
=============================================================
  Movie Genre Classification — FastAPI Backend
  Serves predictions from the trained TF-IDF + NB model.
=============================================================
"""

import os
import re
import sys
import json
import joblib
import ast
import pandas as pd
import httpx
import difflib

# Fix Windows console encoding for emoji/unicode output
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ── TMDB API config ────────────────────────────────────────
# Public read-access token (v4 Bearer) — free TMDB account key
# Set env var TMDB_API_KEY or uses this public fallback bearer token
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")
TMDB_BASE    = "https://api.themoviedb.org/3"
TMDB_IMG     = "https://image.tmdb.org/t/p/w500"

# ── paths ──────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")
DATA_DIR = os.path.join(BASE_DIR, "..", "data")



# ── load model artifacts ───────────────────────────────────
print("🔄  Loading model artifacts …")
model = joblib.load(os.path.join(MODEL_DIR, "genre_model.pkl"))
tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
mlb = joblib.load(os.path.join(MODEL_DIR, "label_binarizer.pkl"))

metrics_path = os.path.join(MODEL_DIR, "metrics.json")
if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        metrics_data = json.load(f)
else:
    metrics_data = {}

print("✅  Model loaded!  Genres:", list(mlb.classes_))

# ── load movies dataset for recommendations ────────────────
print("🔄  Loading movies dataset for recommendations …")
movies_df = pd.DataFrame()
try:
    movies_csv = os.path.join(DATA_DIR, "TMDB_movie_dataset_v11.csv")
    df = pd.read_csv(movies_csv, low_memory=False)

    # Genres are plain comma-separated strings: "Action, Drama, Thriller"
    def parse_genres(val):
        if pd.isna(val) or str(val).strip() == "":
            return []
        return [g.strip() for g in str(val).split(",") if g.strip()]

    df['genres_parsed'] = df['genres'].apply(parse_genres)
    df['year'] = df['release_date'].astype(str).str[:4]
    df['vote_count']   = pd.to_numeric(df['vote_count'],   errors='coerce').fillna(0)
    df['popularity']   = pd.to_numeric(df['popularity'],   errors='coerce').fillna(0)
    df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce').fillna(0)

    # Only keep released movies with meaningful votes
    if 'status' in df.columns:
        df = df[df['status'].str.lower() == 'released']
    df = df[df['vote_count'] >= 5].copy()

    movies_df = df.sort_values(by='popularity', ascending=False)
    print(f"✅  Movies dataset loaded! ({len(movies_df):,} valid movies)")
except Exception as e:
    print(f"⚠️  Could not load movies for recommendations: {e}")


# ── preprocessing (mirrors training) ──────────────────────
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text


# ═══════════════════════════════════════════════════════════
#  FAST API  APP
# ═══════════════════════════════════════════════════════════
app = FastAPI(
    title="Movie Genre Classifier API",
    description="Classify movies into genres using plot descriptions (TF-IDF + Naive Bayes)",
    version="1.0.0",
)

# CORS — allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── request / response models ─────────────────────────────
class PredictionRequest(BaseModel):
    description: str


class GenrePrediction(BaseModel):
    genre: str
    confidence: float


class PredictionResponse(BaseModel):
    genres: list[GenrePrediction]
    input_preview: str


class RecommendationRequest(BaseModel):
    genres: list[str]


class RecommendationResponse(BaseModel):
    recommendations: list[dict]


class RecentFilmsResponse(BaseModel):
    films: list[dict]


# ── endpoints ──────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok", "model": "TF-IDF + MultinomialNB (OneVsRest)"}


@app.get("/api/genres")
def get_genres():
    return {"genres": list(mlb.classes_)}


@app.get("/api/metrics")
def get_metrics():
    if not metrics_data:
        raise HTTPException(status_code=404, detail="Metrics not found. Train the model first.")
    return metrics_data


@app.post("/api/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    if not req.description or len(req.description.strip()) < 10:
        raise HTTPException(
            status_code=400,
            detail="Description too short. Please provide at least a sentence.",
        )

    cleaned = preprocess(req.description)
    vec = tfidf.transform([cleaned])

    # Get probability scores for each genre
    # MultinomialNB inside OneVsRest exposes predict_proba
    try:
        probas = model.predict_proba(vec)[0]
    except AttributeError:
        # Fallback: use decision_function
        probas = model.decision_function(vec)[0]
        # Normalize to 0-1
        probas = (probas - probas.min()) / (probas.max() - probas.min() + 1e-9)

    # Build results sorted by confidence
    results = []
    for genre, prob in zip(mlb.classes_, probas):
        # Only include genres with > 20% confidence
        if prob > 0.20:
            results.append(GenrePrediction(genre=genre, confidence=round(float(prob), 4)))

    # If nothing passed threshold, return top 3
    if not results:
        top_indices = probas.argsort()[-3:][::-1]
        for idx in top_indices:
            results.append(
                GenrePrediction(
                    genre=mlb.classes_[idx],
                    confidence=round(float(probas[idx]), 4),
                )
            )

    results.sort(key=lambda x: x.confidence, reverse=True)

    return PredictionResponse(
        genres=results,
        input_preview=req.description[:200],
    )


@app.post("/api/recommend", response_model=RecommendationResponse)
def recommend_movies(req: RecommendationRequest):
    if movies_df.empty:
        raise HTTPException(status_code=503, detail="Movie database not available for recommendations.")
    
    if not req.genres:
        return RecommendationResponse(recommendations=[])

    target_genres = set([g.lower() for g in req.genres])
    
    def score_match(movie_genres):
        mg = set([g.lower() for g in movie_genres])
        return len(target_genres.intersection(mg))
        
    scores = movies_df['genres_parsed'].apply(score_match)
    
    match_df = movies_df[scores > 0].copy()
    if match_df.empty:
        return RecommendationResponse(recommendations=[])
        
    match_df['match_score'] = scores[scores > 0]
    
    # Simple recommendation sorting (most overlaps first, then highly popular)
    top_matches = match_df.sort_values(by=['match_score', 'popularity'], ascending=[False, False]).head(50)
    
    recs = []
    added_franchises = set()
    added_titles = []
    
    for _, row in top_matches.iterrows():
        title = str(row['title']) if pd.notna(row['title']) else "Unknown"
        
        # Deduplication logic to avoid multiple parts of a franchise
        # 1. Get the base name (before colon or dash)
        import re
        base_name = re.split(r'[:\-]', title)[0].strip().lower()
        
        is_duplicate = False
        
        # Check against base names
        if base_name in added_franchises:
            is_duplicate = True
            
        # Check against existing full titles (in case a sequel doesn't use a colon)
        for added in added_titles:
            # If they share the first 3 words, treat as same franchise (e.g. "The Lord of the Rings")
            words1 = title.lower().split()
            words2 = added.lower().split()
            if len(words1) >= 3 and len(words2) >= 3 and words1[:3] == words2[:3]:
                is_duplicate = True
                break
            
            similarity = difflib.SequenceMatcher(None, title.lower(), added.lower()).ratio()
            if similarity > 0.65:
                is_duplicate = True
                break
        
        if is_duplicate:
            continue
            
        added_franchises.add(base_name)
        added_titles.append(title)
        
        # Build TMDB poster URL if poster_path column exists
        poster_path = ""
        if 'poster_path' in row and pd.notna(row.get('poster_path')):
            raw = str(row['poster_path']).strip()
            if raw and raw != 'nan':
                poster_path = TMDB_IMG + (raw if raw.startswith('/') else '/' + raw)
        
        recs.append({
            "title": title,
            "year": str(row['year']) if pd.notna(row['year']) else "",
            "overview": str(row['overview']) if pd.notna(row['overview']) else "",
            "genres": row['genres_parsed'],
            "rating": round(float(row['vote_average']), 1),
            "match_score": int(row['match_score']),
            "poster": poster_path
        })
        
        if len(recs) >= 10:
            break
        
    return RecommendationResponse(recommendations=recs)


@app.get("/api/search", response_model=RecommendationResponse)
async def search_movie(title: str):
    """
    Search for movies by title in the local dataset.
    """
    if movies_df.empty:
        return RecommendationResponse(recommendations=[])
        
    query = title.strip().lower()
    if not query:
        return RecommendationResponse(recommendations=[])
        
    # Partial matching
    matches = movies_df[movies_df['title'].str.lower().str.contains(query, na=False)].head(12)
    
    recs = []
    for _, row in matches.iterrows():
        poster_path = ""
        if 'poster_path' in row and pd.notna(row.get('poster_path')):
            raw = str(row['poster_path']).strip()
            if raw and raw != 'nan':
                poster_path = TMDB_IMG + (raw if raw.startswith('/') else '/' + raw)
                
        recs.append({
            "title": str(row['title']),
            "year": str(row['year']) if pd.notna(row['year']) else "",
            "overview": str(row['overview']) if pd.notna(row['overview']) else "",
            "genres": row['genres_parsed'],
            "rating": round(float(row['vote_average']), 1),
            "poster": poster_path
        })
        
    return RecommendationResponse(recommendations=recs)


@app.get("/api/recent-films", response_model=RecentFilmsResponse)
async def recent_films():
    """
    Fetch recently released / now-playing movies from TMDB.
    Falls back to a curated static list if the key is absent.
    """
    if not TMDB_API_KEY:
        # ── Static fallback (no API key needed) ──────────────
        static = [
            {"id": 1652880, "title": "Dhurandhar the Revenge", "year": "2026", "genres": ["Action", "Thriller", "Crime"], "rating": 9.5, "poster": "https://upload.wikimedia.org/wikipedia/en/c/c8/Dhurandhar-_The_Revenge_poster.jpg", "overview": "An action-packed saga of retribution and vengeance against all odds."},
            {"id": 533535, "title": "Deadpool & Wolverine", "year": "2024", "genres": ["Action", "Comedy", "Science Fiction"], "rating": 7.8, "poster": "https://image.tmdb.org/t/p/w500/8cdWjvZQUExUUTzyp4t6EDMubfO.jpg", "overview": "A listless Wade Wilson toils away in civilian life with his days as the morally flexible mercenary, Deadpool, behind him."},
            {"id": 857598, "title": "Pushpa 2 - The Rule", "year": "2024", "genres": ["Action", "Crime", "Drama"], "rating": 8.4, "poster": "https://image.tmdb.org/t/p/w500/1T21FblunT0y8fz7YaW8JMYgUKm.jpg", "overview": "The clash between Pushpa and Bhanwar Singh continues in this explosive sequel."},
            {"id": 1022789, "title": "Inside Out 2", "year": "2024", "genres": ["Animation", "Family", "Comedy", "Drama"], "rating": 7.6, "poster": "https://image.tmdb.org/t/p/w500/vpnVM9B6NMmQpWeZvzLvDESb2QY.jpg", "overview": "Teenager Riley's mind headquarters is undergoing a sudden demolition to make room for new Emotions!"},
            {"id": 872906, "title": "Jawan", "year": "2023", "genres": ["Action", "Thriller"], "rating": 8.1, "poster": "https://image.tmdb.org/t/p/w500/w4mPBAfZS5yIXOcqEiEOL8fnuQG.jpg", "overview": "A man is driven by a personal vendetta to rectify the wrongs in society while keeping a promise made years ago."},
            {"id": 76600, "title": "Avatar: Fire and Ash", "year": "2025", "genres": ["Science Fiction", "Action", "Adventure"], "rating": 8.5, "poster": "https://image.tmdb.org/t/p/w500/aabwWZWx6z1aYP4PX2ADvbDKktd.jpg", "overview": "The ongoing saga of Pandora continues as Jake Sully and Neytiri encounter a new clan of Na'vi."},
            {"id": 864692, "title": "Pathaan", "year": "2023", "genres": ["Action", "Thriller", "Adventure"], "rating": 7.9, "poster": "https://image.tmdb.org/t/p/w500/arf00BkwvXo0CFKbaD9OpqdE4Nu.jpg", "overview": "An Indian secret agent comes out of exile to stop a rogue operative from unleashing a deadly virus."},
            {"id": 693134, "title": "Dune: Part Two", "year": "2024", "genres": ["Science Fiction", "Adventure"], "rating": 8.3, "poster": "https://image.tmdb.org/t/p/w500/6izwz7rsy95ARzTR3poZ8H6c5pp.jpg", "overview": "Follow the mythic journey of Paul Atreides as he unites with Chani."}
        ]
        return RecentFilmsResponse(films=static)

    # ── Live TMDB fetch ───────────────────────────────────
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.get(
                f"{TMDB_BASE}/movie/now_playing",
                params={"api_key": TMDB_API_KEY, "language": "en-US", "page": 1}
            )
            resp.raise_for_status()
            data = resp.json()

        # Fetch genre map
        async with httpx.AsyncClient(timeout=8) as client:
            g_resp = await client.get(
                f"{TMDB_BASE}/genre/movie/list",
                params={"api_key": TMDB_API_KEY, "language": "en-US"}
            )
            g_resp.raise_for_status()
            genre_map = {g["id"]: g["name"] for g in g_resp.json().get("genres", [])}

        films = []
        for m in data.get("results", [])[:12]:
            poster = (TMDB_IMG + m["poster_path"]) if m.get("poster_path") else ""
            genres = [genre_map.get(gid, "") for gid in m.get("genre_ids", []) if genre_map.get(gid)]
            films.append({
                "id": m["id"],
                "title": m.get("title", "Unknown"),
                "year": m.get("release_date", "")[:4],
                "overview": m.get("overview", ""),
                "genres": genres,
                "rating": round(m.get("vote_average", 0), 1),
                "poster": poster
            })
        return RecentFilmsResponse(films=films)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"TMDB fetch failed: {e}")


# ── serve frontend ─────────────────────────────────────────
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


# ── run ────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("\n🚀  Starting server at  http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
