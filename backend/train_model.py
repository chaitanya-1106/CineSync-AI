"""
==============================================================
  CineGenre AI — Model Training Script
  Dataset : TMDB_movie_dataset_v11.csv  (~1 million movies)
  Pipeline: TF-IDF  +  Multinomial Naive Bayes (One-vs-Rest)
==============================================================

HOW IT WORKS (plain English)
─────────────────────────────
1. Load the CSV and keep only rows that have both an overview
   (plot text) and at least one genre.

2. Build the feature text by combining the overview with the
   movie's keywords — more words = better signal for the model.

3. Convert that text into numbers using TF-IDF.
   TF-IDF gives higher weight to words that are rare across
   the whole dataset but common in a specific movie.

4. Train a Naive Bayes classifier wrapped in OneVsRestClassifier
   so it can predict MULTIPLE genres at once (multi-label).

5. Evaluate on a held-out test split and save three files:
     • tfidf_vectorizer.pkl  — the text → numbers transformer
     • genre_model.pkl       — the trained classifier
     • label_binarizer.pkl   — maps genre names ↔ 0/1 columns
   Plus a metrics.json that the web UI reads for the stats page.
"""

import os, sys, json, re
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    hamming_loss, f1_score, accuracy_score, classification_report
)

# ── Paths ──────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.abspath(__file__))
DATA    = os.path.join(BASE, "..", "data", "TMDB_movie_dataset_v11.csv")
OUT_DIR = os.path.join(BASE, "model")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Load & clean data ───────────────────────────────────
print("📂  Loading dataset …")
df = pd.read_csv(DATA, low_memory=False)
print(f"    Raw rows : {len(df):,}")

# Keep only movies with a plot AND at least one genre
df = df.dropna(subset=["overview", "genres"])
df = df[df["overview"].str.strip() != ""]
df = df[df["genres"].str.strip() != ""]

# Only use movies that have been released (no future or rumoured titles)
if "status" in df.columns:
    df = df[df["status"].str.lower() == "released"]

# Require a minimum number of votes so quality is decent
if "vote_count" in df.columns:
    df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce").fillna(0)
    df = df[df["vote_count"] >= 5]

print(f"    Clean rows: {len(df):,}")

# ── 2. Parse genres into a list per movie ──────────────────
# The genres column looks like: "Action, Adventure, Drama"
def parse_genres(raw):
    return [g.strip() for g in str(raw).split(",") if g.strip()]

df["genre_list"] = df["genres"].apply(parse_genres)

# ── 3. Build feature text  (overview + keywords) ──────────
def clean(text):
    """Lowercase and remove non-alphabetic characters."""
    return re.sub(r"[^a-z\s]", "", str(text).lower())

keywords_col = "keywords" if "keywords" in df.columns else None

if keywords_col:
    df["features"] = (
        df["overview"].apply(clean) + " " +
        df[keywords_col].fillna("").apply(clean)
    )
else:
    df["features"] = df["overview"].apply(clean)

# ── 4. Encode labels (genre_list → binary matrix) ─────────
print("🏷️   Encoding genres …")
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(df["genre_list"])
print(f"    Genre classes ({len(mlb.classes_)}): {list(mlb.classes_)}")

# ── 5. Train / test split ──────────────────────────────────
X_train, X_test, Y_train, Y_test = train_test_split(
    df["features"], Y, test_size=0.2, random_state=42
)
print(f"    Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── 6. TF-IDF vectorisation ────────────────────────────────
print("🔢  Vectorising text with TF-IDF …")
tfidf = TfidfVectorizer(
    max_features=15_000,   # keep the 15k most informative words
    ngram_range=(1, 2),    # single words AND two-word phrases
    sublinear_tf=True,     # compress very high term frequencies
    min_df=3,              # ignore words that appear in fewer than 3 movies
)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec  = tfidf.transform(X_test)
print(f"    Vocabulary size: {len(tfidf.vocabulary_):,}")

# ── 7. Train Naive Bayes (One-vs-Rest for multi-label) ─────
print("🧠  Training Multinomial Naive Bayes …")
model = OneVsRestClassifier(MultinomialNB(alpha=0.1), n_jobs=-1)
model.fit(X_train_vec, Y_train)

# ── 8. Evaluate ────────────────────────────────────────────
print("📊  Evaluating on test set …")
Y_pred = model.predict(X_test_vec)

hl   = hamming_loss(Y_test, Y_pred)
f1m  = f1_score(Y_test, Y_pred, average="micro",  zero_division=0)
f1M  = f1_score(Y_test, Y_pred, average="macro",  zero_division=0)
acc  = accuracy_score(Y_test, Y_pred)           # subset (exact-match) accuracy

print(f"\n  ✅ Subset Accuracy : {acc*100:.2f}%")
print(f"  ✅ Hamming Loss    : {hl:.4f}")
print(f"  ✅ F1 Micro        : {f1m*100:.2f}%")
print(f"  ✅ F1 Macro        : {f1M*100:.2f}%")

# Per-genre F1
report = classification_report(
    Y_test, Y_pred,
    target_names=mlb.classes_,
    output_dict=True,
    zero_division=0
)
per_genre = {g: {"f1": round(report[g]["f1-score"], 4),
                  "precision": round(report[g]["precision"], 4),
                  "recall": round(report[g]["recall"], 4)}
             for g in mlb.classes_ if g in report}

# ── 9. Save artefacts ──────────────────────────────────────
print("\n💾  Saving model artefacts …")
joblib.dump(tfidf,  os.path.join(OUT_DIR, "tfidf_vectorizer.pkl"))
joblib.dump(model,  os.path.join(OUT_DIR, "genre_model.pkl"))
joblib.dump(mlb,    os.path.join(OUT_DIR, "label_binarizer.pkl"))

metrics = {
    "subset_accuracy": round(acc,  4),
    "hamming_loss":    round(hl,   4),
    "f1_micro":        round(f1m,  4),
    "f1_macro":        round(f1M,  4),
    "train_size":      int(len(X_train)),
    "test_size":       int(len(X_test)),
    "vocab_size":      len(tfidf.vocabulary_),
    "num_genres":      len(mlb.classes_),
    "genres":          list(mlb.classes_),
    "per_genre":       per_genre,
}
with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("✅  Done! Files saved to backend/model/")
print(f"    genre_model.pkl · tfidf_vectorizer.pkl · label_binarizer.pkl · metrics.json")
