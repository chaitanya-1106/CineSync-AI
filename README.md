# 🎬 CineGenre AI — Movie Genre Classification

> **AI Applications Project** — NLP Classification using TF-IDF + Naive Bayes

## 📋 Overview

This project classifies movies into genres based on their **plot summaries and keywords** using Natural Language Processing (NLP) techniques. Built with Python, scikit-learn, and FastAPI.

| Feature | Detail |
|---|---|
| **Problem Type** | Multi-label NLP Classification |
| **Dataset** | The Movies Dataset (Kaggle) — 45,000+ movies |
| **Input Features** | Plot summaries, keywords |
| **ML Techniques** | TF-IDF Vectorization, Multinomial Naive Bayes |
| **Metrics** | Accuracy, F1 Score, Hamming Loss |
| **Framework** | FastAPI (Backend), Vanilla HTML/CSS/JS (Frontend) |

## 🛠️ Tech Stack

- **Python 3.10+** — Core language
- **scikit-learn** — TF-IDF, Naive Bayes, evaluation metrics
- **NLTK** — Stopwords, lemmatization
- **Pandas / NumPy** — Data manipulation
- **FastAPI + Uvicorn** — REST API server
- **HTML / CSS / JavaScript** — Web frontend

## 📁 Project Structure

```
movie-genre-classifier/
├── backend/
│   ├── app.py              # FastAPI server
│   ├── train_model.py      # ML training pipeline
│   ├── requirements.txt    # Python dependencies
│   └── model/              # Saved model artifacts
├── frontend/
│   ├── index.html          # Web interface
│   ├── style.css           # Dark theme UI
│   └── script.js           # API interactions
├── data/
│   └── (dataset CSVs)
└── README.md
```

## 🚀 Setup & Installation

### 1. Download Dataset
Download **The Movies Dataset** from Kaggle:
https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset

Place these files in the `data/` folder :
- `movies_metadata.csv`
- `keywords.csv`
- `credits.csv`

### 2. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 3. Setup & Train the Model
```bash
cd backend
python train_model.py
cd ..
```
This will preprocess the dataset, train the AI, and save model artifacts.

### 4. Run the Application
Open a terminal in the project root (`movie-genre-classifier`) and simply run:
```bash
python run.py
```
This single command will:
1. Start the FastAPI backend engine and serve the user interface on `http://localhost:8000`
2. You can open your browser to `http://localhost:8000` to interact with the project.

## 🧠 Technical Approach

### Pipeline
1. **Data Loading** — Parse 45K movies from CSV, extract genres and keywords from JSON columns
2. **Text Preprocessing** — Lowercase → remove punctuation → remove stopwords → lemmatize
3. **Feature Engineering** — Combine plot overview + keywords into single text feature
4. **TF-IDF Vectorization** — Convert text to 5000-dimensional feature vectors (unigrams + bigrams)
5. **Classification** — OneVsRestClassifier wrapping MultinomialNB for multi-label prediction
6. **Evaluation** — Subset accuracy, Hamming loss, F1 scores (micro/macro/weighted)

### Why These Techniques?
- **TF-IDF** captures word importance relative to the corpus, ideal for genre-distinguishing terms
- **Naive Bayes** works exceptionally well with sparse text features and is computationally efficient
- **OneVsRest** enables multi-label classification (movies can have multiple genres)

## 📊 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/predict` | Predict genres from plot description |
| `GET` | `/api/genres` | List all genres |
| `GET` | `/api/metrics` | Training metrics |

### Example Prediction Request
```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"description": "A young wizard discovers magical powers and battles a dark lord"}'
```

## 📄 License
Academic project — for educational purposes only.
