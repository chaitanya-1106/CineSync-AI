# The CineGenre AI Project: Full Breakdown

This document provides a complete, top-to-bottom explanation of your university AI project. It breaks down what the project accomplishes and exactly how the underlying technologies work together to deliver the final application.

---

## 🎯 1. Project Objective and Overview
**The Goal:** To build a full-stack web application that uses Artificial Intelligence (Natural Language Processing) to automatically classify a movie's genres just by reading its plot summary. Secondly, to provide an AI-driven movie recommendation engine based on user-selected genre groupings.

**The Tech Stack:**
*   **Machine Learning:** Python, Scikit-Learn, Pandas.
*   **Backend Server:** Python, FastAPI, Uvicorn.
*   **Frontend UI:** Vanilla HTML5, CSS3, JavaScript.

---

## 💻 2. The Machine Learning Engine (`train_model.py`)
This script acts as the "brawn" of the project. It runs offline before the web server starts. Its job is to read thousands of movies, study their plots, and mathematically learn how certain words match up with certain genres.

### Step 1: Loading & Cleaning the Data
The script loads the **Kaggle Movies Dataset** containing over 45,000 films. Because the internet is messy, the code first cleans this data. It removes broken rows, safely parses messy JSON lists (extracting arrays of strings like `["Action", "Thriller"]`), and mashes the movie's `plot overview` and `keywords` into one giant string.
Finally, the script cleans the text down to its bare minimum format using **Regular Expressions (Regex)** (lowercasing strings, removing punctuation, and stripping out numbers).

### Step 2: The TF-IDF Vectorizer
Computers can't read English—they only understand numbers. To solve this, the script runs the cleaned text through a `TfidfVectorizer` (Term Frequency-Inverse Document Frequency). 
*   **How it works:** It takes the 5,000 most common words in your dataset and turns them into a massive mathematical matrix representing how frequently a word appears in a specific movie *compared* to how frequently it appears across all 45,000 movies. If the word "Wand" appears often in one plot, but rarely in all other movies, the vectorizer tags "Wand" with a very high mathematical weight!

### Step 3: The Naive Bayes Algorithm
Once the text is mathematical, the script feeds it into a `MultinomialNB` (Multinomial Naive Bayes) algorithm wrapped in a `OneVsRestClassifier`. 
*   **How it works:** Naive Bayes is a probability algorithm. It studies the vectors and calculates the mathematical probability of a certain text belonging to a class. (e.g. *If I see the word "Wand", there is a 95% chance this is a Fantasy movie*).

### Step 4: Saving Artifacts
Once the model is fully trained, it calculates its exact accuracy (saving it to `metrics.json`), and then literally freezes its "brain" into physical files (`genre_model.pkl` and `tfidf_vectorizer.pkl`) so the backend web server can use it later without having to relearn everything from scratch.

---

## 🧠 3. The Backend Server (`app.py`)
This is the "bridge" of your application. Built using **FastAPI**, this script runs a live server on `http://localhost:8000`.

### Upon Startup
When you turn the server on, it reaches into the `model/` folder and loads the `.pkl` files (the pre-trained AI brain) into its live memory. It also loads a lightweight version of the Kaggle CSV into memory for the recommendation engine.

### The Live Endpoints
The server sits and listens for the Frontend to request information from its specific URL "Endpoints":
1.  **`/api/predict`**: The Frontend sends a chunk of raw text. The Backend runs the `TF-IDF` math on it, generates the Naive Bayes probabilities, ignores any genres under a 20% confidence rating, and sends the matched genres back to the user!
2.  **`/api/recommend`**: The Frontend sends an array of genres (e.g. `["Action", "Sci-Fi"]`). The Backend loops through the Kaggle dataset, calculating an **"Overlap Count"** (how many of these targeted genres naturally intersect with each movie's meta-tags), sorts the movies by overlap score and popularity, and returns the top 10 matches.
3.  **`/api/metrics`**: Easily fetches the `metrics.json` file to power the dashboard.

---

## ✨ 4. The Interactive Frontend (HTML / CSS / JS)
This is the "face" of the project—the interactive Liquid Glass dashboard the user actually sees.

*   **`index.html`**: Creates the raw structure of the webpage, dividing it into sections for the live predictor, the recommender grid, and the internal accuracy dashboard.
*   **`style.css`**: Provides the premium aesthetic, forcing a strict dark mode, rounded glassy cards, soft interactive transitions, styling buttons, and color-coding individual genres (Red for Action, Pink for Animation, etc.).
*   **`script.js`**: The puppet master. This script controls the interactivity. When a user clicks the "Predict Genre" button, `script.js` prevents the webpage from reloading, snatches the text from the text box, packages it into a JSON bundle, fires an asynchronous `fetch()` request over to the Backend's `/api/predict` endpoint, waits for the response, and physically modifies the HTML elements in real-time to animate the results sliding onto the screen!
