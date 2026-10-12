"""
movie_recommender.py
====================
Simple Flask Web Service for CineMatch KNN Movie Recommender.
Runs the backend API for the React Single-Page Web Application.
"""

import os
import sys
import pandas as pd
from flask import Flask, request, jsonify

# Ensure local src package is discoverable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.data_processing import prepare_data_pipeline
from src.recommender import CineMatchKNN

# Create Flask application
app = Flask(__name__)

# Enable CORS for frontend requests
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


# Load data and train KNN model
print("[*] Ingesting dataset and training KNN recommender model...")
movies_csv = os.path.join(CURRENT_DIR, "movies.csv")
ratings_csv = os.path.join(CURRENT_DIR, "ratings.csv")

data = prepare_data_pipeline(movies_csv, ratings_csv, min_movie_ratings=10)
knn = CineMatchKNN(metric="cosine", algorithm="brute")
knn.fit(data["pivot_df"], data["sparse_matrix"], data["metadata_df"])
print(f"[+] CineMatch KNN Ready! ({data['pivot_df'].shape[0]} movies in matrix)")


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "online",
        "service": "CineMatch KNN Movie Recommender",
        "movies_in_matrix": data["pivot_df"].shape[0]
    })


@app.route("/api/recommend", methods=["GET"])
def get_recommendations():
    """Generates KNN movie recommendations."""
    movie = request.args.get("movie", "").strip()
    top_k = int(request.args.get("top_k", 6))

    if not movie:
        return jsonify({"status": "error", "message": "Please enter a movie title."}), 400

    res = knn.recommend(movie, top_k=top_k)

    # Attach seed movie details if matched
    seed_details = None
    if res.get("matched_movie"):
        matched = res["matched_movie"]
        meta = knn.metadata_lookup.get(matched, {})
        year_val = meta.get("year")
        seed_details = {
            "title": matched,
            "genres": meta.get("genres", "Unknown"),
            "year": int(year_val) if pd.notna(year_val) and str(year_val).isdigit() else None,
            "avg_rating": float(meta.get("avg_rating", 0.0)),
            "num_ratings": int(meta.get("num_ratings", 0))
        }

    return jsonify({
        **res,
        "seed_details": seed_details
    })


@app.route("/api/movies", methods=["GET"])
def search_movies():
    """Search/autocomplete movie titles."""
    query = request.args.get("q", "").strip().lower()
    limit = int(request.args.get("limit", 6))
    titles = list(data["pivot_df"].index)

    if query:
        matches = [t for t in titles if query in t.lower()][:limit]
    else:
        matches = titles[:limit]

    formatted = []
    for title in matches:
        meta = knn.metadata_lookup.get(title, {})
        formatted.append({
            "title": title,
            "avg_rating": float(meta.get("avg_rating", 0.0)),
            "num_ratings": int(meta.get("num_ratings", 0))
        })

    return jsonify({"movies": formatted})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"[*] Starting CineMatch web service on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
