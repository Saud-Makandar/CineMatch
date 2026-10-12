"""
recommender.py
--------------
Core recommendation algorithm for CineMatch:
Item-Based Collaborative Filtering using K-Nearest Neighbors (KNN) with Cosine Distance.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


class CineMatchKNN:
    """
    Item-Based Collaborative Filtering Recommender using KNN and Cosine Distance.
    """

    def __init__(self, metric="cosine", algorithm="brute"):
        self.metric = metric
        self.algorithm = algorithm
        self.model = NearestNeighbors(metric=self.metric, algorithm=self.algorithm)
        self.pivot_df = None
        self.sparse_matrix = None
        self.metadata_df = None
        self.is_fitted = False

    def fit(self, pivot_df, sparse_matrix, metadata_df=None):
        """
        Fits NearestNeighbors on the compressed sparse movie-user matrix.
        """
        self.pivot_df = pivot_df
        self.sparse_matrix = sparse_matrix
        self.metadata_df = metadata_df
        if metadata_df is not None:
            # Handle duplicate movie titles in metadata by keeping the one with most ratings
            clean_meta = metadata_df.sort_values(by="num_ratings", ascending=False).drop_duplicates(subset=["title"])
            self.metadata_lookup = clean_meta.set_index("title").to_dict(orient="index")
        else:
            self.metadata_lookup = {}
        self.model.fit(self.sparse_matrix)
        self.is_fitted = True
        return self

    def find_movie(self, query):
        """
        Finds the closest movie title in the matrix using exact, transposed article,
        token overlap, and substring matching.
        Returns:
            matched_title (str or None), all_candidates (list)
        """
        if not self.is_fitted or self.pivot_df is None:
            raise RuntimeError("Model is not fitted yet.")

        if not query or not str(query).strip():
            return None, []

        clean_query = str(query).strip().lower()
        titles = list(self.pivot_df.index)

        # 1. Exact match
        for t in titles:
            if t.lower() == clean_query:
                return t, [t]

        # 2. Handle articles like "The Matrix" -> "Matrix, The"
        article_variations = [clean_query]
        for article in ["the ", "a ", "an "]:
            if clean_query.startswith(article):
                core_title = clean_query[len(article):].strip()
                article_name = article.strip()
                article_variations.append(f"{core_title}, {article_name}")
                article_variations.append(core_title)

        for var in article_variations:
            matches = [t for t in titles if var in t.lower()]
            if matches:
                # Prioritize matches where core title starts at the beginning
                matches.sort(key=lambda x: (not x.lower().startswith(var), len(x)))
                return matches[0], matches[:10]

        # 3. Token-based word overlap (filtering common stop words)
        stopwords = {"the", "a", "an", "and", "of", "in", "to", "movie", "film", "part", "vol"}
        meaningful_query_words = set(w for w in clean_query.split() if w not in stopwords)
        if not meaningful_query_words:
            meaningful_query_words = set(clean_query.split())

        scored_matches = []
        for t in titles:
            title_words = set(t.lower().replace("(", "").replace(")", "").replace(",", "").replace(":", "").split())
            overlap = len(meaningful_query_words.intersection(title_words))
            # Require significant overlap (at least 50% of words or 2+ words for multi-word queries)
            min_required = max(1, int(np.ceil(len(meaningful_query_words) * 0.5)))
            if overlap >= min_required and (overlap >= 2 if len(meaningful_query_words) >= 3 else True):
                scored_matches.append((overlap, len(t), t))

        if scored_matches:
            # Sort by highest overlap, then shortest length
            scored_matches.sort(key=lambda item: (-item[0], item[1]))
            top_candidates = [item[2] for item in scored_matches[:10]]
            return top_candidates[0], top_candidates

        return None, []

    def recommend(self, movie_name, top_k=5, genre_filter=None):
        """
        Returns top_k recommendations for a given movie title.
        """
        if not self.is_fitted:
            return {"status": "error", "message": "Model is not trained.", "recommendations": []}

        matched_title, candidates = self.find_movie(movie_name)
        if not matched_title:
            return {
                "status": "not_found",
                "searched_query": movie_name,
                "message": f"Movie '{movie_name}' not found in the matrix.",
                "candidates": [],
                "recommendations": []
            }

        # Index lookup
        movie_idx = self.pivot_df.index.get_loc(matched_title)
        movie_vec = self.pivot_df.iloc[movie_idx, :].values.reshape(1, -1)

        # Retrieve extra neighbors if genre filtering is applied
        n_neighbors_to_fetch = min(len(self.pivot_df), max(top_k * 4 + 1, 30))
        distances, indices = self.model.kneighbors(movie_vec, n_neighbors=n_neighbors_to_fetch)

        flat_dist = distances.flatten()
        flat_idx = indices.flatten()

        recs = []
        meta_dict = self.metadata_lookup

        for d, idx in zip(flat_dist, flat_idx):
            candidate_title = self.pivot_df.index[idx]
            if candidate_title == matched_title:
                continue  # Skip self

            sim_score = max(0.0, float(1.0 - d))
            sim_pct = round(sim_score * 100.0, 2)

            meta = meta_dict.get(candidate_title, {})
            genres = meta.get("genres", "Unknown")
            year = meta.get("year", "N/A")
            avg_rating = meta.get("avg_rating", 0.0)
            num_ratings = meta.get("num_ratings", 0)

            # Apply genre filter if provided
            if genre_filter:
                if isinstance(genre_filter, str):
                    genre_filter = [genre_filter]
                movie_genres = [g.strip().lower() for g in genres.split("|")]
                if not any(gf.lower() in movie_genres for gf in genre_filter):
                    continue

            recs.append({
                "title": candidate_title,
                "similarity_score": round(sim_score, 4),
                "similarity_percentage": sim_pct,
                "genres": genres,
                "year": int(year) if pd.notna(year) and year != "N/A" else None,
                "avg_rating": avg_rating,
                "num_ratings": num_ratings
            })

            if len(recs) == top_k:
                break

        return {
            "status": "success",
            "searched_query": movie_name,
            "matched_movie": matched_title,
            "candidates": candidates,
            "recommendations": recs
        }

