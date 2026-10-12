"""
data_processing.py
------------------
Data ingestion, validation, cleaning, and matrix construction
for CineMatch Recommendation System.
"""

import os
import re
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def load_raw_data(movies_path="movies.csv", ratings_path="ratings.csv"):
    """
    Loads raw CSV files and verifies schema.
    """
    if not os.path.exists(movies_path):
        raise FileNotFoundError(f"Movies file not found: {movies_path}")
    if not os.path.exists(ratings_path):
        raise FileNotFoundError(f"Ratings file not found: {ratings_path}")

    movies_df = pd.read_csv(movies_path)
    ratings_df = pd.read_csv(ratings_path)

    # Validate required columns
    required_movie_cols = {"movieId", "title", "genres"}
    required_rating_cols = {"userId", "movieId", "rating"}

    if not required_movie_cols.issubset(movies_df.columns):
        raise ValueError(f"movies.csv is missing required columns: {required_movie_cols - set(movies_df.columns)}")
    if not required_rating_cols.issubset(ratings_df.columns):
        raise ValueError(f"ratings.csv is missing required columns: {required_rating_cols - set(ratings_df.columns)}")

    return movies_df, ratings_df


def clean_and_merge_data(movies_df, ratings_df):
    """
    Cleans duplicates, handles nulls, extracts movie release year,
    and returns a merged interactions dataframe along with movie metadata.
    """
    # 1. Deduplicate
    clean_movies = movies_df.drop_duplicates(subset=["movieId"]).copy()
    clean_ratings = ratings_df.drop_duplicates(subset=["userId", "movieId"]).copy()

    # 2. Drop unnecessary columns
    if "timestamp" in clean_ratings.columns:
        clean_ratings = clean_ratings.drop(columns=["timestamp"])

    # 3. Handle missing values
    clean_movies = clean_movies.dropna(subset=["movieId", "title"])
    clean_ratings = clean_ratings.dropna(subset=["userId", "movieId", "rating"])

    # Ensure valid rating range (0.5 to 5.0)
    clean_ratings = clean_ratings[(clean_ratings["rating"] >= 0.5) & (clean_ratings["rating"] <= 5.0)]

    # 4. Extract release year from title e.g. "Toy Story (1995)" -> 1995
    def extract_year(title):
        match = re.search(r"\((\d{4})\)", str(title))
        return int(match.group(1)) if match else np.nan

    clean_movies["year"] = clean_movies["title"].apply(extract_year)

    # 5. Calculate global movie statistics
    stats = clean_ratings.groupby("movieId").agg(
        num_ratings=("rating", "count"),
        avg_rating=("rating", "mean")
    ).reset_index()

    metadata_df = pd.merge(clean_movies, stats, on="movieId", how="left")
    metadata_df["num_ratings"] = metadata_df["num_ratings"].fillna(0).astype(int)
    metadata_df["avg_rating"] = metadata_df["avg_rating"].fillna(0.0).round(2)

    # 6. Merge ratings with movie details
    merged_df = pd.merge(clean_ratings, metadata_df[["movieId", "title", "genres", "year"]], on="movieId", how="inner")

    return merged_df, metadata_df


def compute_sparsity(n_users, n_items, n_interactions):
    """
    Computes matrix sparsity percentage:
    Sparsity = (1 - (interactions / (users * items))) * 100
    """
    if n_users == 0 or n_items == 0:
        return 100.0
    total_elements = n_users * n_items
    sparsity = (1.0 - (n_interactions / total_elements)) * 100.0
    return round(sparsity, 2)


def filter_sparse_interactions(merged_df, min_movie_ratings=10, min_user_ratings=5):
    """
    Filters out infrequent movies and users to alleviate cold-start noise
    and sparsity in collaborative filtering.
    """
    # Filter movies
    movie_counts = merged_df.groupby("movieId")["rating"].count()
    qualified_movies = movie_counts[movie_counts >= min_movie_ratings].index
    filtered = merged_df[merged_df["movieId"].isin(qualified_movies)].copy()

    # Filter users
    user_counts = filtered.groupby("userId")["rating"].count()
    qualified_users = user_counts[user_counts >= min_user_ratings].index
    filtered = filtered[filtered["userId"].isin(qualified_users)].copy()

    return filtered


def create_pivot_matrix(filtered_df):
    """
    Creates Item-User pivot matrix:
    Rows = Movie Titles
    Columns = User IDs
    Values = Ratings (unrated filled with 0)
    Returns:
      pivot_df: DataFrame
      sparse_matrix: csr_matrix
    """
    pivot_df = filtered_df.pivot_table(
        index="title",
        columns="userId",
        values="rating"
    ).fillna(0)

    sparse_matrix = csr_matrix(pivot_df.values)
    return pivot_df, sparse_matrix


def prepare_data_pipeline(movies_path="movies.csv", ratings_path="ratings.csv", min_movie_ratings=10):
    """
    Convenience function executing end-to-end data preparation.
    """
    movies_raw, ratings_raw = load_raw_data(movies_path, ratings_path)
    merged_df, metadata_df = clean_and_merge_data(movies_raw, ratings_raw)
    filtered_df = filter_sparse_interactions(merged_df, min_movie_ratings=min_movie_ratings)
    pivot_df, sparse_matrix = create_pivot_matrix(filtered_df)

    initial_sparsity = compute_sparsity(
        ratings_raw["userId"].nunique(),
        movies_raw["movieId"].nunique(),
        len(ratings_raw)
    )

    filtered_sparsity = compute_sparsity(
        filtered_df["userId"].nunique(),
        filtered_df["movieId"].nunique(),
        len(filtered_df)
    )

    return {
        "raw_movies": movies_raw,
        "raw_ratings": ratings_raw,
        "merged_df": merged_df,
        "metadata_df": metadata_df,
        "filtered_df": filtered_df,
        "pivot_df": pivot_df,
        "sparse_matrix": sparse_matrix,
        "initial_sparsity": initial_sparsity,
        "filtered_sparsity": filtered_sparsity
    }
