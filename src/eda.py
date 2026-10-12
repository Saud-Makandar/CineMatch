"""
eda.py
------
Exploratory Data Analysis (EDA) and Statistical Analysis
using Pandas, Matplotlib, and Seaborn.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_descriptive_statistics(ratings_df):
    """
    Computes core statistical measures for ratings:
    Mean, Median, Mode, Variance, Standard Deviation, Skewness, Min, Max, Quantiles.
    """
    ratings = ratings_df["rating"]

    stats = {
        "count": int(ratings.count()),
        "mean": float(ratings.mean()),
        "median": float(ratings.median()),
        "mode": float(ratings.mode()[0]),
        "std_dev": float(ratings.std()),
        "variance": float(ratings.var()),
        "skewness": float(ratings.skew()),
        "min": float(ratings.min()),
        "q25": float(ratings.quantile(0.25)),
        "q75": float(ratings.quantile(0.75)),
        "iqr": float(ratings.quantile(0.75) - ratings.quantile(0.25)),
        "max": float(ratings.max())
    }
    return stats


def calculate_user_movie_stats(merged_df):
    """
    Computes activity stats per user and popularity stats per movie.
    """
    user_counts = merged_df.groupby("userId")["rating"].count()
    movie_counts = merged_df.groupby("movieId")["rating"].count()

    stats = {
        "user_activity": {
            "mean": float(user_counts.mean()),
            "median": float(user_counts.median()),
            "std": float(user_counts.std()),
            "min": int(user_counts.min()),
            "max": int(user_counts.max())
        },
        "movie_popularity": {
            "mean": float(movie_counts.mean()),
            "median": float(movie_counts.median()),
            "std": float(movie_counts.std()),
            "min": int(movie_counts.min()),
            "max": int(movie_counts.max())
        }
    }
    return stats


def calculate_genre_breakdown(metadata_df):
    """
    Explodes the pipe-separated genres column and computes
    movie counts and average rating per genre.
    """
    records = []
    for _, row in metadata_df.iterrows():
        genres = str(row["genres"]).split("|")
        for g in genres:
            if g and g != "(no genres listed)":
                records.append({
                    "genre": g,
                    "movieId": row["movieId"],
                    "avg_rating": row["avg_rating"],
                    "num_ratings": row["num_ratings"]
                })

    genre_df = pd.DataFrame(records)
    if genre_df.empty:
        return pd.DataFrame(columns=["genre", "movie_count", "mean_rating"])

    summary = genre_df.groupby("genre").agg(
        movie_count=("movieId", "count"),
        mean_rating=("avg_rating", "mean"),
        total_ratings=("num_ratings", "sum")
    ).reset_index().sort_values(by="movie_count", ascending=False)

    summary["mean_rating"] = summary["mean_rating"].round(2)
    return summary


# ==========================================
# VISUALIZATION FUNCTIONS (Matplotlib & Seaborn)
# ==========================================

def set_style():
    """Applies clean styling for charts."""
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update({"font.size": 10})


def plot_rating_distribution(ratings_df):
    """
    Plots the frequency distribution of ratings (0.5 to 5.0) with KDE.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    
    sns.histplot(
        ratings_df["rating"],
        bins=10,
        kde=True,
        color="#3b82f6",
        edgecolor="black",
        ax=ax
    )
    mean_val = ratings_df["rating"].mean()
    median_val = ratings_df["rating"].median()

    ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.2f}")
    ax.axvline(median_val, color="green", linestyle=":", linewidth=2, label=f"Median: {median_val:.2f}")

    ax.set_title("Distribution of Movie Ratings (0.5 - 5.0)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Rating Value")
    ax.set_ylabel("Frequency (Count)")
    ax.set_xticks(np.arange(0.5, 5.5, 0.5))
    ax.legend()
    fig.tight_layout()
    return fig


def plot_top_movies(metadata_df, top_n=10):
    """
    Bar chart of the most-rated movies.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    top_movies = metadata_df.sort_values(by="num_ratings", ascending=False).head(top_n)

    # Shorten titles if too long
    clean_titles = [t[:32] + "..." if len(t) > 35 else t for t in top_movies["title"]]

    sns.barplot(
        x=top_movies["num_ratings"],
        y=clean_titles,
        palette="viridis",
        hue=clean_titles,
        legend=False,
        ax=ax
    )

    ax.set_title(f"Top {top_n} Most Rated Movies", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Ratings Received")
    ax.set_ylabel("Movie Title")
    fig.tight_layout()
    return fig


def plot_genre_distribution(genre_summary, top_n=10):
    """
    Horizontal bar chart of top movie genres.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))

    top_genres = genre_summary.head(top_n)
    sns.barplot(
        x=top_genres["movie_count"],
        y=top_genres["genre"],
        palette="crest",
        hue=top_genres["genre"],
        legend=False,
        ax=ax
    )

    ax.set_title(f"Top {top_n} Movie Genres by Title Count", fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Movies")
    ax.set_ylabel("Genre")
    fig.tight_layout()
    return fig


def plot_user_activity(merged_df):
    """
    Log-scale distribution showing user engagement (long-tail behavior).
    """
    set_style()
    fig, ax = plt.subplots(figsize=(7, 4))

    user_counts = merged_df.groupby("userId")["rating"].count()

    sns.histplot(user_counts, bins=40, color="#8b5cf6", edgecolor="black", kde=True, ax=ax)
    ax.set_yscale("log")
    ax.set_title("User Activity Distribution (Log Scale)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Ratings Given per User")
    ax.set_ylabel("Number of Users (Log Scale)")
    fig.tight_layout()
    return fig


def save_all_plots(output_dir="static/plots", movies_path="movies.csv", ratings_path="ratings.csv"):
    """
    Utility function to generate and save all plots to disk for reports/offline viewing.
    """
    from src.data_processing import prepare_data_pipeline

    os.makedirs(output_dir, exist_ok=True)
    data = prepare_data_pipeline(movies_path, ratings_path)

    f1 = plot_rating_distribution(data["raw_ratings"])
    f1.savefig(os.path.join(output_dir, "rating_distribution.png"), dpi=200)
    plt.close(f1)

    f2 = plot_top_movies(data["metadata_df"])
    f2.savefig(os.path.join(output_dir, "top_movies.png"), dpi=200)
    plt.close(f2)

    genre_df = calculate_genre_breakdown(data["metadata_df"])
    f3 = plot_genre_distribution(genre_df)
    f3.savefig(os.path.join(output_dir, "genre_distribution.png"), dpi=200)
    plt.close(f3)

    f4 = plot_user_activity(data["merged_df"])
    f4.savefig(os.path.join(output_dir, "user_activity.png"), dpi=200)
    plt.close(f4)

    print(f"All EDA plots saved successfully to {output_dir}/")
