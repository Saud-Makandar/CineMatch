"""
evaluation.py
-------------
Evaluation module for CineMatch recommendation model.
Computes offline Precision@K and Recall@K on actual user test interactions
for CineMatch Item-Based Collaborative Filtering (KNN).
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


def train_test_split_by_user(filtered_df, test_ratio=0.2, min_ratings=15, random_state=42):
    """
    Performs temporal/stratified user-level train-test split:
    For active users (>= min_ratings), reserves test_ratio fraction of ratings for testing.
    """
    np.random.seed(random_state)
    user_counts = filtered_df.groupby("userId")["movieId"].count()
    active_users = user_counts[user_counts >= min_ratings].index

    train_records = []
    test_records = []

    for user_id, group in filtered_df.groupby("userId"):
        if user_id in active_users:
            shuffled = group.sample(frac=1.0, random_state=random_state)
            n_test = max(1, int(len(shuffled) * test_ratio))
            test_records.append(shuffled.iloc[:n_test])
            train_records.append(shuffled.iloc[n_test:])
        else:
            train_records.append(group)

    train_df = pd.concat(train_records, ignore_index=True)
    test_df = pd.concat(test_records, ignore_index=True) if test_records else pd.DataFrame()

    return train_df, test_df, active_users


def evaluate_models(filtered_df, k_values=(5, 10), relevance_threshold=3.5, sample_users=150, random_state=42):
    """
    Evaluates CineMatch KNN model on real test interactions.
    
    Returns:
        summary_df: DataFrame with Precision@K and Recall@K for CineMatch KNN.
        detailed_stats: dict of computed values.
    """
    train_df, test_df, active_users = train_test_split_by_user(
        filtered_df, test_ratio=0.2, min_ratings=15, random_state=random_state
    )

    # 1. Build train pivot matrix for KNN
    train_pivot = train_df.pivot_table(index="title", columns="userId", values="rating").fillna(0)
    train_sparse = csr_matrix(train_pivot.values)
    knn_model = NearestNeighbors(metric="cosine", algorithm="brute")
    knn_model.fit(train_sparse)

    # 2. Sample evaluation users to keep computation fast & deterministic
    eval_users = list(active_users)
    if len(eval_users) > sample_users:
        np.random.seed(random_state)
        eval_users = list(np.random.choice(eval_users, size=sample_users, replace=False))

    results = {
        k: {
            "knn_precision": [],
            "knn_recall": []
        }
        for k in k_values
    }

    max_k = max(k_values)
    titles_in_matrix = set(train_pivot.index)

    for uid in eval_users:
        user_test = test_df[(test_df["userId"] == uid) & (test_df["rating"] >= relevance_threshold)]
        relevant_test_movies = set(user_test["title"]).intersection(titles_in_matrix)
        if not relevant_test_movies:
            continue

        # Find user's top-rated seed movie in train set
        user_train = train_df[(train_df["userId"] == uid) & (train_df["rating"] >= relevance_threshold)]
        if user_train.empty:
            continue

        seed_movie = user_train.sort_values(by="rating", ascending=False).iloc[0]["title"]
        if seed_movie not in titles_in_matrix:
            continue

        # Get KNN recommendations for seed movie
        movie_idx = train_pivot.index.get_loc(seed_movie)
        seed_vec = train_pivot.iloc[movie_idx, :].values.reshape(1, -1)
        distances, indices = knn_model.kneighbors(seed_vec, n_neighbors=max_k + 1)

        knn_recs = [train_pivot.index[i] for i in indices.flatten()[1:]]  # exclude seed

        # Calculate metrics for each k
        for k in k_values:
            k_knn = knn_recs[:k]
            hits_knn = len(set(k_knn).intersection(relevant_test_movies))
            results[k]["knn_precision"].append(hits_knn / k)
            results[k]["knn_recall"].append(hits_knn / len(relevant_test_movies))

    # Aggregate metrics
    summary_rows = []
    for k in k_values:
        knn_prec = float(np.mean(results[k]["knn_precision"])) if results[k]["knn_precision"] else 0.0
        knn_rec = float(np.mean(results[k]["knn_recall"])) if results[k]["knn_recall"] else 0.0

        summary_rows.append({
            "Metric": f"Precision@{k}",
            "CineMatch KNN": f"{knn_prec * 100:.2f}%",
            "KNN Value": round(knn_prec * 100, 2)
        })
        summary_rows.append({
            "Metric": f"Recall@{k}",
            "CineMatch KNN": f"{knn_rec * 100:.2f}%",
            "KNN Value": round(knn_rec * 100, 2)
        })

    summary_df = pd.DataFrame(summary_rows)
    return summary_df, {
        "n_users_evaluated": len(results[k_values[0]]["knn_precision"]),
        "relevance_threshold": relevance_threshold,
        "k_values": list(k_values)
    }
