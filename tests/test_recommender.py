"""
test_recommender.py
-------------------
Unit tests for CineMatch Recommendation System.
Covers data preprocessing, matrix generation, KNN recommendations,
popularity baseline, and error edge cases.
"""

import unittest
import numpy as np
import pandas as pd
from src.data_processing import (
    load_raw_data,
    clean_and_merge_data,
    compute_sparsity,
    filter_sparse_interactions,
    create_pivot_matrix
)
from src.recommender import CineMatchKNN
from src.eda import calculate_descriptive_statistics


class TestCineMatch(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Loads data once for the test suite."""
        cls.raw_movies, cls.raw_ratings = load_raw_data("movies.csv", "ratings.csv")
        cls.merged_df, cls.metadata_df = clean_and_merge_data(cls.raw_movies, cls.raw_ratings)
        cls.filtered_df = filter_sparse_interactions(cls.merged_df, min_movie_ratings=10, min_user_ratings=5)
        cls.pivot_df, cls.sparse_matrix = create_pivot_matrix(cls.filtered_df)

        cls.knn = CineMatchKNN(metric="cosine", algorithm="brute")
        cls.knn.fit(cls.pivot_df, cls.sparse_matrix, cls.metadata_df)

    def test_schema_and_cleaning(self):
        """Verify data schema and that null values are removed."""
        self.assertIn("movieId", self.merged_df.columns)
        self.assertIn("title", self.merged_df.columns)
        self.assertIn("rating", self.merged_df.columns)
        self.assertFalse(self.merged_df["rating"].isnull().any())
        self.assertTrue((self.merged_df["rating"] >= 0.5).all())
        self.assertTrue((self.merged_df["rating"] <= 5.0).all())

    def test_sparsity_formula(self):
        """Verify the sparsity percentage formula."""
        # 10 users, 10 items, 20 ratings -> 20 / 100 = 20% density -> 80% sparsity
        sparsity = compute_sparsity(10, 10, 20)
        self.assertEqual(sparsity, 80.0)

    def test_pivot_matrix_dimensions(self):
        """Verify that pivot table has no NaNs and non-zero shape."""
        self.assertGreater(self.pivot_df.shape[0], 0)
        self.assertGreater(self.pivot_df.shape[1], 0)
        self.assertFalse(self.pivot_df.isna().any().any())

    def test_knn_recommendations_valid_movie(self):
        """Verify that KNN returns exactly K recommendations with valid similarity scores."""
        res = self.knn.recommend("Toy Story", top_k=5)
        self.assertEqual(res["status"], "success")
        self.assertEqual(len(res["recommendations"]), 5)

        for rec in res["recommendations"]:
            self.assertIn("title", rec)
            self.assertIn("similarity_score", rec)
            self.assertTrue(0.0 <= rec["similarity_score"] <= 1.0)
            self.assertTrue(0.0 <= rec["similarity_percentage"] <= 100.0)

    def test_knn_genre_filter(self):
        """Verify that genre filter restricts recommendation genres."""
        res = self.knn.recommend("Toy Story", top_k=5, genre_filter="Animation")
        self.assertEqual(res["status"], "success")
        for rec in res["recommendations"]:
            genres = [g.strip().lower() for g in rec["genres"].split("|")]
            self.assertIn("animation", genres)

    def test_edge_case_unknown_movie(self):
        """Verify handling of movies not present in the dataset."""
        res = self.knn.recommend("This Movie Absolutely Does Not Exist 9999", top_k=5)
        self.assertEqual(res["status"], "not_found")
        self.assertEqual(len(res["recommendations"]), 0)

    def test_edge_case_empty_input(self):
        """Verify handling of empty or blank string."""
        res = self.knn.recommend("   ", top_k=5)
        self.assertEqual(res["status"], "not_found")

    def test_descriptive_statistics(self):
        """Verify statistical metrics are computed without errors."""
        stats = calculate_descriptive_statistics(self.raw_ratings)
        self.assertIn("mean", stats)
        self.assertIn("median", stats)
        self.assertIn("variance", stats)
        self.assertIn("skewness", stats)
        self.assertAlmostEqual(stats["mean"], 3.50, places=1)


if __name__ == "__main__":
    unittest.main()
