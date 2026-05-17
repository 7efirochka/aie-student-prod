import pandas as pd
import numpy as np
import ast
import joblib
from pathlib import Path
import logging
from typing import Optional, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MovieRecommender:
    def __init__(self, data_path, artifacts_dir):
        self.data_path = Path(data_path)
        self.artifacts_dir = Path(artifacts_dir)

        self.matrix_path = self.artifacts_dir / "cosine_hybrid_matrix.npy"
        self.encoders_path = self.artifacts_dir / "encoders.joblib"

        self.df = None
        self.similarity_matrix = None
        self.title_to_index = {}

        self._load_model_and_data()

    def _load_model_and_data(self):
        logger.info("Loading data and model artifacts")
        self.df = pd.read_csv(self.data_path, low_memory=False)

        self.df["title_clean"] = (
            self.df["title"].str.lower().replace(",", "").str.strip()
        )
        self.title_to_index = dict(zip(self.df["title_clean"], self.df.index))

        logger.info("Load similarity matrix")

        self.similarity_matrix = np.load(self.matrix_path)

        logger.info("Model load")

    def _find_movie_indices(self, titles):
        indices = []
        not_found = []

        for title in titles:
            clean_title = title.lower().strip()
            if clean_title in self.title_to_index:
                indices.append(self.title_to_index[clean_title])
            else:
                not_found.append(title)

        return indices, not_found

    def recommend(self, query_titles, n=5, min_year=None):
        raw_titles = [i.strip() for i in query_titles.split(",") if i.strip()]
        if not raw_titles:
            return {"error": "Please write at least one movie title."}

        seed_indices, not_found = self._find_movie_indices(raw_titles)

        if not seed_indices:
            return {"error": f"Movies not found in database: {', '.join(not_found)}"}

        if not_found:
            logger.warning(f"Some movies were not found: {not_found}")

        seed_vectors = self.similarity_matrix[seed_indices]
        avg_sim_scores = np.mean(seed_vectors, axis=0)

        for idx in seed_indices:
            avg_sim_scores[idx] = -1

        top_indices = np.argsort(avg_sim_scores)[-n - 70 :]
        top_indices = top_indices[::-1]

        candidates = self.df.iloc[top_indices].copy()
        candidates["sim_score"] = avg_sim_scores[top_indices]

        if min_year is not None:
            candidates = candidates[candidates["year"] >= min_year]

        candidates = candidates.sort_values(
            by=["sim_score", "rating"], ascending=[False, False]
        )

        result_df = candidates.head(n)

        recommendations = []

        for _, row in result_df.iterrows():
            recommendations.append(
                {
                    "title": row["title"],
                    "year": int(row["year"]) if pd.notna(row["year"]) else None,
                    "rating": round(float(row["rating"]), 1),
                    "genres": row["genres"],
                    "similarity_score": round(float(row["sim_score"]), 4),
                }
            )

        return recommendations


if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_PATH = BASE_DIR / "data" / "movies_1990_2026.csv"
    ARTIFACTS_DIR = BASE_DIR / "artifacts"

    recommender = MovieRecommender(DATA_PATH, ARTIFACTS_DIR)
    query = "The Matrix, Inception"
    print(f"\nRecommendations for: '{query}'")
    recs = recommender.recommend(query, n=5, min_year=2010)

    for i, rec in enumerate(recs, 1):
        print(
            f"{i}. {rec['title']} ({rec['year']}) | Rating: {rec['rating']} | Score: {rec['similarity_score']}"
        )
