import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.service.recommender import MovieRecommender
import time
import logging
from pathlib import Path
import ast
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "movies_1990_2026.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

MODEL_MATRIX_PATH = ARTIFACTS_DIR / "cosine_description_matrix.npy"

df = pd.read_csv(DATA_PATH)
df["title_clean"] = df["title"].str.lower().replace(",", "").str.strip()

dict_df = dict(zip(df["title_clean"], df.index))


# Модель, которая выдаёт рандомные фильмы
def random_films(query_titles, n=5):

    candidates = df.copy()
    result = candidates.sample(n=min(n, len(candidates)))
    return [
        {
            "title": row["title"],
            "year": int(row["year"]) if pd.notna(row["year"]) else None,
            "rating": round(float(row["rating"]), 1),
            "adult": row["adult"],
            "genres": row["genres"],
        }
        for _, row in result.iterrows()
    ]


# Модель, которая выдаёт самые популярные фильмы
def recommend_popular(query_titles, n=5):

    candidates = df.copy()
    result = candidates.sort_values(by="rating", ascending=False).head(n)

    return [
        {
            "title": row["title"],
            "year": int(row["year"]) if pd.notna(row["year"]) else None,
            "rating": round(float(row["rating"]), 1),
            "adult": row["adult"],
            "genres": row["genres"],
        }
        for _, row in result.iterrows()
    ]


def load_data():
    tfid = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
    df["description"] = df["description"].fillna(" ")
    description_matrix = tfid.fit_transform(df["description"])
    similarity_matrix = cosine_similarity(description_matrix).astype(np.float32)

    np.save(MODEL_MATRIX_PATH, similarity_matrix)
    logger.info("Description matrix saved")


# Модель, у которой контентный поиск построен только по описанию фильмов
def recommend_description(query_titles, n=5):

    similarity_matrix = np.load(MODEL_MATRIX_PATH)

    indices = []

    for title in query_titles.split(","):
        clean_title = title.lower().strip()
        if clean_title in dict_df:
            indices.append(dict_df[clean_title])

    if not indices:
        return "Error, don't have your films in database("

    seed_vectors = similarity_matrix[indices]

    avg_sim_scores = np.mean(seed_vectors, axis=0)

    for idx in indices:
        avg_sim_scores[idx] = -1

    top_indices = np.argsort(avg_sim_scores)[-n - 70 :]
    top_indices = top_indices[::-1]

    candidates = df.iloc[top_indices].copy()
    candidates["sim_score"] = avg_sim_scores[top_indices]

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
                "adult": row["adult"],
                "similarity_score": round(float(row["sim_score"]), 4),
            }
        )
    return recommendations


load_data()

#   Сравнение моделей


def comparing_models(model):

    # Тест 1. Фильм с жанром боевик. Условие - минимум в 3 из 5 фильмов в жанрах должен быть Action.

    recommendations_test_1 = model("John Wick")
    cnt_genres_test_1 = 0

    for i in range(len(recommendations_test_1)):
        genres_1 = ast.literal_eval(recommendations_test_1[i]["genres"])
        if "Action" in genres_1:
            cnt_genres_test_1 += 1

    # Тест 2. Семейный фильм. Условия - минимум в 3 из 5 фильмах в жанрах должен быть Family
    # & минимум в 4 из 5 фильмах adults = False
    # & жанрах не должно быть жанров Horror, Crime, Thriller (максимум 1 фильм)

    recommendations_test_2 = model("Toy Story 4")
    cnt_genres_test_2 = 0
    cnt_adult_test_2 = 0
    cnt_wrong_genres_test_2 = 0

    wrong_genres_test_2 = ["Horror", "Crime", "Thriller"]

    for i in range(len(recommendations_test_2)):
        genres_2 = ast.literal_eval(recommendations_test_2[i]["genres"])
        if "Family" in genres_2:
            cnt_genres_test_2 += 1
        if recommendations_test_2[0]["adult"]:
            cnt_adult_test_2 += 1
        if any(gen in genres_2 for gen in wrong_genres_test_2):
            cnt_wrong_genres_test_2 += 1

    # Тест 3. Хоррор фильм. Условия - минимум в 3 фильмах из 5 в жанрах должен быть Horror
    # & минимум в 4 из 5 фильмах adults = True
    # & в жанрах не должно быть Family (максимум 1 фильм)

    recommendations_test_3 = model("The Purge")
    cnt_genres_test_3 = 0
    cnt_adult_test_3 = 0
    cnt_wrong_genres_test_3 = 0

    wrong_genres_test_3 = ["Family"]

    for i in range(len(recommendations_test_3)):
        genres_3 = ast.literal_eval(recommendations_test_3[i]["genres"])
        if "Horror" in genres_3:
            cnt_genres_test_3 += 1
        if recommendations_test_3[i]["adult"]:
            cnt_adult_test_3 += 1
        if any(gen in genres_3 for gen in wrong_genres_test_3):
            cnt_wrong_genres_test_3 += 1

    # Тест 4. Научная фантастика. Условие: минимум в 3 из 5 фильмов должен быть жанр "Science Fiction"
    recommendations_test_4 = model("Interstellar")
    cnt_genres_test_4 = 0
    for i in range(len(recommendations_test_4)):
        genres = ast.literal_eval(recommendations_test_4[i]["genres"])
        if "Science Fiction" in genres:
            cnt_genres_test_4 += 1

    # Тест 5. Высокий рейтинг. Условие: средний рейтинг рекомендаций >= 5.0
    recommendations_test_5 = model("The Shawshank Redemption")
    ratings_test_5 = [rec["rating"] for rec in recommendations_test_5[:5]]
    avg_rating_test_5 = np.mean(ratings_test_5) if ratings_test_5 else 0

    return {
        "test_1": {
            "valid_genres": cnt_genres_test_1,
            "result": "success ✅️" if cnt_genres_test_1 >= 3 else "fail ❌",
        },
        "test_2": {
            "valid_genres": cnt_genres_test_2,
            "adult_films": cnt_adult_test_2,
            "wrong_genres": cnt_wrong_genres_test_2,
            "result": (
                "success ✅️"
                if cnt_genres_test_2 >= 3
                and cnt_adult_test_2 <= 1
                and cnt_wrong_genres_test_2 <= 1
                else "fail ❌"
            ),
        },
        "test_3": {
            "valid_genres": cnt_genres_test_3,
            "adult_films": cnt_adult_test_3,
            "wrong_genres": cnt_wrong_genres_test_3,
            "result": (
                "success ✅️"
                if cnt_genres_test_3 >= 3
                and cnt_adult_test_3 >= 4
                and cnt_wrong_genres_test_3 <= 1
                else "fail ❌"
            ),
        },
        "test_4": {
            "valid_genres": cnt_genres_test_4,
            "result": "success ✅️" if cnt_genres_test_4 >= 3 else "fail ❌",
        },
        "test_5": {
            "avg_rating": avg_rating_test_5,
            "result": "success ✅️" if avg_rating_test_5 >= 5.5 else "fail ❌",
        },
    }


hybride_model_1 = MovieRecommender(DATA_PATH, ARTIFACTS_DIR)
data = {
    "hybride_model": comparing_models(hybride_model_1.recommend),
    "description_model": comparing_models(recommend_description),
    "popular": comparing_models(recommend_popular),
    "random": comparing_models(random_films),
}

json_path = ARTIFACTS_DIR / "compare_models.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
