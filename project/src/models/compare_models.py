import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("../data/movies_1990_2026.csv")
df["title_clean"] = df["title"].str.lower().replace(",", "").str.strip()

dict_df = dict(zip(df["title_clean"], df.index))


def random_films(query_titles, n=10):

    candidates = df.copy()
    result = candidates.sample(n=min(n, len(candidates)))
    return [
        {
            "title": row["title"],
            "year": int(row["year"]) if pd.notna(row["year"]) else None,
            "rating": round(float(row["rating"]), 1),
            "genres": row["genres"],
        }
        for _, row in result.iterrows()
    ]


def recommend_popular(query_titles, n=10):
    candidates = df.copy()

    result = candidates.sort_values(by="rating", ascending=False).head(n)

    return [
        {
            "title": row["title"],
            "year": int(row["year"]) if pd.notna(row["year"]) else None,
            "rating": round(float(row["rating"]), 1),
            "genres": row["genres"],
            "method": "popular",
        }
        for _, row in result.iterrows()
    ]


def recommend_description(query_titles, n=10):

    tfid = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
    df["description"] = df["description"].fillna(" ")
    description_matrix = tfid.fit_transform(df["description"])
    similarity_matrix = cosine_similarity(description_matrix).astype(np.float32)

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
                "similarity_score": round(float(row["sim_score"]), 4),
            }
        )
    return recommendations
