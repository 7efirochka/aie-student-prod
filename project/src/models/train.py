import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "movies_1990_2026.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

MODEL_MATRIX_PATH = ARTIFACTS_DIR / "cosine_hybrid_matrix.npy"
ENCODERS_PATH = ARTIFACTS_DIR / "encoders.joblib"

WEIGHT_GENRES = 0.3
WEIGHT_DESC = 0.7
MAX_FEATURES_TFIDF = 5000


def str_to_list(genres):
    if pd.isna(genres) or genres == "":
        return []
    try:
        return ast.literal_eval(genres)
    except (ValueError, SyntaxError):
        logger.warning(f"Could not parse genres: {genres}")
        return []


def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    df["genres"] = df["genres"].apply(str_to_list)
    df["description"] = df["description"].fillna("").astype(str)
    return df


def train_hybrid_model(df):

    logger.info("Fitting MultiLabelBinarizer")
    mlb = MultiLabelBinarizer()
    genre_matrix = mlb.fit_transform(df["genres"])

    logger.info("Calculating Genre Similarity")
    cosine_genres = cosine_similarity(genre_matrix).astype(np.float32)

    logger.info("Fitting TfidfVectorizer")
    tfid = TfidfVectorizer(
        stop_words="english", max_features=MAX_FEATURES_TFIDF, ngram_range=(1, 2)
    )
    description_matrix = tfid.fit_transform(df["description"])

    logger.info("Calculating Description Similarity")
    cosine_description = cosine_similarity(description_matrix).astype(np.float32)

    logger.info("Combine similarities")
    hybrid_matrix = WEIGHT_GENRES * cosine_genres + WEIGHT_DESC * cosine_description
    np.save(MODEL_MATRIX_PATH, hybrid_matrix)

    logger.info(f"Matrix saved to {MODEL_MATRIX_PATH}")

    encoders = {
        "mlb": mlb,
        "tfidf": tfid,
        "movie_ids": df["id"].values,
        "titles": df["title"].values,
    }
    joblib.dump(encoders, ENCODERS_PATH)

    logger.info(f"Model saved to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    try:
        df = load_and_preprocess(DATA_PATH)
        train_hybrid_model(df)
    except Exception as e:
        logger.error(f"Load or training failed: {e}")
        raise
