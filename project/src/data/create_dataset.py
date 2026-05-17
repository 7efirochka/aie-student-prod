import requests
import pandas as pd
import time
from dotenv import load_dotenv
import os
from pathlib import Path

ROOT_DIR_PROJECT = Path(__file__).parent.parent.parent
DATA_PATH = ROOT_DIR_PROJECT / "data"

load_dotenv()

API_KEY = os.getenv("TMDB_API_KEY")

BASE_URL = "https://api.themoviedb.org/3"


def get_genre_map():
    url = f"https://api.themoviedb.org/3/genre/movie/list"
    params = {"api_key": API_KEY, "language": "en-US"}
    response = requests.get(url, params=params)

    if response.status_code == 200:
        genres = response.json()["genres"]
        return {g["id"]: g["name"] for g in genres}


genre_map_api = get_genre_map()


def get_movies_by_year(year, max_pages=10):
    all_year_movies = []
    for page in range(1, max_pages + 1):
        url = f"{BASE_URL}/discover/movie"
        params = {
            "api_key": API_KEY,
            "primary_release_year": year,
            "language": "en-US",
            "page": page,
            "sort_by": "popularity.desc",
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            results = data["results"]

        if not results:
            break

        for i in results:
            i["release_date"] = i["release_date"][:4]
            i["genre_ids"] = list(map(lambda x: genre_map_api[x], i["genre_ids"]))
            if (
                "Crime" in i["genre_ids"]
                or "Horror" in i["genre_ids"]
                or "Thriller" in i["genre_ids"]
                or "War" in i["genre_ids"]
            ):
                i["adult"] = True
        all_year_movies.extend(results)

    return all_year_movies


all_movies = []


for year in range(1990, 2027):
    movies = get_movies_by_year(year, max_pages=15)
    if movies:
        all_movies.extend(movies)
    time.sleep(0.2)


df = pd.DataFrame(all_movies)


cols_to_keep = [
    "id",
    "title",
    "release_date",
    "overview",
    "genre_ids",
    "vote_average",
    "vote_count",
    "adult",
]
df = df[cols_to_keep]


df = df.rename(
    columns={
        "release_date": "year",
        "genre_ids": "genres",
        "overview": "description",
        "vote_average": "rating",
        "vote_count": "num_votes",
    },
    inplace=False,
)

df_clean = df.dropna(subset=["description", "genres", "year", "rating"])

df_clean.to_csv(DATA_PATH / "movies_1990_2026.csv", index=False)
