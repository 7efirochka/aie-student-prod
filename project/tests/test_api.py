import pytest
from fastapi.testclient import TestClient
from src.service.api import app
from pathlib import Path
import logging
import ast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "movies_1990_2026.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"


@pytest.fixture(scope="function")
def client():
    with TestClient(app) as c:
        yield c


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200


def test_recommend_valid_input(client):
    response = client.post("/recommend", json={"titles": "The Matrix", "n": 15})
    data = response.json()

    movie = data["recommendations"][0]
    assert "title" in movie
    assert "year" in movie
    assert "genres" in movie
    assert "rating" in movie

    assert response.status_code == 200
    assert len(data["recommendations"]) == 15


def test_invalid_input(client):
    titles = "Unreal film which doesn't exist"
    response = client.post("/recommend", json={"titles": titles, "n": 8})
    data = response.json()

    logging.info(data)
    assert data["recommendations"] == {
        "error": "Movies not found in database: Unreal film which doesn't exist"
    }


def test_empty_input(client):
    response = client.post("/recommend", json={"titles": "", "n": 3})
    data = response.json()

    assert data["recommendations"] == {
        "error": "Please write at least one movie title."
    }


def test_exclude_match(client):
    response = client.post("/recommend", json={"titles": "Life of Pi", "n": 3})
    result = response.json()
    titles = [i["title"] for i in result["recommendations"]]
    assert "Life of Pi" not in titles


def test_minimum_output_n(client):
    result_1 = client.post("/recommend", json={"titles": "Pulp Fiction"}).json()
    result_2 = client.post("/recommend", json={"titles": "Fight Club", "n": 10}).json()

    assert len(result_1["recommendations"]) == 5
    assert len(result_2["recommendations"]) == 10


def test_rating_sort(client):
    result = client.post(
        "/recommend", json={"titles": "Harry Potter and the Prisoner of Azkaban"}
    ).json()
    rating = [i["rating"] for i in result["recommendations"]]
    assert rating == sorted(rating, reverse=True)


def test_case_insensitive(client):
    result_1 = client.post("/recommend", json={"titles": "the matrix"}).json()
    result_2 = client.post("/recommend", json={"titles": "The Matrix"}).json()

    titles_1 = [i["title"] for i in result_1["recommendations"]]
    titles_2 = [i["title"] for i in result_2["recommendations"]]

    assert titles_1 == titles_2


def test_special_characters(client):
    result = client.post("/recommend", json={"titles": "Spider-Man"}).json()
    assert len(result) >= 1


def test_normal_genres(client):

    result_1 = client.post("/recommend", json={"titles": "The Matrix", "n": 10}).json()
    genres_1 = []

    for movie in result_1["recommendations"]:
        genres_raw = movie["genres"]
        genres_list = ast.literal_eval(genres_raw)

        for i in genres_list:
            genres_1.append(i)

    assert "Science Fiction" in genres_1
    assert "Action" in genres_1

    assert "Children" not in genres_1
    assert "Family" not in genres_1

    result_2 = client.post("/recommend", json={"titles": "Spider-Man", "n": 10}).json()
    genres_2 = []

    for movie in result_2["recommendations"]:
        genres_raw = movie["genres"]
        genres_list = ast.literal_eval(genres_raw)

        for i in genres_list:
            genres_2.append(i)

    assert "Adventure" in genres_2
    assert "Action" in genres_2
    assert "Science Fiction" in genres_2

    assert "Documentary" not in genres_2
    assert "News" not in genres_2

    result_3 = client.post(
        "/recommend", json={"titles": "The Addams Family", "n": 10}
    ).json()
    genres_3 = []

    for movie in result_3["recommendations"]:
        genres_raw = movie["genres"]
        genres_list = ast.literal_eval(genres_raw)

        for i in genres_list:
            genres_3.append(i)

    assert "Fantasy" in genres_3
    assert "Comedy" in genres_3
    assert "Family" in genres_3

    assert "Documentary" not in genres_3
    assert "War" not in genres_3
    assert "Adult" not in genres_3

    result_3 = client.post(
        "/recommend", json={"titles": "Marty Supreme", "n": 10}
    ).json()
    genres_4 = []

    for movie in result_3["recommendations"]:
        genres_raw = movie["genres"]
        genres_list = ast.literal_eval(genres_raw)

        for i in genres_list:
            genres_4.append(i)

    assert "Thriller" in genres_4
    assert "Drama" in genres_4

    assert "Science Fiction" not in genres_4
    assert "Animation" not in genres_4
