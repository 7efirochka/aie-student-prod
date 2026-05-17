from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
from contextlib import asynccontextmanager
from src.service.recommender import MovieRecommender
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

recommender = None

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "movies_1990_2026.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global recommender
    logger.info("Loading recommendation model")
    recommender = MovieRecommender(str(DATA_PATH), str(ARTIFACTS_DIR))
    logger.info("Model loaded successfully!")

    yield


app = FastAPI(
    lifespan=lifespan,
    title="Movie recommender API",
    description="Movie recommendations system based on your favorites films",
    version="1.0.0",
)


class RecommendationRequest(BaseModel):
    titles: str
    n: Optional[int] = 5
    min_year: Optional[int] = None
    adult: Optional[bool] = True


@app.get("/")
async def root():
    return {"message": "Welcome to the movie recommendation API"}


@app.post("/recommend")
async def reccommend(request: RecommendationRequest):
    try:
        recommendations = recommender.recommend(
            request.titles, request.n, request.min_year, request.adult
        )
        return {"recommendations": recommendations}

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "OK", "model_loaded": True}
