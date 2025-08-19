# config.py - configuration settings

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()  # loads .env if present

@dataclass
class Config:
    # api keys (read from env)
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # data collection settings
    TARGET_ARTICLES: int = 10000
    BATCH_SIZE: int = 100
    MAX_WORKERS: int = 8
    CHUNK_SIZE: int = 1000

    # model settings
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    RANKING_MODEL: str = "xgboost"

    # paths
    DATA_DIR: str = "data"
    RAW_DATA_PATH: str = "data/raw"
    PROCESSED_DATA_PATH: str = "data/processed"
    INDICES_PATH: str = "data/indices"

    # search settings
    TOP_K_RESULTS: int = 20

    # user simulation settings
    NUM_USERS: int = 1000

    def __post_init__(self):
        os.makedirs(self.RAW_DATA_PATH, exist_ok=True)
        os.makedirs(self.PROCESSED_DATA_PATH, exist_ok=True)
        os.makedirs(self.INDICES_PATH, exist_ok=True)
