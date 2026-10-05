import os
from pydantic import BaseModel


class Settings(BaseModel):
    APP_NAME: str = "Sentiment Analysis API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = (
        "Production-grade Sentiment Analysis API classifying text into positive, "
        "negative, and neutral sentiments with confidence and per-class probability scores."
    )
    MODEL_NAME: str = os.getenv(
        "MODEL_NAME", "cardiffnlp/twitter-roberta-base-sentiment-latest"
    )
    MAX_TEXT_LENGTH: int = 2000
    MAX_BATCH_SIZE: int = 10
    MAX_TOKEN_LENGTH: int = 512


settings = Settings()
