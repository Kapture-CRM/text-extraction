from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_TITLE: str = "Text Extraction Service"
    APP_DESCRIPTION: str = "Document processing and keyword search pipelines."
    APP_VERSION: str = "1.0.0"
    API_BASE_PATH: str = "api/v1"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
