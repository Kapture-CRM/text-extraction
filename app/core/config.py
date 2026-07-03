from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_TITLE: str = "Text Extraction Service"
    APP_DESCRIPTION: str = "Document processing and keyword search pipelines."
    APP_VERSION: str = "1.0.0"
    API_BASE_PATH: str = "/api/v1"

    # JWT
    JWT_SECRET: str
    JWT_ALGORITHM: str
    JWT_EXPIRE_MINUTES: int = 525600  # 1 year

    # Seed user
    AUTH_USERNAME: str
    AUTH_PASSWORD: str

    # Gemini / Vertex AI
    GOOGLE_APPLICATION_CREDENTIALS_JSON: str
    GEMINI_MODEL: str = "gemini-3.5-flash"
    GCP_PROJECT_ID: str = "sublime-amp-160011"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
