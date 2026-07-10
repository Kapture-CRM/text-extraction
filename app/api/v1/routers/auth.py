from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.core.auth import create_access_token, verify_password
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger("auth")

router = APIRouter(prefix="/auth", tags=["Auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/token", response_model=TokenResponse, summary="Login to get a JWT access token")
def login(body: LoginRequest):
    if body.username != settings.AUTH_USERNAME or not verify_password(
        body.password, _hashed_seed_password()
    ):
        logger.warning(f"Failed login attempt for username={body.username!r}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    token = create_access_token(subject=body.username)
    logger.info(f"Successful login for username={body.username!r}")
    return TokenResponse(access_token=token)


_cached_hash: str | None = None


def _hashed_seed_password() -> str:
    global _cached_hash
    if _cached_hash is None:
        from app.core.auth import hash_password
        _cached_hash = hash_password(settings.AUTH_PASSWORD)
    return _cached_hash
