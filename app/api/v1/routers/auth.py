from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from app.core.auth import create_access_token, verify_password
from app.core.config import settings

router = APIRouter(prefix="/auth", tags=["Auth"])


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


@router.post("/token", response_model=TokenResponse, summary="Login to get a JWT access token")
def login(form: OAuth2PasswordRequestForm = Depends()):
    if form.username != settings.AUTH_USERNAME or not verify_password(
        form.password, _hashed_seed_password()
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(subject=form.username)
    return TokenResponse(access_token=token)


# Cache the bcrypt hash of the seed password so it's only computed once.
_cached_hash: str | None = None


def _hashed_seed_password() -> str:
    global _cached_hash
    if _cached_hash is None:
        from app.core.auth import hash_password
        _cached_hash = hash_password(settings.AUTH_PASSWORD)
    return _cached_hash
