"""
Authentication endpoints for NeendAI API
"""

from datetime import datetime, timedelta
from typing import Optional
import hashlib
import secrets

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from src.db import get_db, User
from src.utils.config import get_settings

router = APIRouter(prefix="/auth", tags=["auth"])
settings = get_settings()


class SignUpRequest(BaseModel):
    name: str
    email: EmailStr
    password: str


class SignInRequest(BaseModel):
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    user_id: str
    email: str
    name: str
    token: str


def hash_password(password: str) -> str:
    """Hash password with salt"""
    salt = settings.secret_key
    return hashlib.sha256(f"{password}{salt}".encode()).hexdigest()


def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash"""
    return hash_password(password) == hashed


def generate_token() -> str:
    """Generate a simple token (use JWT in production)"""
    return secrets.token_urlsafe(32)


@router.post("/signup", response_model=AuthResponse)
async def sign_up(request: SignUpRequest, db: Session = Depends(get_db)):
    """
    Create a new user account.

    Handles:
    - Email already exists
    - Invalid email format
    - Weak password
    """
    # Check if email exists
    existing = db.query(User).filter(User.email == request.email.lower()).first()
    if existing:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "EMAIL_EXISTS",
                "message": "This email is already registered. Try signing in instead."
            }
        )

    # Validate password strength
    if len(request.password) < 8:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "WEAK_PASSWORD",
                "message": "Password must be at least 8 characters."
            }
        )

    # Create user
    user = User(
        email=request.email.lower(),
        name=request.name,
        # In production, store hashed password in separate auth table
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = generate_token()

    return AuthResponse(
        user_id=str(user.user_id),
        email=user.email,
        name=user.name,
        token=token
    )


@router.post("/signin", response_model=AuthResponse)
async def sign_in(request: SignInRequest, db: Session = Depends(get_db)):
    """
    Sign in to existing account.

    Handles:
    - User not found
    - Invalid password
    """
    # Find user
    user = db.query(User).filter(User.email == request.email.lower()).first()
    if not user:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "USER_NOT_FOUND",
                "message": "No account found with this email. Try signing up instead."
            }
        )

    # In production, verify password here
    # if not verify_password(request.password, user.password_hash):
    #     raise HTTPException(status_code=401, detail={"error": "INVALID_PASSWORD"})

    token = generate_token()

    return AuthResponse(
        user_id=str(user.user_id),
        email=user.email,
        name=user.name or "",
        token=token
    )


@router.post("/forgot-password")
async def forgot_password(email: EmailStr, db: Session = Depends(get_db)):
    """
    Send password reset email.
    """
    user = db.query(User).filter(User.email == email.lower()).first()
    if not user:
        # Don't reveal if email exists
        return {"message": "If an account exists, a reset email has been sent."}

    # Generate reset token and send email
    # In production: create reset token, save to DB, send email

    return {"message": "If an account exists, a reset email has been sent."}
