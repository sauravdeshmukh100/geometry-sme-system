#!/usr/bin/env python
"""
Authentication Routes
Handles user registration, login, token refresh, and profile management
"""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer
from typing import List
import logging
from datetime import timedelta

from ..models.auth_models import (
    User, UserCreate, UserLogin, Token, UserUpdate,
    PasswordChange, RefreshTokenRequest, UserRole
)
from ..services.auth_service import AuthService
from ..database.user_repository import get_user_repository
from .dependencies import (
    get_auth_service, get_current_user, get_current_active_user,
    require_admin, require_teacher, verify_admin_access
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
    responses={401: {"description": "Unauthorized"}}
)

security = HTTPBearer()


# ========== Public Endpoints (No Authentication Required) ==========

@router.post("/register", response_model=User, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_create: UserCreate,
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Register a new user.
    
    - **email**: Valid email address
    - **username**: Unique username (3-50 characters)
    - **password**: Strong password (min 8 chars, must include uppercase, lowercase, and digit)
    - **role**: User role (default: student)
    """
    try:
        user_repo = get_user_repository()
        
        # Check if username or email already exists
        if user_repo.user_exists(username=user_create.username):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
        
        if user_repo.user_exists(email=user_create.email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create user with hashed password
        user_in_db = auth_service.create_user(user_create)
        
        # Save to repository
        user_repo.create_user(user_in_db)
        
        # Return user without password
        user = User(**user_in_db.dict(exclude={'hashed_password'}))
        
        logger.info(f"New user registered: {user.username} ({user.role.value})")
        
        return user
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Registration error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/login", response_model=Token)
async def login(
    user_login: UserLogin,
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Login with username/email and password.
    
    Returns JWT access token and refresh token.
    
    - **username**: Username or email address
    - **password**: User password
    """
    try:
        user_repo = get_user_repository()
        
        # Find user by username or email
        user_in_db = user_repo.get_user_by_username(user_login.username)
        if not user_in_db:
            user_in_db = user_repo.get_user_by_email(user_login.username)
        
        if not user_in_db:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Authenticate user
        user = auth_service.authenticate_user(user_in_db, user_login.password)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Update last login
        user_repo.update_last_login(user.user_id)
        
        # Create tokens
        access_token = auth_service.create_access_token(user)
        refresh_token = auth_service.create_refresh_token(user)
        
        logger.info(f"User logged in: {user.username}")
        
        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=auth_service.access_token_expire_minutes * 60
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post("/refresh", response_model=Token)
async def refresh_token(
    refresh_request: RefreshTokenRequest,
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Refresh access token using refresh token.
    
    - **refresh_token**: Valid refresh token
    """
    try:
        user_repo = get_user_repository()
        
        # Decode refresh token
        token_data = auth_service.decode_token(refresh_request.refresh_token)
        
        # Get user
        user_in_db = user_repo.get_user_by_id(token_data.user_id)
        if not user_in_db or not user_in_db.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        user = User(**user_in_db.dict(exclude={'hashed_password'}))
        
        # Generate new access token
        new_access_token = auth_service.refresh_access_token(
            refresh_request.refresh_token,
            user
        )
        
        logger.info(f"Token refreshed for user: {user.username}")
        
        return Token(
            access_token=new_access_token,
            refresh_token=refresh_request.refresh_token,  # Same refresh token
            token_type="bearer",
            expires_in=auth_service.access_token_expire_minutes * 60
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )


# ========== Protected Endpoints (Authentication Required) ==========

@router.get("/me", response_model=User)
async def get_current_user_info(
    current_user: User = Depends(get_current_active_user)
):
    """
    Get current user profile information.
    
    Requires valid JWT token in Authorization header.
    """
    return current_user


@router.put("/me", response_model=User)
async def update_current_user(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_active_user),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Update current user's profile.
    
    Can update: full_name, grade_level, school, preferences
    """
    try:
        user_repo = get_user_repository()
        
        # Prepare update data (exclude None values)
        update_data = user_update.dict(exclude_unset=True, exclude_none=True)
        
        # Update user
        updated_user = user_repo.update_user(current_user.user_id, update_data)
        
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        logger.info(f"User profile updated: {current_user.username}")
        
        return User(**updated_user.dict(exclude={'hashed_password'}))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Profile update error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Profile update failed"
        )


@router.post("/me/change-password")
async def change_password(
    password_change: PasswordChange,
    current_user: User = Depends(get_current_active_user),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Change current user's password.
    
    - **current_password**: Current password for verification
    - **new_password**: New password (must meet strength requirements)
    """
    try:
        user_repo = get_user_repository()
        
        # Get user with password
        user_in_db = user_repo.get_user_by_id(current_user.user_id)
        
        # Verify current password
        if not auth_service.verify_password(
            password_change.current_password,
            user_in_db.hashed_password
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect current password"
            )
        
        # Hash new password
        new_hashed_password = auth_service.hash_password(password_change.new_password)
        
        # Update password
        user_repo.update_password(current_user.user_id, new_hashed_password)
        
        logger.info(f"Password changed for user: {current_user.username}")
        
        return {"message": "Password changed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password change error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password change failed"
        )


@router.delete("/me")
async def delete_current_user(
    current_user: User = Depends(get_current_active_user)
):
    """
    Delete current user account.
    
    This action is irreversible.
    """
    try:
        user_repo = get_user_repository()
        
        # Delete user
        success = user_repo.delete_user(current_user.user_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        logger.info(f"User account deleted: {current_user.username}")
        
        return {"message": "Account deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Account deletion error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Account deletion failed"
        )


# ========== Admin Endpoints ==========

@router.get("/users", response_model=List[User])
async def list_users(
    role: UserRole = None,
    is_active: bool = None,
    limit: int = 100,
    offset: int = 0,
    current_user: User = Depends(verify_admin_access)
):
    """
    List all users (Admin only).
    
    - **role**: Filter by role (optional)
    - **is_active**: Filter by active status (optional)
    - **limit**: Maximum number of users to return
    - **offset**: Number of users to skip
    """
    try:
        user_repo = get_user_repository()
        users = user_repo.list_users(
            role=role,
            is_active=is_active,
            limit=limit,
            offset=offset
        )
        
        return users
        
    except Exception as e:
        logger.error(f"List users error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list users"
        )


@router.get("/users/{user_id}", response_model=User)
async def get_user_by_id(
    user_id: str,
    current_user: User = Depends(verify_admin_access)
):
    """
    Get user by ID (Admin only).
    
    - **user_id**: User ID
    """
    try:
        user_repo = get_user_repository()
        user_in_db = user_repo.get_user_by_id(user_id)
        
        if not user_in_db:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return User(**user_in_db.dict(exclude={'hashed_password'}))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get user error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user"
        )


@router.put("/users/{user_id}", response_model=User)
async def update_user_by_admin(
    user_id: str,
    user_update: UserUpdate,
    current_user: User = Depends(verify_admin_access)
):
    """
    Update user by ID (Admin only).
    
    - **user_id**: User ID to update
    """
    try:
        user_repo = get_user_repository()
        
        # Prepare update data
        update_data = user_update.dict(exclude_unset=True, exclude_none=True)
        
        # Update user
        updated_user = user_repo.update_user(user_id, update_data)
        
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        logger.info(f"User updated by admin: {user_id}")
        
        return User(**updated_user.dict(exclude={'hashed_password'}))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Admin update user error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user"
        )


@router.delete("/users/{user_id}")
async def delete_user_by_admin(
    user_id: str,
    current_user: User = Depends(verify_admin_access)
):
    """
    Delete user by ID (Admin only).
    
    - **user_id**: User ID to delete
    """
    try:
        user_repo = get_user_repository()
        
        # Prevent admin from deleting themselves
        if user_id == current_user.user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete your own account"
            )
        
        # Delete user
        success = user_repo.delete_user(user_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        logger.info(f"User deleted by admin: {user_id}")
        
        return {"message": "User deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Admin delete user error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete user"
        )


@router.get("/stats")
async def get_user_statistics(
    current_user: User = Depends(verify_admin_access)
):
    """
    Get user statistics (Admin only).
    
    Returns counts by role, active status, etc.
    """
    try:
        user_repo = get_user_repository()
        stats = user_repo.get_user_statistics()
        
        return stats
        
    except Exception as e:
        logger.error(f"Get user stats error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get statistics"
        )