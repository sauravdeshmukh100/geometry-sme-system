#!/usr/bin/env python
"""
FastAPI Dependencies for Authentication and Authorization
Provides dependency injection for protected routes
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import logging
import jwt

from ..models.auth_models import TokenData, User, UserRole, Permission
from ..services.auth_service import AuthService
from ..database.user_repository import get_user_repository
from ..config.settings import settings

logger = logging.getLogger(__name__)

# HTTP Bearer token scheme
security = HTTPBearer()

# Initialize auth service
auth_service = AuthService(
    secret_key=getattr(settings, 'JWT_SECRET_KEY', AuthService.generate_secret_key()),
    algorithm=getattr(settings, 'JWT_ALGORITHM', 'HS256'),
    access_token_expire_minutes=getattr(settings, 'ACCESS_TOKEN_EXPIRE_MINUTES', 60),
    refresh_token_expire_days=getattr(settings, 'REFRESH_TOKEN_EXPIRE_DAYS', 7)
)


# ========== Base Dependencies ==========

async def get_current_token_data(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> TokenData:
    """
    Dependency to extract and validate JWT token.
    
    Args:
        credentials: Bearer token from request header
    
    Returns:
        TokenData extracted from token
    
    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        token = credentials.credentials
        token_data = auth_service.decode_token(token)
        return token_data
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication credentials: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    token_data: TokenData = Depends(get_current_token_data)
) -> User:
    """
    Dependency to get current authenticated user.
    
    Args:
        token_data: Token data from JWT
    
    Returns:
        User object
    
    Raises:
        HTTPException: If user not found or inactive
    """
    user_repo = get_user_repository()
    user_in_db = user_repo.get_user_by_id(token_data.user_id)
    
    if user_in_db is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    if not user_in_db.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    
    # Convert to User (exclude password)
    user = User(**user_in_db.dict(exclude={'hashed_password'}))
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Dependency to ensure user is active.
    
    Args:
        current_user: Current user from token
    
    Returns:
        Active user
    
    Raises:
        HTTPException: If user is inactive
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user account"
        )
    return current_user


# ========== Role-Based Dependencies ==========

def require_role(required_role: UserRole):
    """
    Dependency factory for role-based access control.
    
    Args:
        required_role: Minimum required role
    
    Returns:
        Dependency function
    """
    async def role_checker(
        current_user: User = Depends(get_current_active_user)
    ) -> User:
        """Check if user has required role."""
        role_hierarchy = {
            UserRole.GUEST: 0,
            UserRole.STUDENT: 1,
            UserRole.TEACHER: 2,
            UserRole.ADMIN: 3
        }
        
        user_level = role_hierarchy.get(current_user.role, 0)
        required_level = role_hierarchy.get(required_role, 0)
        
        if user_level < required_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient privileges. Required role: {required_role.value}"
            )
        
        return current_user
    
    return role_checker


# Pre-defined role dependencies
require_admin = require_role(UserRole.ADMIN)
require_teacher = require_role(UserRole.TEACHER)
require_student = require_role(UserRole.STUDENT)


# ========== Permission-Based Dependencies ==========

def require_permission(required_permission: Permission):
    """
    Dependency factory for permission-based access control.
    
    Args:
        required_permission: Required permission
    
    Returns:
        Dependency function
    """
    async def permission_checker(
        token_data: TokenData = Depends(get_current_token_data),
        current_user: User = Depends(get_current_active_user)
    ) -> User:
        """Check if user has required permission."""
        if not current_user.has_permission(required_permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required permission: {required_permission.value}"
            )
        
        return current_user
    
    return permission_checker


def require_permissions(*required_permissions: Permission):
    """
    Dependency factory for multiple permission requirements.
    
    Args:
        *required_permissions: Required permissions (must have all)
    
    Returns:
        Dependency function
    """
    async def permissions_checker(
        token_data: TokenData = Depends(get_current_token_data),
        current_user: User = Depends(get_current_active_user)
    ) -> User:
        """Check if user has all required permissions."""
        missing_permissions = [
            p for p in required_permissions
            if not current_user.has_permission(p)
        ]
        
        if missing_permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required permissions: {', '.join(p.value for p in missing_permissions)}"
            )
        
        return current_user
    
    return permissions_checker


# Pre-defined permission dependencies
require_generate_quiz = require_permission(Permission.GENERATE_QUIZ)
require_send_email = require_permission(Permission.SEND_EMAIL)
require_view_system_stats = require_permission(Permission.VIEW_SYSTEM_STATS)


# ========== Optional Authentication ==========

async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[User]:
    """
    Dependency for optional authentication.
    Returns user if authenticated, None otherwise.
    
    Args:
        credentials: Optional bearer token
    
    Returns:
        User object or None
    """
    if credentials is None:
        return None
    
    try:
        token = credentials.credentials
        token_data = auth_service.decode_token(token)
        
        user_repo = get_user_repository()
        user_in_db = user_repo.get_user_by_id(token_data.user_id)
        
        if user_in_db and user_in_db.is_active:
            return User(**user_in_db.dict(exclude={'hashed_password'}))
        
        return None
        
    except (jwt.InvalidTokenError, Exception) as e:
        logger.debug(f"Optional auth failed: {e}")
        return None


# ========== Helper Functions ==========

def get_auth_service() -> AuthService:
    """
    Get the authentication service instance.
    
    Returns:
        AuthService instance
    """
    return auth_service


async def verify_admin_access(current_user: User = Depends(require_admin)) -> User:
    """
    Dependency for admin-only endpoints.
    
    Args:
        current_user: Current user (must be admin)
    
    Returns:
        Admin user
    """
    return current_user


async def verify_teacher_access(current_user: User = Depends(require_teacher)) -> User:
    """
    Dependency for teacher+ endpoints.
    
    Args:
        current_user: Current user (must be teacher or admin)
    
    Returns:
        Teacher/Admin user
    """
    return current_user