#!/usr/bin/env python
"""
Authentication Service
Handles JWT token generation, validation, and user authentication
"""

import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging
import secrets
import uuid

from ..models.auth_models import (
    User, UserInDB, UserCreate, TokenData, UserRole, Permission, ROLE_PERMISSIONS
)

logger = logging.getLogger(__name__)


class AuthService:
    """
    Authentication service for JWT token management and password hashing.
    """
    
    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 60,
        refresh_token_expire_days: int = 7
    ):
        """
        Initialize authentication service.
        
        Args:
            secret_key: Secret key for JWT encoding/decoding
            algorithm: JWT algorithm (default: HS256)
            access_token_expire_minutes: Access token expiration time
            refresh_token_expire_days: Refresh token expiration time
        """
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days
        
        logger.info("AuthService initialized")
    
    # ========== Password Hashing ==========
    
    def hash_password(self, password: str) -> str:
        """
        Hash a password using bcrypt.
        
        Args:
            password: Plain text password
        
        Returns:
            Hashed password
        """
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify a password against its hash.
        
        Args:
            plain_password: Plain text password
            hashed_password: Hashed password to compare against
        
        Returns:
            True if password matches, False otherwise
        """
        try:
            return bcrypt.checkpw(
                plain_password.encode('utf-8'),
                hashed_password.encode('utf-8')
            )
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False
    
    # ========== JWT Token Management ==========
    
    def create_access_token(
        self,
        user: User,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create JWT access token for a user.
        
        Args:
            user: User object
            expires_delta: Optional custom expiration time
        
        Returns:
            JWT access token string
        """
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        # Token payload
        payload = {
            "sub": user.user_id,  # Subject (user ID)
            "username": user.username,
            "email": user.email,
            "role": user.role.value,
            "permissions": [p.value for p in user.get_permissions()],
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        logger.info(f"Access token created for user: {user.username}")
        
        return token
    
    def create_refresh_token(self, user: User) -> str:
        """
        Create JWT refresh token for a user.
        
        Args:
            user: User object
        
        Returns:
            JWT refresh token string
        """
        expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        
        payload = {
            "sub": user.user_id,
            "username": user.username,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh",
            "jti": str(uuid.uuid4())  # JWT ID for token revocation
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        logger.info(f"Refresh token created for user: {user.username}")
        
        return token
    
    def decode_token(self, token: str) -> TokenData:
        """
        Decode and validate JWT token.
        
        Args:
            token: JWT token string
        
        Returns:
            TokenData object
        
        Raises:
            jwt.InvalidTokenError: If token is invalid or expired
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            
            # Extract token data
            token_data = TokenData(
                user_id=payload.get("sub"),
                username=payload.get("username"),
                email=payload.get("email"),
                role=UserRole(payload.get("role")),
                permissions=payload.get("permissions", []),
                exp=datetime.fromtimestamp(payload.get("exp")) if payload.get("exp") else None
            )
            
            return token_data
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            raise jwt.InvalidTokenError("Token has expired")
        except jwt.InvalidTokenError as e:
            logger.error(f"Invalid token: {e}")
            raise
        except Exception as e:
            logger.error(f"Token decode error: {e}")
            raise jwt.InvalidTokenError(f"Token decode failed: {str(e)}")
    
    def verify_token(self, token: str) -> bool:
        """
        Verify if token is valid without decoding full data.
        
        Args:
            token: JWT token string
        
        Returns:
            True if valid, False otherwise
        """
        try:
            self.decode_token(token)
            return True
        except jwt.InvalidTokenError:
            return False
    
    # ========== User Authentication ==========
    
    def authenticate_user(
        self,
        user_in_db: UserInDB,
        password: str
    ) -> Optional[User]:
        """
        Authenticate a user with password.
        
        Args:
            user_in_db: User object from database
            password: Plain text password to verify
        
        Returns:
            User object if authentication successful, None otherwise
        """
        if not self.verify_password(password, user_in_db.hashed_password):
            logger.warning(f"Failed authentication attempt for user: {user_in_db.username}")
            return None
        
        if not user_in_db.is_active:
            logger.warning(f"Inactive user attempted login: {user_in_db.username}")
            return None
        
        # Convert UserInDB to User (exclude hashed_password)
        user = User(**user_in_db.dict(exclude={'hashed_password'}))
        logger.info(f"User authenticated successfully: {user.username}")
        
        return user
    
    def create_user(self, user_create: UserCreate) -> UserInDB:
        """
        Create a new user with hashed password.
        
        Args:
            user_create: User creation data
        
        Returns:
            UserInDB object with hashed password
        """
        # Generate unique user ID
        user_id = f"usr_{uuid.uuid4().hex[:12]}"
        
        # Hash password
        hashed_password = self.hash_password(user_create.password)
        
        # Create user object
        user_in_db = UserInDB(
            user_id=user_id,
            email=user_create.email,
            username=user_create.username,
            full_name=user_create.full_name,
            role=user_create.role or UserRole.STUDENT,
            hashed_password=hashed_password,
            is_active=True,
            is_verified=False,
            created_at=datetime.utcnow(),
            grade_level=user_create.grade_level,
            school=user_create.school
        )
        
        logger.info(f"User created: {user_in_db.username} ({user_in_db.role.value})")
        
        return user_in_db
    
    # ========== Authorization ==========
    
    def check_permission(self, token_data: TokenData, required_permission: Permission) -> bool:
        """
        Check if user has required permission.
        
        Args:
            token_data: Token data from decoded JWT
            required_permission: Permission to check
        
        Returns:
            True if user has permission, False otherwise
        """
        return required_permission.value in token_data.permissions
    
    def check_permissions(self, token_data: TokenData, required_permissions: list[Permission]) -> bool:
        """
        Check if user has all required permissions.
        
        Args:
            token_data: Token data from decoded JWT
            required_permissions: List of permissions to check
        
        Returns:
            True if user has all permissions, False otherwise
        """
        return all(p.value in token_data.permissions for p in required_permissions)
    
    def check_role(self, token_data: TokenData, required_role: UserRole) -> bool:
        """
        Check if user has required role or higher.
        
        Role hierarchy: ADMIN > TEACHER > STUDENT > GUEST
        
        Args:
            token_data: Token data from decoded JWT
            required_role: Minimum required role
        
        Returns:
            True if user has sufficient role, False otherwise
        """
        role_hierarchy = {
            UserRole.GUEST: 0,
            UserRole.STUDENT: 1,
            UserRole.TEACHER: 2,
            UserRole.ADMIN: 3
        }
        
        user_level = role_hierarchy.get(token_data.role, 0)
        required_level = role_hierarchy.get(required_role, 0)
        
        return user_level >= required_level
    
    # ========== Token Refresh ==========
    
    def refresh_access_token(self, refresh_token: str, user: User) -> str:
        """
        Generate new access token from refresh token.
        
        Args:
            refresh_token: Valid refresh token
            user: User object
        
        Returns:
            New access token
        
        Raises:
            jwt.InvalidTokenError: If refresh token is invalid
        """
        try:
            # Decode refresh token
            payload = jwt.decode(
                refresh_token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            
            # Verify token type
            if payload.get("type") != "refresh":
                raise jwt.InvalidTokenError("Not a refresh token")
            
            # Verify user ID matches
            if payload.get("sub") != user.user_id:
                raise jwt.InvalidTokenError("Token user mismatch")
            
            # Create new access token
            return self.create_access_token(user)
            
        except jwt.ExpiredSignatureError:
            logger.warning("Refresh token has expired")
            raise jwt.InvalidTokenError("Refresh token has expired")
        except jwt.InvalidTokenError as e:
            logger.error(f"Invalid refresh token: {e}")
            raise
    
    # ========== Utility Methods ==========
    
    @staticmethod
    def generate_secret_key() -> str:
        """
        Generate a secure random secret key.
        
        Returns:
            Base64-encoded random secret key
        """
        return secrets.token_urlsafe(32)
    
    def get_token_expiration(self, token: str) -> Optional[datetime]:
        """
        Get expiration time of a token.
        
        Args:
            token: JWT token string
        
        Returns:
            Expiration datetime or None if invalid
        """
        try:
            token_data = self.decode_token(token)
            return token_data.exp
        except jwt.InvalidTokenError:
            return None
    
    def is_token_expired(self, token: str) -> bool:
        """
        Check if token is expired.
        
        Args:
            token: JWT token string
        
        Returns:
            True if expired, False otherwise
        """
        try:
            self.decode_token(token)
            return False
        except jwt.ExpiredSignatureError:
            return True
        except jwt.InvalidTokenError:
            return True