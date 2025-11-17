#!/usr/bin/env python
"""
Authentication and User Models
Defines user roles, permissions, and authentication data structures
"""

from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class UserRole(str, Enum):
    """User roles with hierarchical permissions."""
    ADMIN = "admin"              # Full system access
    TEACHER = "teacher"          # Can create quizzes, view all student data
    STUDENT = "student"          # Limited access to learning features
    GUEST = "guest"              # Read-only access


class Permission(str, Enum):
    """Granular permissions for different operations."""
    # Question & Answer
    ASK_QUESTIONS = "ask_questions"
    VIEW_ANSWERS = "view_answers"
    
    # Quiz Management
    GENERATE_QUIZ = "generate_quiz"
    VIEW_QUIZ = "view_quiz"
    EDIT_QUIZ = "edit_quiz"
    DELETE_QUIZ = "delete_quiz"
    
    # Document Generation
    GENERATE_DOCUMENT = "generate_document"
    DOWNLOAD_DOCUMENT = "download_document"
    
    # Email
    SEND_EMAIL = "send_email"
    
    # User Management (Admin only)
    CREATE_USER = "create_user"
    VIEW_USERS = "view_users"
    EDIT_USER = "edit_user"
    DELETE_USER = "delete_user"
    
    # System
    VIEW_SYSTEM_STATS = "view_system_stats"
    MODIFY_SYSTEM_CONFIG = "modify_system_config"


# Role-Permission Mapping
ROLE_PERMISSIONS: Dict[UserRole, List[Permission]] = {
    UserRole.ADMIN: [p for p in Permission],  # All permissions
    
    UserRole.TEACHER: [
        Permission.ASK_QUESTIONS,
        Permission.VIEW_ANSWERS,
        Permission.GENERATE_QUIZ,
        Permission.VIEW_QUIZ,
        Permission.EDIT_QUIZ,
        Permission.DELETE_QUIZ,
        Permission.GENERATE_DOCUMENT,
        Permission.DOWNLOAD_DOCUMENT,
        Permission.SEND_EMAIL,
        Permission.VIEW_SYSTEM_STATS,
    ],
    
    UserRole.STUDENT: [
        Permission.ASK_QUESTIONS,
        Permission.VIEW_ANSWERS,
        Permission.VIEW_QUIZ,
        Permission.GENERATE_DOCUMENT,
        Permission.DOWNLOAD_DOCUMENT,
    ],
    
    UserRole.GUEST: [
        Permission.ASK_QUESTIONS,
        Permission.VIEW_ANSWERS,
    ]
}


class User(BaseModel):
    """User model with authentication and profile information."""
    user_id: str = Field(..., description="Unique user identifier")
    email: EmailStr = Field(..., description="User email address")
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    full_name: Optional[str] = Field(None, description="User's full name")
    role: UserRole = Field(UserRole.STUDENT, description="User role")
    is_active: bool = Field(True, description="Whether user account is active")
    is_verified: bool = Field(False, description="Whether email is verified")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    
    # Profile information
    grade_level: Optional[str] = Field(None, description="For students: current grade")
    school: Optional[str] = Field(None, description="School/institution name")
    preferences: Dict[str, Any] = Field(default_factory=dict, description="User preferences")
    
    # Usage tracking
    total_questions_asked: int = Field(0, description="Total questions asked")
    total_quizzes_generated: int = Field(0, description="Total quizzes generated")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "usr_123456",
                "email": "student@school.edu",
                "username": "john_doe",
                "full_name": "John Doe",
                "role": "student",
                "grade_level": "Grade 8",
                "school": "Springfield High School"
            }
        }
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if user has a specific permission."""
        return permission in ROLE_PERMISSIONS.get(self.role, [])
    
    def get_permissions(self) -> List[Permission]:
        """Get all permissions for this user's role."""
        return ROLE_PERMISSIONS.get(self.role, [])


class UserInDB(User):
    """User model with hashed password (for database storage)."""
    hashed_password: str = Field(..., description="Bcrypt hashed password")


class UserCreate(BaseModel):
    """Request model for user registration."""
    email: EmailStr = Field(..., description="Email address")
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    password: str = Field(..., min_length=8, max_length=100, description="Password")
    full_name: Optional[str] = Field(None, description="Full name")
    role: Optional[UserRole] = Field(UserRole.STUDENT, description="User role")
    grade_level: Optional[str] = None
    school: Optional[str] = None
    
    @validator('password')
    def validate_password(cls, v):
        """Validate password strength."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v
    
    @validator('username')
    def validate_username(cls, v):
        """Validate username format."""
        if not v.replace('_', '').isalnum():
            raise ValueError("Username can only contain letters, numbers, and underscores")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "email": "newuser@school.edu",
                "username": "new_student",
                "password": "SecurePass123",
                "full_name": "New Student",
                "role": "student",
                "grade_level": "Grade 9"
            }
        }


class UserLogin(BaseModel):
    """Request model for user login."""
    username: str = Field(..., description="Username or email")
    password: str = Field(..., description="Password")
    
    class Config:
        json_schema_extra = {
            "example": {
                "username": "john_doe",
                "password": "SecurePass123"
            }
        }


class Token(BaseModel):
    """JWT token response model."""
    access_token: str = Field(..., description="JWT access token")
    refresh_token: Optional[str] = Field(None, description="JWT refresh token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 3600
            }
        }


class TokenData(BaseModel):
    """Data extracted from JWT token."""
    user_id: str
    username: str
    email: str
    role: UserRole
    permissions: List[str]
    exp: Optional[datetime] = None


class UserUpdate(BaseModel):
    """Request model for updating user profile."""
    full_name: Optional[str] = None
    grade_level: Optional[str] = None
    school: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "full_name": "John Updated Doe",
                "grade_level": "Grade 9",
                "preferences": {
                    "difficulty": "intermediate",
                    "notifications": True
                }
            }
        }


class PasswordChange(BaseModel):
    """Request model for password change."""
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, description="New password")
    
    @validator('new_password')
    def validate_password(cls, v):
        """Validate new password strength."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class RefreshTokenRequest(BaseModel):
    """Request model for refreshing access token."""
    refresh_token: str = Field(..., description="Refresh token")
    
    class Config:
        json_schema_extra = {
            "example": {
                "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
            }
        }