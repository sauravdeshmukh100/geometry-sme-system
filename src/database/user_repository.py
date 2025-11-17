#!/usr/bin/env python
"""
User Repository - Complete Implementation
File: src/database/user_repository.py

This provides the get_user_repository() function you need to import.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
import json
from pathlib import Path

from ..models.auth_models import User, UserInDB, UserRole

logger = logging.getLogger(__name__)


class UserRepository:
    """
    Repository for managing user data using JSON file storage.
    """
    
    def __init__(self, persist_to_file: bool = True, storage_path: str = "data/users.json"):
        """
        Initialize user repository.
        
        Args:
            persist_to_file: Whether to persist users to file
            storage_path: Path to user storage file
        """
        # In-memory indexes for O(1) lookups
        self.users_by_id: Dict[str, UserInDB] = {}
        self.users_by_username: Dict[str, UserInDB] = {}
        self.users_by_email: Dict[str, UserInDB] = {}
        
        self.persist_to_file = persist_to_file
        self.storage_path = Path(storage_path)
        
        # Load existing users if persistence enabled
        if self.persist_to_file:
            self._load_from_file()
        
        logger.info(f"UserRepository initialized with {len(self.users_by_id)} users")
    
    def create_user(self, user: UserInDB) -> UserInDB:
        """Create a new user."""
        if user.username in self.users_by_username:
            raise ValueError(f"Username '{user.username}' already exists")
        
        if user.email in self.users_by_email:
            raise ValueError(f"Email '{user.email}' already exists")
        
        self.users_by_id[user.user_id] = user
        self.users_by_username[user.username] = user
        self.users_by_email[user.email] = user
        
        logger.info(f"User created: {user.username} (ID: {user.user_id})")
        
        if self.persist_to_file:
            self._save_to_file()
        
        return user
    
    def get_user_by_id(self, user_id: str) -> Optional[UserInDB]:
        """Get user by ID."""
        return self.users_by_id.get(user_id)
    
    def get_user_by_username(self, username: str) -> Optional[UserInDB]:
        """Get user by username."""
        return self.users_by_username.get(username)
    
    def get_user_by_email(self, email: str) -> Optional[UserInDB]:
        """Get user by email."""
        return self.users_by_email.get(email)
    
    def update_user(self, user_id: str, update_data: Dict[str, Any]) -> Optional[UserInDB]:
        """Update user data."""
        user = self.users_by_id.get(user_id)
        if not user:
            return None
        
        for field, value in update_data.items():
            if field not in ['user_id', 'hashed_password']:
                setattr(user, field, value)
        
        logger.info(f"User updated: {user.username}")
        
        if self.persist_to_file:
            self._save_to_file()
        
        return user
    
    def update_password(self, user_id: str, hashed_password: str) -> Optional[UserInDB]:
        """Update user password."""
        user = self.users_by_id.get(user_id)
        if not user:
            return None
        
        user.hashed_password = hashed_password
        logger.info(f"Password updated for user: {user.username}")
        
        if self.persist_to_file:
            self._save_to_file()
        
        return user
    
    def delete_user(self, user_id: str) -> bool:
        """Delete a user."""
        user = self.users_by_id.get(user_id)
        if not user:
            return False
        
        del self.users_by_id[user_id]
        del self.users_by_username[user.username]
        del self.users_by_email[user.email]
        
        logger.info(f"User deleted: {user.username}")
        
        if self.persist_to_file:
            self._save_to_file()
        
        return True
    
    def list_users(
        self,
        role: Optional[UserRole] = None,
        is_active: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[User]:
        """List users with optional filtering."""
        users = list(self.users_by_id.values())
        
        if role is not None:
            users = [u for u in users if u.role == role]
        
        if is_active is not None:
            users = [u for u in users if u.is_active == is_active]
        
        users = users[offset:offset + limit]
        
        return [User(**u.dict(exclude={'hashed_password'})) for u in users]
    
    def count_users(self, role: Optional[UserRole] = None) -> int:
        """Count users, optionally filtered by role."""
        if role is None:
            return len(self.users_by_id)
        
        return sum(1 for u in self.users_by_id.values() if u.role == role)
    
    def update_last_login(self, user_id: str) -> Optional[UserInDB]:
        """Update user's last login timestamp."""
        user = self.users_by_id.get(user_id)
        if not user:
            return None
        
        user.last_login = datetime.utcnow()
        
        if self.persist_to_file:
            self._save_to_file()
        
        return user
    
    def increment_usage_stats(self, user_id: str, stat_name: str) -> Optional[UserInDB]:
        """Increment user usage statistics."""
        user = self.users_by_id.get(user_id)
        if not user:
            return None
        
        if hasattr(user, stat_name):
            current_value = getattr(user, stat_name)
            setattr(user, stat_name, current_value + 1)
            
            if self.persist_to_file:
                self._save_to_file()
        
        return user
    
    def _save_to_file(self):
        """Save users to JSON file."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            users_data = {
                user_id: user.dict()
                for user_id, user in self.users_by_id.items()
            }
            
            with open(self.storage_path, 'w') as f:
                json.dump(users_data, f, indent=2, default=str)
            
            logger.debug(f"Saved {len(users_data)} users to {self.storage_path}")
            
        except Exception as e:
            logger.error(f"Failed to save users to file: {e}")
    
    def _load_from_file(self):
        """Load users from JSON file."""
        try:
            if not self.storage_path.exists():
                logger.info("No existing user file found, starting with empty database")
                return
            
            with open(self.storage_path, 'r') as f:
                users_data = json.load(f)
            
            for user_id, user_dict in users_data.items():
                if 'created_at' in user_dict and isinstance(user_dict['created_at'], str):
                    user_dict['created_at'] = datetime.fromisoformat(user_dict['created_at'])
                if 'last_login' in user_dict and user_dict['last_login']:
                    user_dict['last_login'] = datetime.fromisoformat(user_dict['last_login'])
                
                user = UserInDB(**user_dict)
                
                self.users_by_id[user.user_id] = user
                self.users_by_username[user.username] = user
                self.users_by_email[user.email] = user
            
            logger.info(f"Loaded {len(self.users_by_id)} users from {self.storage_path}")
            
        except Exception as e:
            logger.error(f"Failed to load users from file: {e}")
            logger.warning("Starting with empty user database")
    
    def user_exists(self, username: Optional[str] = None, email: Optional[str] = None) -> bool:
        """Check if a user exists by username or email."""
        if username and username in self.users_by_username:
            return True
        if email and email in self.users_by_email:
            return True
        return False
    
    def get_user_statistics(self) -> Dict[str, Any]:
        """Get repository statistics."""
        return {
            'total_users': len(self.users_by_id),
            'users_by_role': {
                role.value: self.count_users(role)
                for role in UserRole
            },
            'active_users': sum(1 for u in self.users_by_id.values() if u.is_active),
            'verified_users': sum(1 for u in self.users_by_id.values() if u.is_verified),
            'storage_type': 'JSON file' if self.persist_to_file else 'In-memory',
            'storage_path': str(self.storage_path) if self.persist_to_file else None
        }
    
    def clear(self):
        """Clear all users (use with caution!)."""
        self.users_by_id.clear()
        self.users_by_username.clear()
        self.users_by_email.clear()
        
        if self.persist_to_file:
            self._save_to_file()
        
        logger.warning("All users cleared from repository")


# ========== IMPORTANT: This is what you import ==========

# Global singleton instance
_user_repository = None


def get_user_repository() -> UserRepository:
    """
    Get the global user repository instance.
    
    This is the function you import:
        from src.database.user_repository import get_user_repository
    
    Returns:
        UserRepository instance
    """
    global _user_repository
    if _user_repository is None:
        _user_repository = UserRepository()
    return _user_repository


def reset_user_repository():
    """
    Reset the global user repository instance.
    Useful for testing.
    """
    global _user_repository
    _user_repository = None


# Example usage:
if __name__ == "__main__":
    # This shows how to use the repository
    repo = get_user_repository()
    print(f"Repository initialized with {len(repo.users_by_id)} users")
    print(f"Storage path: {repo.storage_path}")