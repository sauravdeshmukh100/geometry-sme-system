#!/usr/bin/env python
"""
Authentication Initialization Script
Creates default admin user and sets up authentication system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.auth_service import AuthService
from src.database.user_repository import get_user_repository
from src.models.auth_models import UserCreate, UserRole
from src.config.settings import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_default_admin():
    """Create default admin user."""
    
    print("="*70)
    print("GEOMETRY SME - AUTHENTICATION INITIALIZATION")
    print("="*70)
    
    # Initialize services
    auth_service = AuthService(
        secret_key=settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM
    )
    
    user_repo = get_user_repository()
    
    # Check if admin already exists
    admin_exists = user_repo.get_user_by_username("admin")
    if admin_exists:
        print("\n⚠️  Admin user already exists!")
        print(f"   Username: admin")
        print(f"   Email: {admin_exists.email}")
        return
    
    # Prompt for admin details
    print("\n📝 Creating default admin user...")
    print("-" * 70)
    
    # Use default values for quick setup
    admin_email = input("Admin email (default: admin@geometrysme.com): ").strip() or "admin@geometrysme.com"
    admin_password = input("Admin password (default: Admin@123): ").strip() or "Admin@123"
    admin_name = input("Admin full name (default: System Administrator): ").strip() or "System Administrator"
    
    try:
        # Create admin user
        admin_create = UserCreate(
            email=admin_email,
            username="admin",
            password=admin_password,
            full_name=admin_name,
            role=UserRole.ADMIN
        )
        
        user_in_db = auth_service.create_user(admin_create)
        user_repo.create_user(user_in_db)
        
        print("\n✓ Admin user created successfully!")
        print(f"   Username: admin")
        print(f"   Email: {admin_email}")
        print(f"   Role: admin")
        print("\n⚠️  IMPORTANT: Change the default password after first login!")
        
    except Exception as e:
        print(f"\n✗ Failed to create admin user: {e}")
        return
    
    # Create sample users
    print("\n" + "="*70)
    create_samples = input("\nCreate sample users? (y/n, default: y): ").strip().lower()
    
    if create_samples != 'n':
        print("\n📝 Creating sample users...")
        
        sample_users = [
            {
                "email": "teacher@school.edu",
                "username": "teacher1",
                "password": "Teacher@123",
                "full_name": "Sample Teacher",
                "role": UserRole.TEACHER
            },
            {
                "email": "student@school.edu",
                "username": "student1",
                "password": "Student@123",
                "full_name": "Sample Student",
                "role": UserRole.STUDENT,
                "grade_level": "Grade 8"
            }
        ]
        
        for user_data in sample_users:
            try:
                # Extract grade_level if present
                grade_level = user_data.pop("grade_level", None)
                
                user_create = UserCreate(**user_data, grade_level=grade_level)
                user_in_db = auth_service.create_user(user_create)
                user_repo.create_user(user_in_db)
                
                print(f"   ✓ Created {user_data['role'].value}: {user_data['username']}")
                
            except Exception as e:
                print(f"   ✗ Failed to create {user_data['username']}: {e}")
        
        print("\n✓ Sample users created successfully!")
        print("\nSample Credentials:")
        print("  Teacher:")
        print("    Username: teacher1")
        print("    Password: Teacher@123")
        print("\n  Student:")
        print("    Username: student1")
        print("    Password: Student@123")
    
    # Display summary
    print("\n" + "="*70)
    print("SETUP COMPLETE")
    print("="*70)
    
    stats = user_repo.get_user_statistics()
    print(f"\nTotal users: {stats['total_users']}")
    print("Users by role:")
    for role, count in stats['users_by_role'].items():
        if count > 0:
            print(f"  {role}: {count}")
    
    print("\n📝 Next Steps:")
    print("1. Start the API server: python -m src.api.main")
    print("2. Login using POST /auth/login")
    print("3. Use the JWT token in Authorization header: Bearer <token>")
    print("4. Change default passwords immediately!")
    
    print("\n🔗 API Documentation: http://localhost:8000/docs")
    print("="*70)


def show_jwt_secret():
    """Display JWT secret key information."""
    print("\n" + "="*70)
    print("JWT CONFIGURATION")
    print("="*70)
    
    print(f"\nJWT Secret Key: {settings.JWT_SECRET_KEY[:20]}...")
    print(f"Algorithm: {settings.JWT_ALGORITHM}")
    print(f"Access Token Expiry: {settings.ACCESS_TOKEN_EXPIRE_MINUTES} minutes")
    print(f"Refresh Token Expiry: {settings.REFRESH_TOKEN_EXPIRE_DAYS} days")
    
    if settings.JWT_SECRET_KEY.startswith("your-secret-key"):
        print("\n⚠️  WARNING: You are using the default JWT secret key!")
        print("   Generate a secure key: python -c \"from src.services.auth_service import AuthService; print(AuthService.generate_secret_key())\"")
        print("   Set it in your .env file: JWT_SECRET_KEY=<generated-key>")


if __name__ == "__main__":
    try:
        show_jwt_secret()
        print("\n")
        create_default_admin()
        
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Setup failed: {e}", exc_info=True)
        print(f"\n✗ Setup failed: {e}")
        sys.exit(1)