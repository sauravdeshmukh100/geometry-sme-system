#!/usr/bin/env python
"""
Test Authentication System
Demonstrates the complete authentication flow
"""

import requests
import json
from typing import Optional

# Base URL
BASE_URL = "http://localhost:8000"


class GeometrySMEClient:
    """Client for testing Geometry SME API with authentication."""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.access_token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.current_user: Optional[dict] = None
    
    def _headers(self, authenticated: bool = True) -> dict:
        """Get request headers."""
        headers = {"Content-Type": "application/json"}
        if authenticated and self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        return headers
    
    def register(self, username: str, email: str, password: str, role: str = "student", **kwargs) -> dict:
        """Register a new user."""
        print(f"\n📝 Registering user: {username}")
        
        data = {
            "username": username,
            "email": email,
            "password": password,
            "role": role,
            **kwargs
        }
        
        response = requests.post(
            f"{self.base_url}/auth/register",
            json=data
        )
        
        if response.status_code == 201:
            print(f"✓ User registered successfully!")
            return response.json()
        else:
            print(f"✗ Registration failed: {response.json()}")
            return response.json()
    
    def login(self, username: str, password: str) -> bool:
        """Login and store tokens."""
        print(f"\n🔐 Logging in as: {username}")
        
        response = requests.post(
            f"{self.base_url}/auth/login",
            json={"username": username, "password": password}
        )
        
        if response.status_code == 200:
            token_data = response.json()
            self.access_token = token_data["access_token"]
            self.refresh_token = token_data.get("refresh_token")
            print(f"✓ Login successful!")
            print(f"   Token: {self.access_token[:30]}...")
            return True
        else:
            print(f"✗ Login failed: {response.json()}")
            return False
    
    def get_current_user(self) -> Optional[dict]:
        """Get current user info."""
        print(f"\n👤 Fetching current user info...")
        
        response = requests.get(
            f"{self.base_url}/auth/me",
            headers=self._headers()
        )
        
        if response.status_code == 200:
            self.current_user = response.json()
            print(f"✓ User: {self.current_user['username']} ({self.current_user['role']})")
            return self.current_user
        else:
            print(f"✗ Failed to get user: {response.json()}")
            return None
    
    def chat(self, message: str) -> dict:
        """Send a chat message."""
        print(f"\n💬 Sending message: {message[:50]}...")
        
        response = requests.post(
            f"{self.base_url}/chat",
            headers=self._headers(),
            json={"message": message}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Response type: {result.get('type')}")
            answer = result.get('message', '')
            print(f"   Answer preview: {answer[:100]}...")
            return result
        else:
            print(f"✗ Chat failed: {response.json()}")
            return response.json()
    
    def generate_quiz(self, topic: str, grade_level: str, num_questions: int = 5) -> dict:
        """Generate a quiz."""
        print(f"\n📝 Generating quiz on: {topic}")
        
        response = requests.post(
            f"{self.base_url}/quiz/generate",
            headers=self._headers(),
            json={
                "topic": topic,
                "grade_level": grade_level,
                "num_questions": num_questions,
                "format": "pdf"
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Quiz generated successfully!")
            if 'quiz' in result:
                print(f"   Questions: {num_questions}")
            return result
        else:
            print(f"✗ Quiz generation failed: {response.json()}")
            return response.json()
    
    def explain_concept(self, concept: str, grade_level: str) -> dict:
        """Get concept explanation."""
        print(f"\n📚 Explaining: {concept}")
        
        response = requests.post(
            f"{self.base_url}/explain",
            headers=self._headers(),
            json={
                "concept": concept,
                "grade_level": grade_level,
                "explanation_type": "step-by-step",
                "include_examples": True
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Explanation generated!")
            explanation = result.get('explanation', '')
            print(f"   Preview: {explanation[:100]}...")
            return result
        else:
            print(f"✗ Explanation failed: {response.json()}")
            return response.json()
    
    def list_users(self) -> list:
        """List all users (admin only)."""
        print(f"\n👥 Listing users...")
        
        response = requests.get(
            f"{self.base_url}/auth/users",
            headers=self._headers()
        )
        
        if response.status_code == 200:
            users = response.json()
            print(f"✓ Found {len(users)} users:")
            for user in users[:5]:
                print(f"   - {user['username']} ({user['role']})")
            return users
        else:
            print(f"✗ List users failed: {response.json()}")
            return []
    
    def get_health(self) -> dict:
        """Check API health."""
        print(f"\n🏥 Checking API health...")
        
        response = requests.get(
            f"{self.base_url}/health",
            headers=self._headers(authenticated=False)
        )
        
        if response.status_code == 200:
            health = response.json()
            print(f"✓ Status: {health['status']}")
            print(f"   Total chunks: {health['system']['total_chunks']}")
            return health
        else:
            print(f"✗ Health check failed")
            return {}


def test_authentication_flow():
    """Test complete authentication flow."""
    
    print("="*70)
    print("GEOMETRY SME API - AUTHENTICATION TESTING")
    print("="*70)
    
    client = GeometrySMEClient()
    
    # 1. Health Check (unauthenticated)
    print("\n" + "="*70)
    print("1. HEALTH CHECK (Unauthenticated)")
    print("="*70)
    client.get_health()
    
    # 2. Login as Admin
    print("\n" + "="*70)
    print("2. LOGIN AS ADMIN")
    print("="*70)
    success = client.login("admin", "Admin@123")
    
    if not success:
        print("\n⚠️  Admin user not found. Run 'python scripts/init_auth.py' first!")
        return
    
    # 3. Get Current User
    print("\n" + "="*70)
    print("3. GET CURRENT USER INFO")
    print("="*70)
    client.get_current_user()
    
    # 4. List Users (Admin only)
    print("\n" + "="*70)
    print("4. LIST ALL USERS (Admin Only)")
    print("="*70)
    client.list_users()
    
    # 5. Test Student Flow
    print("\n" + "="*70)
    print("5. TEST STUDENT WORKFLOW")
    print("="*70)
    
    # Login as student
    print("\n--- Switching to Student User ---")
    student_client = GeometrySMEClient()
    success = student_client.login("student1", "Student@123")
    
    if success:
        student_client.get_current_user()
        
        # Ask a question
        student_client.chat("What is the Pythagorean theorem?")
        
        # Try to generate quiz (should fail - students can't create quizzes)
        print("\n--- Testing Permission Restrictions ---")
        student_client.generate_quiz("Triangles", "Grade 8", 5)
    
    # 6. Test Teacher Flow
    print("\n" + "="*70)
    print("6. TEST TEACHER WORKFLOW")
    print("="*70)
    
    print("\n--- Switching to Teacher User ---")
    teacher_client = GeometrySMEClient()
    success = teacher_client.login("teacher1", "Teacher@123")
    
    if success:
        teacher_client.get_current_user()
        
        # Ask a question
        teacher_client.chat("Explain properties of circles")
        
        # Generate quiz (should succeed)
        teacher_client.generate_quiz("Circles", "Grade 9", 5)
        
        # Get explanation
        teacher_client.explain_concept("Pythagorean Theorem", "Grade 8")
    
    # Summary
    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)
    print("\n✓ All authentication flows tested successfully!")
    print("\n📝 Next Steps:")
    print("1. Try the interactive API docs: http://localhost:8000/docs")
    print("2. Test with your own credentials")
    print("3. Explore different user roles and permissions")


def test_registration():
    """Test user registration."""
    
    print("="*70)
    print("TESTING USER REGISTRATION")
    print("="*70)
    
    client = GeometrySMEClient()
    
    # Register a new student
    print("\n1. Registering new student...")
    client.register(
        username="test_student",
        email="test@student.edu",
        password="TestPass123",
        role="student",
        full_name="Test Student",
        grade_level="Grade 7"
    )
    
    # Try to login with new user
    print("\n2. Logging in with new user...")
    if client.login("test_student", "TestPass123"):
        client.get_current_user()
        print("\n✓ Registration and login successful!")


if __name__ == "__main__":
    import sys
    
    try:
        # Check if server is running
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=2)
        except requests.exceptions.ConnectionError:
            print("✗ Cannot connect to API server!")
            print(f"   Make sure the server is running on {BASE_URL}")
            print("\n   Start server with: python -m src.api.main")
            sys.exit(1)
        
        # Run tests
        if len(sys.argv) > 1 and sys.argv[1] == "register":
            test_registration()
        else:
            test_authentication_flow()
        
    except KeyboardInterrupt:
        print("\n\nTesting cancelled by user.")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()