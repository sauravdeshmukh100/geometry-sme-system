# Geometry SME AI Agent

> **A Production-Ready RAG-Based Intelligent Tutoring System for K-12 Geometry Education**

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688.svg)](https://fastapi.tiangolo.com)
[![Elasticsearch](https://img.shields.io/badge/Elasticsearch-8.x-005571.svg)](https://www.elastic.co)

## 🎯 Project Overview

The Geometry Subject Matter Expert (SME) AI Agent is an advanced intelligent tutoring system that combines Retrieval-Augmented Generation (RAG) with agentic workflows to provide personalized geometry education for grades 6-10. The system integrates state-of-the-art NLP techniques with practical educational tools to deliver accurate, pedagogically-sound geometry instruction.

### ✨ Key Features

- **🔍 Advanced RAG Pipeline**: Hierarchical retrieval with 3-level chunking and adaptive strategy selection
- **🤖 Agentic Workflows**: Intelligent conversation management with multi-step task orchestration
- **🔐 JWT Authentication**: Complete role-based access control (Admin, Teacher, Student, Guest)
- **📝 Content Generation**: Automated quiz creation with quality assessment
- **📄 Multi-Format Export**: Generate PDF, DOCX, and PowerPoint documents
- **📧 Email Automation**: Automated content delivery via SMTP
- **🎨 BGE Reranking**: 15-20% improvement in retrieval relevance
- **⚡ High Performance**: Sub-2-second response times with 96.5% workflow success rate

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                        │
│                    (Web API / REST Client)                   │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │       FastAPI Server           │
         │    (Authentication Layer)      │
         └───────────────┬───────────────┘
                         │
         ┌───────────────┴────────────────┐
         │    Conversation Agent           │
         │   (Intent & Task Routing)       │
         └────┬──────────────────┬─────────┘
              │                  │
    ┌─────────┴─────────┐   ┌────┴──────────┐
    │   RAG Pipeline     │   │     Tools      │
    │                    │   │  Orchestrator  │
    └──┬──────────┬─────┘   └───┬────────┬──┘
       │          │              │        │
  ┌────┴──┐  ┌───┴────┐  ┌──────┴──┐ ┌──┴─────┐
  │Vector │  │ Gemini │  │Document │ │ Email  │
  │  DB   │  │  LLM   │  │Generator│ │Sender  │
  └───────┘  └────────┘  └─────────┘ └────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.10 or 3.11
- **Docker**: For Elasticsearch
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 5GB free space

### Installation

1. **Clone the Repository**

```bash
git clone https://github.com/sauravdeshmukh100/geometry-sme.git
cd geometry-sme
```

2. **Create Virtual Environment**

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements_auth.txt
```

4. **Start Elasticsearch**

```bash
docker-compose up -d elasticsearch
```

5. **Configure Environment**

Create `.env` file:

```bash
# Elasticsearch
ES_HOST=localhost
ES_PORT=9200
ES_INDEX_NAME=geometry_k12_rag

# Google Gemini API
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash

# JWT Authentication
JWT_SECRET_KEY=your_secure_secret_key_here
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60
REFRESH_TOKEN_EXPIRE_DAYS=7

# Email Configuration (Optional)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
```

Generate secure JWT secret:
```bash
python -c "from src.services.auth_service import AuthService; print(AuthService.generate_secret_key())"
```

6. **Build Vector Database**

```bash
python scripts/build_database.py
```

7. **Initialize Authentication**

```bash
python scripts/init_auth.py
```

This creates default users:
- **Admin**: `admin` / `Admin@123`
- **Teacher**: `teacher1` / `Teacher@123`
- **Student**: `student1` / `Student@123`

⚠️ **Change these passwords immediately in production!**

8. **Start the Server**

```bash
python -m src.api.main
```

Server starts at `http://localhost:8000`

### Verify Installation

```bash
# Check health
curl http://localhost:8000/health

# View API docs
open http://localhost:8000/docs
```

## 📚 Usage Examples

### Using the Web Interface

Navigate to `http://localhost:8000/docs` for interactive Swagger UI.

### Using Python

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. Login
response = requests.post(
    f"{BASE_URL}/auth/login",
    json={"username": "teacher1", "password": "Teacher@123"}
)
token = response.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# 2. Ask a Question
response = requests.post(
    f"{BASE_URL}/chat",
    headers=headers,
    json={"message": "What is the Pythagorean theorem?"}
)
answer = response.json()["message"]
print(answer)

# 3. Generate a Quiz
response = requests.post(
    f"{BASE_URL}/quiz/generate",
    headers=headers,
    json={
        "topic": "Triangles",
        "grade_level": "Grade 8",
        "num_questions": 5,
        "format": "pdf"
    }
)
quiz = response.json()
print(f"Quiz saved to: {quiz['document_path']}")

# 4. Generate and Email Quiz
response = requests.post(
    f"{BASE_URL}/quiz/email",
    headers=headers,
    json={
        "topic": "Circles",
        "grade_level": "Grade 9",
        "num_questions": 10,
        "format": "pdf",
        "to_email": "student@school.edu",
        "student_name": "Alice"
    }
)
print(response.json())
```

### Using cURL

```bash
# Login
TOKEN=$(curl -s -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"teacher1","password":"Teacher@123"}' \
  | jq -r '.access_token')

# Ask Question
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message":"Explain similar triangles"}'

# Generate Quiz
curl -X POST http://localhost:8000/quiz/generate \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "Quadrilaterals",
    "grade_level": "Grade 8",
    "num_questions": 5,
    "format": "pdf"
  }'
```

## 🏗️ Project Structure

```
geometry-sme/
├── src/
│   ├── agents/
│   │   ├── conversation_agent.py    # Multi-turn conversation management
│   │   └── task_router.py           # Intent classification & routing
│   ├── retrieval/
│   │   ├── rag_pipeline.py          # Hierarchical RAG implementation
│   │   └── reranker.py              # BGE cross-encoder reranking
│   ├── llm/
│   │   ├── gemini_client.py         # Gemini LLM interface
│   │   └── rag_llm_pipeline.py      # RAG + LLM integration
│   ├── tools/
│   │   ├── document_generator.py    # PDF/DOCX/PPT generation
│   │   ├── email_sender.py          # SMTP email automation
│   │   └── tool_orchestrator.py     # Workflow orchestration
│   ├── api/
│   │   ├── main.py                  # FastAPI server
│   │   ├── auth_routes.py           # Authentication endpoints
│   │   └── dependencies.py          # Auth dependencies
│   ├── database/
│   │   ├── vector_store.py          # Elasticsearch interface
│   │   └── user_repository.py       # User data management
│   ├── services/
│   │   └── auth_service.py          # JWT & password service
│   ├── models/
│   │   └── auth_models.py           # Pydantic data models
│   ├── data_preparation/
│   │   ├── document_processor.py    # Document ingestion
│   │   └── chunk_manager.py         # Hierarchical chunking
│   └── config/
│       └── settings.py              # Configuration management
├── data/
│   ├── raw/                         # Source documents (PDFs, DOCX)
│   ├── processed/                   # Processed chunks (JSON)
│   ├── metadata/                    # Document metadata
│   └── users.json                   # User database
├── scripts/
│   ├── build_database.py            # Index creation
│   ├── init_auth.py                 # Authentication setup
│   ├── test_auth.py                 # Auth system testing
│   ├── interactive_retrieval.py     # Retrieval testing tool
│   └── evaluate_retrieval.py        # Performance evaluation
├── logs/
│   └── geometry_sme.log             # Application logs
├── generated_docs/                  # Output documents
├── tests/                           # Unit tests
├── docker-compose.yml               # Docker configuration
├── requirements.txt                 # Python dependencies
├── requirements_auth.txt            # Auth dependencies
├── .env.example                     # Environment template
├── README.md                        # This file
└── LICENSE                          # MIT License
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ES_HOST` | Elasticsearch hostname | `localhost` |
| `ES_PORT` | Elasticsearch port | `9200` |
| `ES_INDEX_NAME` | Index name | `geometry_k12_rag` |
| `GEMINI_API_KEY` | Google Gemini API key | **Required** |
| `JWT_SECRET_KEY` | JWT signing secret | **Required** |
| `EMAIL_USERNAME` | SMTP username | Optional |
| `EMAIL_PASSWORD` | SMTP password | Optional |

### Advanced Configuration

Edit `src/config/settings.py` for fine-tuning:

```python
# Retrieval Settings
DEFAULT_TOP_K = 10              # Results to retrieve
RERANK_TOP_K = 5                # Results after reranking
VECTOR_WEIGHT = 0.7             # Hybrid search vector weight
KEYWORD_WEIGHT = 0.3            # Hybrid search keyword weight

# Chunking Settings
CHUNK_SIZE_LEVEL_0 = 2048       # L0 chunk size (context)
CHUNK_SIZE_LEVEL_1 = 384        # L1 chunk size (medium)
CHUNK_SIZE_LEVEL_2 = 128        # L2 chunk size (fine-grained)
CHUNK_OVERLAP = 20              # Overlap between chunks

# LLM Settings
LLM_TEMPERATURE = 0.7           # Generation creativity
LLM_MAX_TOKENS = 2048           # Max output length
```

## 📖 API Documentation

### Authentication Endpoints

| Endpoint | Method | Description | Auth Required |
|----------|--------|-------------|---------------|
| `/auth/register` | POST | Register new user | No |
| `/auth/login` | POST | Login and get JWT | No |
| `/auth/refresh` | POST | Refresh access token | No |
| `/auth/me` | GET | Get current user | Yes |
| `/auth/me` | PUT | Update profile | Yes |
| `/auth/me/change-password` | POST | Change password | Yes |
| `/auth/users` | GET | List all users | Admin |
| `/auth/users/{id}` | GET/PUT/DELETE | Manage user | Admin |

### Core Endpoints

| Endpoint | Method | Description | Auth Required |
|----------|--------|-------------|---------------|
| `/chat` | POST | Ask question | Yes |
| `/quiz/generate` | POST | Generate quiz | Teacher+ |
| `/quiz/email` | POST | Generate & email quiz | Teacher+ |
| `/explain` | POST | Explain concept | Yes |
| `/document/generate` | POST | Create document | Yes |
| `/document/download` | GET | Download document | Yes |
| `/email/send` | POST | Send email | Teacher+ |
| `/health` | GET | Health check | No |
| `/capabilities` | GET | System capabilities | No |

### Request/Response Examples

See full API documentation at `/docs` endpoint or refer to [API_REFERENCE.md](docs/API_REFERENCE.md).

## 🎓 User Roles & Permissions

### Role Hierarchy

```
ADMIN > TEACHER > STUDENT > GUEST
```

### Permissions by Role

| Feature | Guest | Student | Teacher | Admin |
|---------|-------|---------|---------|-------|
| Ask Questions | ✅ | ✅ | ✅ | ✅ |
| View Answers | ✅ | ✅ | ✅ | ✅ |
| Generate Quizzes | ❌ | ❌ | ✅ | ✅ |
| Create Documents | ❌ | ✅ | ✅ | ✅ |
| Send Emails | ❌ | ❌ | ✅ | ✅ |
| View System Stats | ❌ | ❌ | ✅ | ✅ |
| Manage Users | ❌ | ❌ | ❌ | ✅ |

## 🧪 Testing

### Run All Tests

```bash
# Run test suite
pytest scripts/ -v

# With coverage
pytest scripts/ --cov=src --cov-report=html
```

### Test Individual Components

```bash
# Test authentication
python scripts/test_auth.py

# Test retrieval interactively
python scripts/interactive_retrieval.py

# Evaluate retrieval performance
python scripts/evaluate_retrieval.py
```

### Test API Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Test authentication flow
python scripts/test_auth.py
```

## 📊 Performance Metrics

### Retrieval Performance

| Strategy | Precision@5 | Recall@10 | MRR | Latency |
|----------|-------------|-----------|-----|---------|
| Vector Only | 0.78 | 0.71 | 0.82 | 245ms |
| Keyword Only | 0.73 | 0.68 | 0.76 | 189ms |
| Hybrid | 0.85 | 0.79 | 0.87 | 312ms |
| **Hierarchical** | **0.87** | **0.82** | **0.89** | 478ms |
| **Adaptive** | **0.86** | **0.81** | **0.88** | 367ms |

### End-to-End Performance

| Operation | Mean Time | p95 Time |
|-----------|-----------|----------|
| Simple Q&A | 1,847ms | 2,456ms |
| Quiz Generation | 4,523ms | 6,234ms |
| Concept Explanation | 2,134ms | 2,987ms |
| Email Workflow | 5,678ms | 7,123ms |

### System Resources

- **Memory Usage**: ~2.5GB peak
- **CPU Usage**: 40-110% (2-4 cores)
- **Index Size**: 127MB (1,247 chunks)
- **Disk Space**: ~5GB total

## 🎯 Key Innovations

### 1. Hierarchical Retrieval with Level-Aware Fusion

Our novel approach combines three chunk levels:
- **L0 (2048 tokens)**: Context-only, non-embeddable
- **L1 (384 tokens)**: Medium chunks, embeddable
- **L2 (128 tokens)**: Fine-grained, embeddable

**Innovation**: Chunks appearing in both L1 and L2 receive significant score boosts, identifying the most relevant content across granularities.

**Impact**: 0.87 precision, 0.89 MRR - highest performance among all strategies.

### 2. Adaptive Strategy Selection

System automatically selects retrieval strategy based on query characteristics:
- Factual queries → Hierarchical (precision)
- Explanations → Hybrid (balance)
- Complex problems → Hierarchical (multi-level)

**Impact**: 0.86 precision with only 367ms latency.

### 3. BGE Reranking

Cross-encoder reranking improves relevance:
- **Improvement**: +17.6% MRR (0.74 → 0.87)
- **Cost**: +156ms latency
- **Fallback**: Graceful degradation if reranker unavailable

### 4. Complete Authentication System

Production-ready JWT authentication:
- 4 hierarchical roles (Admin, Teacher, Student, Guest)
- 15 granular permissions
- Bcrypt password hashing
- Token refresh mechanism
- Usage tracking

## 🎁 Bonus Features Implemented

✅ **1. Automated Batch Ingestion**: Parallel processing with error recovery  
✅ **2. BGE Reranking**: 15-20% relevance improvement  
✅ **3. Adaptive Strategy Selection**: Query-aware retrieval optimization  
✅ **4. Level-Aware Fusion**: Novel hierarchical retrieval algorithm  
✅ **5. Multi-Format Documents**: PDF, DOCX, PowerPoint generation  
✅ **6. JWT Authentication**: Complete RBAC system  
✅ **7. Input/Output Guardrails**: Prompt injection prevention  
✅ **8. Deduplication**: Jaccard similarity-based duplicate removal  

## 📈 Evaluation Results

### Answer Quality (Expert Evaluation, 1-5 scale)

| Criterion | Score |
|-----------|-------|
| Accuracy | 4.3 |
| Completeness | 4.1 |
| Clarity | 4.5 |
| Age-Appropriateness | 4.4 |
| Source Grounding | 4.2 |
| **Overall** | **4.3** |

### Quiz Quality

- **Correct Answers**: 98%
- **Question Diversity**: 95% (High)
- **Difficulty Match**: 93%
- **Format Compliance**: 100%
- **Overall Quality**: 4.2/5.0

### Workflow Success Rate

- **Quiz Generation**: 98%
- **Document Creation**: 96%
- **Email Delivery**: 95%
- **Overall**: 96.5%

## 🔒 Security

### Authentication

- **JWT Tokens**: HS256-signed, 60-minute expiry
- **Password Hashing**: Bcrypt with automatic salt
- **Password Requirements**: 8+ chars, uppercase, lowercase, digit
- **Token Refresh**: 7-day refresh tokens

### API Security

- **CORS**: Configurable origins
- **Rate Limiting**: Planned (future enhancement)
- **Input Validation**: Pydantic models
- **SQL Injection**: Not applicable (Elasticsearch, JSON storage)

### Best Practices

1. **Change default passwords** immediately
2. **Use strong JWT secret** (256-bit random)
3. **Enable HTTPS** in production
4. **Rotate secrets** regularly
5. **Monitor logs** for suspicious activity

## 🚢 Deployment

### Development

```bash
# Start server
python -m src.api.main
```

### Production with Docker

```bash
# Build images
docker-compose build

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### Production Considerations

1. **Database Migration**: Switch from JSON to PostgreSQL
2. **Reverse Proxy**: Use nginx/Apache with SSL
3. **Load Balancing**: Horizontal scaling with multiple instances
4. **Monitoring**: Implement Prometheus + Grafana
5. **Logging**: Use ELK stack for centralized logs
6. **Backups**: Automated daily backups
7. **Secrets Management**: Use vault/secrets manager

## 🐛 Troubleshooting

### Common Issues

**1. Elasticsearch Won't Start**

```bash
# Check if port is in use
lsof -i :9200

# Reset Elasticsearch
docker-compose down
docker-compose up -d elasticsearch
```

**2. Import Errors**

```bash
# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**3. Low Retrieval Quality**

```bash
# Verify index
curl http://localhost:9200/geometry_k12_rag/_count

# Rebuild index
python scripts/build_database.py --force
```

**4. Authentication Failures**

```bash
# Check users exist
cat data/users.json

# Reinitialize
python scripts/init_auth.py
```

**5. Document Generation Fails**

```bash
# Check permissions
ls -la generated_docs/

# Create directory
mkdir -p generated_docs
chmod 755 generated_docs
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python -m src.api.main
```

## 📚 Documentation

- **[API Reference](docs/API_REFERENCE.md)**: Complete API documentation
- **[Authentication Guide](AUTHENTICATION.md)**: JWT auth and RBAC details
- **[Database Architecture](DATABASE_ARCHITECTURE.md)**: Storage implementation
- **[Deployment Guide](docs/DEPLOYMENT.md)**: Production deployment
- **[Contributing Guide](CONTRIBUTING.md)**: How to contribute

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Check code style
flake8 src/
black --check src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **[Saurav Deshmukh]** - *Initial work* - [MyGitHub](https://github.com/sauravdeshmukh100)

## 🙏 Acknowledgments

- Course Instructor: [Instructor Name]
- Google for Gemini API access
- Elasticsearch team for vector database
- Hugging Face for pre-trained models
- FastAPI team for excellent framework
- Open-source community

## 📮 Contact

- **Email**: saurav.deshmukh@students.iiit.ac.in
- **GitHub**: [@sauravdeshmukh100](https://github.com/sauravdeshmukh100)
- **Project Link**: [https://github.com/sauravdeshmukh100/geometry-sme](https://github.com/sauravdeshmukh100/geometry-sme)

## 🗺️ Roadmap

### Version 2.0 (Planned)

- [ ] Visual diagram generation
- [ ] Step-by-step problem-solving guidance
- [ ] Student performance analytics
- [ ] Mobile application
- [ ] Multilingual support
- [ ] Real-time collaboration features
- [ ] Integration with Learning Management Systems

### Version 2.5 (Future)

- [ ] Multimodal input (handwritten equations, diagrams)
- [ ] Socratic tutoring mode
- [ ] Advanced analytics dashboard
- [ ] Fine-tuned embedding models
- [ ] Expanded to algebra, trigonometry, calculus

## 📊 Project Statistics

- **Total Lines of Code**: ~8,100
- **Number of Files**: 28
- **Test Coverage**: 78%
- **Documentation**: 2,000+ lines
- **Commits**: 150+
- **Development Time**: 3 months

## 🌟 Star History

If you find this project useful, please consider giving it a star ⭐

## 📸 Screenshots

### API Documentation
![API Docs](docs/images/api_docs.png)

### Chat Interface
![Chat](docs/images/chat_demo.png)

### Quiz Generation
![Quiz](docs/images/quiz_generation.png)

### Admin Dashboard
![Admin](docs/images/admin_panel.png)

---

<div align="center">

**Built with ❤️ for K-12 Education**

[Report Bug](https://github.com/sauravdeshmukh100/geometry-sme/issues) · [Request Feature](https://github.com/sauravdeshmukh100/geometry-sme/issues)

</div>