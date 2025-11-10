# Geometry Subject Matter Expert AI Agent

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A Retrieval-Augmented Generation (RAG) system for K-12 Geometry education, specializing in Grades 6-10 curriculum.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Data Collection](#data-collection)
- [Usage Examples](#usage-examples)
- [Evaluation](#evaluation)
- [Project Status](#project-status)
- [Team](#team)
- [License](#license)

---

## 🎯 Overview

This project implements an advanced **Subject Matter Expert (SME) AI Agent** for K-12 Geometry education. The system uses state-of-the-art RAG techniques to provide accurate, grade-appropriate answers to geometry questions, generate educational content, and assist both students and teachers.

**Domain:** K-12 Education - Geometry (Shapes, Angles, Theorems)  
**Target Grades:** 6-10  
**Project Type:** Major Project - LMA Monsoon 2025

### Key Capabilities

- ✅ **Question Answering**: Grade-appropriate geometry explanations
- ✅ **Hierarchical Retrieval**: Multi-level context retrieval (2048/512/128 tokens)
- ✅ **Hybrid Search**: Combines semantic (vector) + keyword (BM25) search
- ✅ **Reranking**: BGE CrossEncoder for improved relevance
- ✅ **Grade-Specific Filtering**: Target specific educational levels
- ✅ **Metadata-Rich**: Topics, difficulty, formulas, theorems

---

## 🚀 Features

### Core Features (Required)

| Feature | Status | Description |
|---------|--------|-------------|
| Multi-format Document Processing | ✅ | PDF, DOCX, PPTX, TXT, MD support |
| Hierarchical Chunking | ✅ | 3-level chunking (2048, 512, 128 tokens) |
| Vector Embeddings | ✅ | sentence-transformers (768-dim) |
| Elasticsearch Indexing | ✅ | Dense vector + BM25 keyword search |
| RAG Pipeline | ✅ | Complete retrieval system |
| Metadata Extraction | ✅ | Grade, difficulty, topics |

### Bonus Features (Implemented)

| Feature | Status | Description |
|---------|--------|-------------|
| BGE Reranker | ✅ | 15-30% relevance improvement |
| Hybrid Search | ✅ | Reciprocal Rank Fusion (RRF) |
| Redis Caching | ✅ | 5x faster repeated queries |
| Context Expansion | ✅ | Parent-child chunk relationships |
| Grade Classification | ✅ | Automatic & filename-based |

---

## 🏗️ System Architecture

```
┌─────────────────┐
│  Raw Documents  │ (PDF, DOCX, PPTX)
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│   Document Processor            │
│  - Format detection             │
│  - Text extraction              │
│  - Grade classification         │
│  - Metadata extraction          │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   Chunk Manager                 │
│  - Level 0: 2048 tokens         │
│  - Level 1: 512 tokens          │
│  - Level 2: 128 tokens          │
│  - 20-token overlap             │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   Embedding Generation          │
│  - all-mpnet-base-v2            │
│  - 768 dimensions               │
│  - Redis caching                │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   Elasticsearch Index           │
│  - Vector search (cosine)       │
│  - Keyword search (BM25)        │
│  - Metadata filtering           │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│   RAG Pipeline                  │
│  - Hybrid retrieval             │
│  - BGE reranking                │
│  - Context assembly             │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Query Results  │
└─────────────────┘
```

---

## 💻 Installation

### Prerequisites

- Python 3.10 or higher
- Docker & Docker Compose (for Elasticsearch & Redis)
- 8GB RAM minimum (16GB recommended)
- GPU optional (for faster embedding generation)

### Step 1: Clone Repository

```bash
git clone https://github.com/sauravdeshmukh100/geometry-sme-agent.git
cd geometry-sme-agent
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Start Infrastructure

```bash
# Start Elasticsearch and Redis
docker-compose up -d

# Wait 30 seconds for services to initialize
# Verify services are running
curl http://localhost:9200/_cluster/health
redis-cli ping
```

### Step 5: Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings (optional)
nano .env
```

---

## 🚀 Quick Start

### 1. Prepare Data

Place your geometry documents in `data/raw/`:

```bash
# Example structure
data/raw/
├── class6_9.pdf
├── class7_5.pdf
├── class10_11.pdf
├── Geometry Textbook.pdf
└── presentations/
    └── geometry_intro.pptx
```

### 2. Build Database

```bash
# Process documents and build vector database
python scripts/build_database.py
```

Expected output:
```
Processing: class6_9.pdf
  ✓ Grade: Grade 6, Difficulty: Beginner
...
Total processed: 19/19
Total chunks created: 2,547
```

### 3. Verify Setup

```bash
# Verify data processing
python scripts/verify_data_processing.py

# Test retrieval system
python scripts/test_rag_pipeline.py
```

### 4. Interactive Testing

```bash
# Start interactive query interface
python scripts/interactive_retrieval.py

# Try queries:
> What is the Pythagorean theorem?
> Explain properties of triangles
> How to calculate area of a circle?
```

---

## 📁 Project Structure

```
geometry-sme-system/
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py              # Configuration management
│   ├── data_preparation/
│   │   ├── __init__.py
│   │   ├── document_processor.py     # Multi-format document processing
│   │   ├── chunk_manager.py          # Hierarchical chunking
│   │   └── metadata_extractor.py     # Metadata extraction
│   ├── database/
│   │   ├── __init__.py
│   │   ├── elasticsearch_client.py   # Elasticsearch interface
│   │   └── vector_store.py           # Vector storage & retrieval
│   └── retrieval/
│       ├── __init__.py
│       ├── rag_pipeline.py           # Complete RAG pipeline
│       └── reranker.py               # BGE reranking
├── scripts/
│   ├── build_database.py             # Main database builder
│   ├── verify_data_processing.py     # Data verification
│   ├── test_rag_pipeline.py          # Integration tests
│   └── interactive_retrieval.py      # Interactive query tool
├── tests/
│   ├── test_vector_store.py          # Unit tests
│   ├── test_reranker.py
│   └── test_rag_pipeline.py
├── data/
│   ├── raw/                          # Source documents
│   ├── processed/                    # Processed JSON files
│   └── metadata/                     # Metadata cache
├── docs/
│   ├── data_collection_report.md     # Data documentation
│   ├── system_architecture.md        # Architecture details
│   └── implementation_report.md      # Implementation notes
├── logs/
│   └── geometry_sme.log              # System logs
├── requirements.txt                   # Python dependencies
├── .env                              # Environment configuration
├── docker-compose.yml                # Infrastructure setup
└── README.md                         # This file
```

---

## 📚 Data Collection

### Sources

Our dataset comprises **19 documents** from authoritative sources:

#### 1. NCERT Textbooks (17 documents)
- **Source**: Official Indian school curriculum
- **Grades**: Classes 6-10
- **Format**: `classX_Y.pdf` (X=class, Y=chapter)
- **Example**: `class6_9.pdf` (Class 6, Chapter 9)
- **Coverage**: Shapes, angles, triangles, quadrilaterals, circles, mensuration

#### 2. Comprehensive Textbook (1 document)
- **Title**: *Geometry for Enjoyment and Challenge*
- **Publisher**: McDougal Littell
- **Pages**: 770
- **Grades**: 6-12
- **Coverage**: Basic to advanced geometry with proofs

#### 3. Presentation Materials (1+ documents)
- **Format**: PPTX
- **Purpose**: Visual learning aids
- **Coverage**: Topic introductions, visual explanations

### Statistics

| Metric | Value |
|--------|-------|
| Total Documents | 19 |
| Total Words | 81,564 |
| Estimated Chunks | 2,500+ |
| Grade 6 Documents | 5 (26.3%) |
| Grade 7 Documents | 2 (10.5%) |
| Grade 8 Documents | 1 (5.3%) |
| Grade 9 Documents | 6 (31.6%) |
| Grade 10 Documents | 4 (21.1%) |
| Beginner Difficulty | 52.6% |
| Intermediate Difficulty | 36.8% |
| Advanced Difficulty | 10.5% |

---

## 💡 Usage Examples

### Example 1: Basic Query

```python
from src.retrieval.rag_pipeline import GeometryRAGPipeline, RetrievalConfig

# Initialize pipeline
pipeline = GeometryRAGPipeline(enable_reranker=True)

# Configure retrieval
config = RetrievalConfig(
    top_k=5,
    rerank=True,
    rerank_top_k=3
)

# Query
result = pipeline.retrieve("What is the Pythagorean theorem?", config)

print(f"Found {len(result.chunks)} chunks")
print(f"Context: {result.context[:200]}...")
```

### Example 2: Grade-Specific Query

```python
from src.retrieval.rag_pipeline import RetrievalStrategy

# Search for Grade 6 content only
config = RetrievalConfig(
    strategy=RetrievalStrategy.HYBRID,
    filters={'grade_level': 'Grade 6'},
    top_k=5
)

result = pipeline.retrieve("Explain basic shapes", config)
```

### Example 3: Advanced Retrieval

```python
# Hierarchical retrieval with context expansion
config = RetrievalConfig(
    strategy=RetrievalStrategy.HIERARCHICAL,
    top_k=10,
    rerank=True,
    include_parents=True,  # Include broader context
    use_metadata_boost=True
)

result = pipeline.retrieve("Circle theorems with proofs", config)
```

---

## 📊 Evaluation

### Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| Average Query Time (no rerank) | 80ms | With cache |
| Average Query Time (with rerank) | 280ms | First query |
| Average Query Time (cached) | 10ms | Repeated query |
| Retrieval Accuracy (Top-3) | 92% | Manual evaluation |
| Grade Classification Accuracy | 95% | NCERT documents |
| Index Size | 150 MB | Including embeddings |

### Test Results

```bash
$ python scripts/test_rag_pipeline.py

TEST SUMMARY
─────────────────────────────────────
Total Tests: 8
Passed: 8 ✓
Failed: 0 ✗
Success Rate: 100.0%
```

---

## 🎯 Project Status

### Completed (Phase 1 & 2)

- [x] **Document Processing** - Multi-format support
- [x] **Hierarchical Chunking** - 3-level strategy
- [x] **Embedding Generation** - sentence-transformers
- [x] **Vector Database** - Elasticsearch indexed
- [x] **RAG Pipeline** - Complete retrieval system
- [x] **Hybrid Search** - Vector + BM25
- [x] **Reranking** - BGE CrossEncoder
- [x] **Caching** - Redis integration
- [x] **Testing Suite** - Comprehensive tests

### In Progress (Phase 3)

- [ ] **LLM Integration** - Answer generation
- [ ] **Content Generation** - Quiz/explanation generation
- [ ] **Multi-step Reasoning** - Chain-of-thought

### Planned (Phase 4-6)

- [ ] **API Server** - FastAPI REST API
- [ ] **Tool Integration** - PDF/DOCX/PPT generation
- [ ] **Email Automation** - Report delivery
- [ ] **Fine-tuning** - Domain-specific LLM
- [ ] **Self-learning** - Human feedback integration

---

## 🛠️ Configuration

Key configuration options in `.env`:

```bash
# Elasticsearch
ES_HOST=localhost
ES_PORT=9200
ES_INDEX_NAME=geometry_k12_rag

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
EMBEDDING_DIMENSION=768
DEVICE=cuda  # or cpu

# Chunking
CHUNK_SIZE_LEVEL_0=2048
CHUNK_SIZE_LEVEL_1=512
CHUNK_SIZE_LEVEL_2=128
CHUNK_OVERLAP=20

# Reranker
RERANKER_MODEL=BAAI/bge-reranker-base
ENABLE_RERANKER=true

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Retrieval
DEFAULT_TOP_K=10
RERANK_TOP_K=5
VECTOR_WEIGHT=0.7
KEYWORD_WEIGHT=0.3
```

---

## 🧪 Testing

### Run All Tests

```bash
# Unit tests
pytest tests/ -v

# Integration tests
python scripts/test_rag_pipeline.py

# Coverage report
pytest tests/ --cov=src --cov-report=html
```

### Manual Testing

```bash
# Interactive testing
python scripts/interactive_retrieval.py

# Test queries:
> What is the Pythagorean theorem?
> Explain properties of isosceles triangles
> How to calculate area of a circle?
> Prove that sum of angles in triangle is 180 degrees
```

---

## 📈 Performance Optimization

### Tips for Better Performance

1. **Enable GPU**: Set `DEVICE=cuda` in `.env` for 5x faster embeddings
2. **Use Caching**: Redis cache reduces query time by 80%
3. **Batch Queries**: Process multiple queries together
4. **Adjust top_k**: Lower values = faster retrieval
5. **Disable Reranking**: For speed-critical applications

### Scaling Recommendations

- **<1000 documents**: Current setup works well
- **1000-10000 documents**: Add more RAM, consider GPU
- **>10000 documents**: Distributed Elasticsearch cluster

---

## 🐛 Troubleshooting

### Common Issues

**Issue**: Elasticsearch connection refused
```bash
# Solution: Check if ES is running
docker-compose ps
docker-compose restart elasticsearch
```

**Issue**: Out of memory during indexing
```bash
# Solution: Reduce batch size in build_database.py
# Line ~150: batch_size=32 → batch_size=16
```

**Issue**: Slow query performance
```bash
# Solution 1: Enable Redis caching
# Solution 2: Use vector-only search (faster)
# Solution 3: Reduce top_k value
```

**Issue**: Import errors
```bash
# Solution: Ensure all __init__.py files exist
touch src/__init__.py
touch src/retrieval/__init__.py
```

---

## 👥 Team

- **Team Size**: [1/2]
- **Domain**: K-12 Education - Geometry
- **Institution**: [IIITH]
- **Course**: Major Project - LMA Monsoon 2025

### Contributors

- **[Saurav Deshmukh]** - [Leader] - [saurav.deshmukh@students.iiit.ac.in]
- **[Shubham Raut]** - [Support(junior)] - [shubham.raut@students.iiit.ac.in] 

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Contact & Support

- **GitHub Issues**: [Project Issues](https://github.com/sauravdeshmukh100/geometry-sme-agent/issues)
- **Email**: [saurav.deshmukh@students.iiit.ac.in]
- **Documentation**: See `docs/` folder for detailed documentation

---

## 🙏 Acknowledgments

- **NCERT** - For authoritative curriculum materials
- **McDougal Littell** - Geometry textbook resources
- **Sentence Transformers** - Embedding models
- **Elasticsearch** - Vector search capabilities
- **Project Guide** - [Vasudev Sir]

---

## 📚 References

1. [Sentence Transformers Documentation](https://www.sbert.net/)
2. [Elasticsearch Guide](https://www.elastic.co/guide/)
3. [BGE Reranker Paper](https://arxiv.org/abs/2309.07597)
4. [RAG Best Practices](https://python.langchain.com/docs/use_cases/question_answering/)

---

## 🎓 Citation

If you use this project in your research, please cite:

```bibtex
@misc{geometry-sme-2025,
  title={Geometry Subject Matter Expert AI Agent},
  author={Saurav Deshmukh},
  year={2025},
  publisher={GitHub},
  url={https://github.com/sauravdeshmukh100/geometry-sme-agent}
}
```

---

<div align="center">
  <p>Made with ❤️ for K-12 Education</p>
  <p>⭐ Star this repo if you find it helpful!</p>
</div>
