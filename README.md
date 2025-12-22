<p align="center">
  <img src="https://iiitd.ac.in/sites/default/files/images/logo/style1colorindian.png" alt="IIITD Logo" width="200"/>
</p>

<h1 align="center">🤖 IIITD Chatbot</h1>

<p align="center">
  <strong>An AI-powered conversational assistant for IIIT Delhi</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#api-reference">API</a> •
  <a href="#contributing">Contributing</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python"/>
  <img src="https://img.shields.io/badge/Next.js-15-black.svg" alt="Next.js"/>
  <img src="https://img.shields.io/badge/FastAPI-0.100+-green.svg" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/LangChain-latest-orange.svg" alt="LangChain"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"/>
</p>

---

## 📖 Overview

**IIITD Chatbot** is an intelligent conversational assistant designed to help students, faculty, and visitors navigate IIIT Delhi's ecosystem. Built with a sophisticated **Dual-Engine RAG (Retrieval-Augmented Generation)** architecture, it provides accurate, context-aware responses about:

- 📚 **Course Information** - Syllabi, prerequisites, credits, instructors, lecture plans
- 🎓 **Admissions** - B.Tech, M.Tech, Ph.D. admission processes and deadlines
- 🏫 **Campus Life** - Hostels, mess, library, clubs, facilities
- 💼 **Placements** - Company visits, placement statistics, internship drives
- 📋 **Academic Rules** - Grading policies, attendance requirements, examination guidelines
- 🔬 **Research** - Labs, centers, ongoing projects, faculty expertise

---

## ✨ Features

### 🧠 Dual-Engine Architecture
- **Engine A (General KB)**: Handles general IIITD queries using a comprehensive 1600+ line knowledge base
- **Engine B (Course Directory)**: Specialized course retrieval with Waterfall strategy across 177+ courses

### 🔍 Intelligent Query Routing
- Automatic intent classification (course vs. general queries)
- Fast-path detection for course codes and keywords
- Graceful handling of greetings and off-topic queries

### 📊 Hybrid Retrieval
- **Vector Search**: Semantic similarity using HuggingFace embeddings
- **BM25**: Keyword-based retrieval for precise matching
- **Cross-Encoder Reranking**: Result refinement using transformer models

### 🌊 Waterfall Course Retrieval
```
Tier 1: Exact Code Match    → "CSE121" finds exact course
Tier 2: Fuzzy Name Match    → "machine learning course" finds related courses
Tier 3: Instructor Match    → "courses by Dr. X" finds professor's courses
Tier 4: Semantic + BM25     → Fallback for complex queries
```

### 💬 Conversational Memory
- Multi-turn conversation support
- Context-aware query condensation
- Standalone question reformulation

### 🎨 Modern UI
- Clean, responsive Next.js 15 frontend
- Dark/Light theme support
- Real-time streaming responses
- Markdown rendering with syntax highlighting

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Query                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Dual Intent Router                           │
│  ┌─────────────────┐   ┌─────────────────┐   ┌───────────────┐ │
│  │ Fast-Path Check │ → │ Keyword Match   │ → │ LLM Classify  │ │
│  │ (Course Codes)  │   │ (COURSE_KEYWORDS)│   │ (Fallback)    │ │
│  └─────────────────┘   └─────────────────┘   └───────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│    INTENT_COURSE         │    │    INTENT_GENERAL        │
│    (Engine B)            │    │    (Engine A)            │
└──────────────────────────┘    └──────────────────────────┘
              │                               │
              ▼                               ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│  Waterfall Retrieval     │    │  3-Source Hybrid RAG     │
│  ├── Tier 1: Exact Code  │    │  ├── BM25 Keyword Search │
│  ├── Tier 2: Fuzzy Name  │    │  ├── Global Vector Search│
│  ├── Tier 3: Instructor  │    │  └── Scoped Vector+Rerank│
│  └── Tier 4: Semantic    │    │                          │
└──────────────────────────┘    └──────────────────────────┘
              │                               │
              ▼                               ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│  Course-Specific Prompt  │    │  General IIITD Prompt    │
│  (Structured JSON data)  │    │  (Knowledge Base Context)│
└──────────────────────────┘    └──────────────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         LLM Response                            │
│               (Qwen3-14B / Gemini 2.5 Flash)                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
IIITD-Chatbot/
├── 📄 README.md                 # This file
├── 📄 iiitd_kb_master.md        # Master knowledge base (1600+ lines)
│
├── 📂 backend/                  # FastAPI Backend
│   ├── 📄 app.py                # Main application & API endpoints
│   ├── 📄 ingest.py             # General KB ingestion script
│   ├── 📄 ingest_courses.py     # Course JSON ingestion script
│   ├── 📄 requirements.txt      # Python dependencies
│   ├── 📄 Dockerfile            # Container configuration
│   ├── 📄 .env                  # Environment variables
│   │
│   ├── 📂 core/                 # Core modules
│   │   ├── 📄 config.py         # Configuration management
│   │   ├── 📄 ingestion.py      # General data ingestion
│   │   ├── 📄 course_ingestion.py   # Course JSON processing
│   │   ├── 📄 retrieval.py      # Hybrid retriever (Engine A)
│   │   ├── 📄 course_retrieval.py   # Waterfall retriever (Engine B)
│   │   ├── 📄 router.py         # Dual intent router
│   │   └── 📄 generation.py     # RAG pipeline & LLM integration
│   │
│   └── 📂 data/                 # Generated indexes (auto-created)
│       ├── 📂 chroma_db/        # General vector store
│       ├── 📂 course_chroma_db/ # Course vector store
│       ├── 📄 bm25_retriever.pkl
│       ├── 📄 course_bm25_retriever.pkl
│       ├── 📄 course_index.pkl
│       └── 📄 course_master_list.txt
│
├── 📂 Frontend/                 # Next.js 15 Frontend
│   ├── 📄 package.json
│   ├── 📄 next.config.ts
│   ├── 📄 tsconfig.json
│   │
│   └── 📂 src/
│       ├── 📂 app/              # App router pages
│       │   ├── 📄 page.tsx      # Landing page
│       │   └── 📂 chat/         # Chat interface
│       ├── 📂 components/       # React components
│       ├── 📂 hooks/            # Custom hooks
│       └── 📂 lib/              # Utilities
│
└── 📂 jsons/                    # Course JSON data (177 files)
    ├── 📄 CSE121.json
    ├── 📄 ECE214.json
    ├── 📄 BIO213.json
    └── ...
```

---

## 🚀 Installation

### Prerequisites

- **Python** 3.10+
- **Node.js** 18+
- **Git**
- **CUDA** (optional, for GPU acceleration)

### 1. Clone the Repository

```bash
git clone https://github.com/Beingstupid4me/IIITD-Chatbot.git

cd IIITD-Chatbot
```

### 2. Backend Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
cd backend
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the `backend/` directory:

```env
# Option 1: Local Model (Recommended for privacy)
LOCAL_MODEL_API=http://localhost:3000/v1
LOCAL_MODEL_NAME=qwen3-14b

# Option 2: Google Gemini API
GEMINI_API_KEY=your_gemini_api_key_here

# Embedding Model
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

# Reranker Model
RERANKER_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L-6-v2
```

### 4. Ingest Data

```bash
# Ingest general knowledge base
python ingest.py

# Ingest course data (177 JSON files)
python ingest_courses.py
```

### 5. Start the Backend

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 6. Frontend Setup

```bash
cd ../Frontend
npm install
npm run dev
```

The application will be available at:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

---

## 🖥️ Local LLM Setup (Optional)

For enhanced privacy and offline usage, you can run a local LLM using **llama.cpp**:

### 1. Install llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build && cd build
cmake .. -DLLAMA_CUDA=ON  # Enable CUDA for GPU
make -j
```

### 2. Download Model

Download a compatible GGUF model (e.g., Qwen3-14B):

```bash
# Using huggingface-cli
huggingface-cli download Qwen/Qwen3-14B-GGUF Qwen3-14B-Q4_K_M.gguf --local-dir ~/models/
```

### 3. Start the Server

```bash
cd ~/llama.cpp/build/bin

./llama-server \
  -m ~/models/Qwen3-14B-Q4_K_M.gguf \
  --host 0.0.0.0 \
  --port 3000 \
  --ctx-size 32768 \
  --n-gpu-layers 999 \
  --threads 20 \
  --rope-scaling linear
```

---

## 🐳 Docker Deployment

### Build and Run

```bash
cd backend

# Build image
docker build -t iiitd-chatbot-backend .

# Run container
docker run -d \
  --name iiitd-chatbot \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -e GEMINI_API_KEY=your_key \
  iiitd-chatbot-backend
```

### Docker Compose (Recommended)

Create a `docker-compose.yml`:

```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    volumes:
      - ./backend/data:/app/data
      - ./iiitd_kb_master.md:/app/iiitd_kb_master.md
      - ./jsons:/app/jsons
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    restart: unless-stopped

  frontend:
    build: ./Frontend
    ports:
      - "3000:3000"
    depends_on:
      - backend
    environment:
      - NEXT_PUBLIC_API_URL=http://backend:8000
    restart: unless-stopped
```

Run with:

```bash
docker-compose up -d
```

---

## 📡 API Reference

### Base URL

```
http://localhost:8000
```

### Endpoints

#### `POST /chat`

Send a message to the chatbot.

**Request Body:**
```json
{
  "message": "What are the prerequisites for CSE121?",
  "chat_history": [
    {
      "role": "user",
      "content": "Tell me about CSE courses"
    },
    {
      "role": "assistant", 
      "content": "IIITD offers various CSE courses..."
    }
  ]
}
```

**Response:**
```json
{
  "response": "CSE121 (Discrete Mathematics) has the following prerequisites...",
  "sources": ["CSE121.json"],
  "intent": "course",
  "retrieval_tier": "exact_code"
}
```

#### `POST /ingest`

Re-ingest the general knowledge base.

```bash
curl -X POST http://localhost:8000/ingest
```

#### `POST /ingest-courses`

Re-ingest course JSON files.

```bash
curl -X POST http://localhost:8000/ingest-courses
```

#### `GET /status`

Check system health and engine status.

**Response:**
```json
{
  "status": "healthy",
  "engines": {
    "general": true,
    "course": true
  },
  "total_courses": 177,
  "knowledge_base_version": "11.25.01"
}
```

---

## 🧪 Testing

### Run Backend Tests

```bash
cd backend
python -m pytest test_rag.py -v
```

### Manual Testing

```bash
# Test general query
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the hostel fees?"}'

# Test course query
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me about CSE121"}'
```

---

## 📊 Knowledge Base

### General Knowledge Base (`iiitd_kb_master.md`)

The master knowledge base contains **1600+ lines** of curated information organized into sections:

| Section | Content |
|---------|---------|
| **Section 1** | Current Context, Admission Cycles, Placement Updates |
| **Section 2** | Institutional Profile, Campus Infrastructure |
| **Section 3** | Academic Programs (B.Tech, M.Tech, Ph.D.) |
| **Section 4** | Academic Policies, Grading System |
| **Section 5** | Student Life, Hostels, Clubs |
| **Section 6** | Placements, Internships, Statistics |
| **Section 7** | Research Centers, Labs, Projects |
| **Section 8** | Administration, Contacts |

### Course Database (`jsons/`)

**177 course JSON files** with structured data:

```json
{
  "Course Code": "CSE121",
  "Course Name": "Discrete Mathematics",
  "Credits": "4",
  "Course Offered to": ["UG - Core"],
  "Prerequisites": {
    "Mandatory": [],
    "Desirable": ["Basic Mathematics"]
  },
  "Course Outcomes": {...},
  "Weekly Lecture Plan": [...],
  "Assessment Plan": {...}
}
```

---

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LOCAL_MODEL_API` | Local LLM server URL | `null` |
| `LOCAL_MODEL_NAME` | Local model name | `qwen3-14b` |
| `GEMINI_API_KEY` | Google Gemini API key | - |
| `EMBEDDING_MODEL_NAME` | HuggingFace embedding model | `all-MiniLM-L6-v2` |
| `RERANKER_MODEL_NAME` | Cross-encoder reranker model | `ms-marco-MiniLM-L-6-v2` |
| `CHROMA_PERSIST_DIRECTORY` | Vector store path | `./data/chroma_db` |

### Retrieval Parameters

Located in `backend/core/config.py`:

```python
# Retrieval settings
BM25_TOP_K = 10           # BM25 candidates
VECTOR_TOP_K = 10         # Vector search candidates
FINAL_TOP_K = 5           # Final reranked results

# Chunk settings
CHUNK_SIZE = 1000         # Characters per chunk
CHUNK_OVERLAP = 200       # Overlap between chunks
```

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Code Style

- **Python**: Follow PEP 8, use type hints
- **TypeScript**: Use ESLint + Prettier
- **Commits**: Use conventional commit messages

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Author

- **Amartya Singh** 

---

## 🙏 Acknowledgments

- **IIIT Delhi** for providing the knowledge base and course data
- **LangChain** for the excellent RAG framework
- **Hugging Face** for embedding and reranker models
- **llama.cpp** for efficient local LLM inference

---

<p align="center">
  Made with ❤️ at IIIT Delhi
</p>

<p align="center">
  <a href="https://iiitd.ac.in">Website</a> •
  <a href="https://github.com/Beingstupid4me/IIITD-Chatbot/issues">Report Bug</a> •
  <a href="https://github.com/Beingstupid4me/IIITD-Chatbot/issues">Request Feature</a>
</p>
