# 🎓 CampLex — AI College Chatbot (RAG + NLP)

CampLex is a production-grade, Retrieval-Augmented Generation chatbot that answers
student, parent, faculty, and staff questions **exclusively from a college's official
knowledge base**. It never invents information — if the knowledge base doesn't contain
the answer, it says so.

This repository is a working scaffold: real, runnable Clean Architecture code for the
full RAG pipeline, FastAPI backend, Streamlit frontend, auth, and infra — designed so
every layer can be extended without touching the others.

---

## 1. High-Level Architecture

```
                                   ┌────────────────────────┐
                                   │        Students /       │
                                   │  Parents / Faculty /     │
                                   │        Staff             │
                                   └────────────┬─────────────┘
                                                │ HTTPS
                                   ┌────────────▼─────────────┐
                                   │   Streamlit Frontend      │
                                   │ (Chat, Docs, Analytics,   │
                                   │  Admin, Settings, Auth)   │
                                   └────────────┬─────────────┘
                                                │ REST / JWT
                                   ┌────────────▼─────────────┐
                                   │      FastAPI Backend      │
                                   │  ┌──────────────────────┐ │
                                   │  │   API Layer (routers) │ │
                                   │  ├──────────────────────┤ │
                                   │  │  Service Layer (use   │ │
                                   │  │  cases: chat, auth,    │ │
                                   │  │  documents, analytics) │ │
                                   │  ├──────────────────────┤ │
                                   │  │  Repository Layer      │ │
                                   │  │  (SQLAlchemy async)    │ │
                                   │  └──────────────────────┘ │
                                   └───┬─────────────┬────────┘
                                       │             │
                     ┌─────────────────▼─┐      ┌────▼──────────────┐
                     │   PostgreSQL       │      │   RAG Pipeline     │
                     │  users, convos,    │      │  ┌───────────────┐ │
                     │  messages, docs,   │      │  │  Ingestion     │ │
                     │  feedback          │      │  │  (loaders+OCR) │ │
                     └────────────────────┘      │  ├───────────────┤ │
                                                  │  │  Chunking      │ │
                     ┌────────────────────┐      │  ├───────────────┤ │
                     │       Redis         │      │  │  Embeddings    │ │
                     │  rate limiting,     │      │  │  (HF Sentence  │ │
                     │  caching, sessions  │      │  │  Transformers) │ │
                     └────────────────────┘      │  ├───────────────┤ │
                                                  │  │  ChromaDB      │ │
                                                  │  │  (vector store)│ │
                                                  │  ├───────────────┤ │
                                                  │  │  Retrieval +   │ │
                                                  │  │  score filter  │ │
                                                  │  ├───────────────┤ │
                                                  │  │  Prompt Builder│ │
                                                  │  │  (grounding    │ │
                                                  │  │   guardrails)  │ │
                                                  │  ├───────────────┤ │
                                                  │  │  LLM Factory   │ │
                                                  │  │  (Groq default;│ │
                                                  │  │  OpenAI/Gemini/│ │
                                                  │  │  Claude/Azure/ │ │
                                                  │  │  Ollama/Mistral│ │
                                                  │  │  swappable)    │ │
                                                  │  └───────────────┘ │
                                                  └────────────────────┘
```

---

## 2. Component Diagram (Clean Architecture layering)

```
┌───────────────────────────────────────────────────────────────────────┐
│  Presentation Layer                                                    │
│  frontend/  (Streamlit)          backend/app/api/v1/endpoints/ (FastAPI)│
├───────────────────────────────────────────────────────────────────────┤
│  Application / Service Layer  (backend/app/services/)                  │
│  auth_service · chat_service · document_service · analytics_service ·  │
│  memory_service · export_service · rag/ (pipeline) · llm/ (providers)  │
├───────────────────────────────────────────────────────────────────────┤
│  Domain Layer  (backend/app/models/, schemas/)                         │
│  User · Conversation · Message · Document · Feedback (ORM + DTOs)      │
├───────────────────────────────────────────────────────────────────────┤
│  Infrastructure Layer  (backend/app/repositories/, db/, core/)         │
│  Postgres (SQLAlchemy async) · ChromaDB · Redis · JWT · Config         │
└───────────────────────────────────────────────────────────────────────┘
```

**Dependency rule:** outer layers depend on inner layers, never the reverse.
Endpoints depend on services; services depend on repositories/domain; nothing
in `models/`, `schemas/`, or `services/rag/` imports FastAPI. This is what
lets you swap Streamlit for React, or Postgres for MySQL, without rewriting
business logic.

**SOLID in practice here:**
- **S**ingle Responsibility — each service does one job (`ChunkingService` only chunks, `EmbeddingService` only embeds).
- **O**pen/Closed — new LLM providers or document loaders are added by creating a new class + registry entry, never by editing existing classes.
- **L**iskov Substitution — every `BaseLLMProvider` / `BaseLoader` implementation is interchangeable.
- **I**nterface Segregation — `BaseLLMProvider` exposes only `generate`/`stream`, nothing provider-specific leaks upward.
- **D**ependency Inversion — `RAGPipeline` depends on the `BaseLLMProvider` abstraction, injected via `get_llm_provider()`, not a concrete SDK.

---

## 3. RAG Data Flow (a single chat turn)

```
1. User sends message → POST /api/v1/chat/message  (JWT-authenticated)
2. ChatService loads/creates Conversation, pulls bounded history via MemoryService
3. RAGPipeline.answer(query, history):
     a. RetrievalService embeds the query → ChromaDB similarity search (top-K)
     b. Chunks below RETRIEVAL_SCORE_THRESHOLD are discarded
     c. If ZERO chunks remain → return fixed fallback answer, LLM is never called
        (this is the hard guarantee against hallucination)
     d. Else → PromptBuilder assembles a strict system prompt embedding only the
        retrieved chunks + conversation history
     e. LLM provider (Groq Llama-3.3-70B by default) generates the answer
        constrained to that context
4. ChatService persists both messages + source citations (document, category, score)
5. Response returned with: answer, sources[], is_grounded flag, suggested_questions[]
```

### Document ingestion flow (add knowledge without retraining)

```
Upload file (PDF/DOCX/TXT/CSV/HTML/MD, incl. scanned PDFs)
        │
        ▼
LoaderFactory picks the right loader by extension
        │  (PDFLoader auto-detects low-text pages → Tesseract OCR fallback)
        ▼
Raw text → ChunkingService (RecursiveCharacterTextSplitter, 800/120 overlap)
        │
        ▼
EmbeddingService (all-MiniLM-L6-v2) embeds each chunk
        │
        ▼
VectorStoreService writes chunks + metadata (source, category, document_id)
into ChromaDB — searchable immediately, no model retraining, no downtime.
```

---

## 4. Folder Structure

```
CampLex/
├── backend/
│   ├── app/
│   │   ├── main.py                    # FastAPI app factory
│   │   ├── core/                      # config, security (JWT/bcrypt), logging, exceptions
│   │   ├── api/v1/endpoints/          # auth, chat, documents, analytics, admin, health
│   │   ├── api/deps.py                # get_current_user, RBAC guards
│   │   ├── middleware/                # rate limiting, request logging
│   │   ├── models/                    # SQLAlchemy ORM: user, conversation, document
│   │   ├── schemas/                   # Pydantic DTOs
│   │   ├── repositories/              # data access (Postgres)
│   │   ├── services/
│   │   │   ├── rag/                   # chunking, embeddings, vector store, retrieval,
│   │   │   │   └── loaders/           #   prompt builder, ingestion, pipeline
│   │   │   │                          #   PDF(+OCR)/DOCX/TXT/CSV/HTML/MD loaders
│   │   │   ├── llm/                   # BaseLLMProvider + Groq/OpenAI/Gemini/Claude/
│   │   │   │                          #   Azure/Ollama/Mistral + factory
│   │   │   ├── auth_service.py, chat_service.py, document_service.py,
│   │   │   └── analytics_service.py, memory_service.py, export_service.py
│   │   └── db/                        # async session, declarative base
│   ├── tests/{unit,integration}/
│   ├── Dockerfile, requirements.txt, pyproject.toml
├── frontend/
│   ├── app.py                         # Streamlit entry (auth gate)
│   ├── pages/                         # 1_Chat, 2_Documents, 3_Analytics, 4_Admin, 5_Settings
│   ├── components/                    # api_client, auth, chat_ui
│   └── Dockerfile, requirements.txt
├── data/{knowledge_base,uploads,vector_store}/
├── docs/                               # extended architecture/API/deployment docs
├── .github/workflows/ci.yml
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## 5. Technology Justification

| Layer | Choice | Why |
|---|---|---|
| Backend framework | **FastAPI** | Native async, Pydantic validation, auto OpenAPI docs, best-in-class performance for I/O-bound LLM/vector calls |
| Frontend | **Streamlit** | Fast to build a polished internal-facing chat UI with auth, file upload, and charts without a separate JS build pipeline |
| Orchestration | **LangChain** | Standardizes document loaders, text splitters, and LLM/embedding interfaces — swapping vendors is config, not rewrites |
| Vector DB | **ChromaDB** | Embedded, zero-ops persistence for single-institution scale; swappable behind `VectorStoreService` for Pinecone/Qdrant/Weaviate at larger scale |
| Embeddings | **HF Sentence Transformers (all-MiniLM-L6-v2)** | Free, local, fast, no per-call cost or external dependency for the retrieval-critical path |
| Default LLM | **Groq Llama-3.3-70B-Versatile** | Extremely low latency inference, generous free tier, strong instruction-following for grounded QA |
| Relational DB | **PostgreSQL** | ACID guarantees for users/conversations/documents/feedback; async via SQLAlchemy 2.0 |
| Cache/rate-limit | **Redis** | Sub-millisecond rate-limit counters and session/caching layer |
| Auth | **JWT (python-jose) + bcrypt (passlib)** | Stateless, horizontally scalable auth; industry-standard password hashing |
| OCR | **Tesseract + pdf2image** | Open-source, reliable fallback for scanned PDFs (admission forms, notices, scanned circulars) |
| PDF export | **ReportLab** | Deterministic, dependency-light PDF generation for chat transcripts |
| Containerization | **Docker / docker-compose** | Reproducible dev & prod parity, one-command spin-up of Postgres+Redis+API+UI |

---

## 6. Database Design (PostgreSQL — relational metadata)

```
users                          conversations                  messages
──────────────────────         ──────────────────────         ──────────────────────
id (PK, UUID)                  id (PK, UUID)                  id (PK, UUID)
full_name                      user_id (FK -> users.id)       conversation_id (FK)
email (unique)                 title                          role (user/assistant/system)
hashed_password                created_at                     content
role (enum: student/parent/                                   sources (JSON citations)
      faculty/staff/admin)                                    created_at
is_active
department
created_at / updated_at

documents                      feedback
──────────────────────         ──────────────────────
id (PK, UUID)                  id (PK, UUID)
filename                       message_id (FK -> messages.id)
file_type                      rating (1-5)
category (enum: 18 topics)     comment
storage_path                   created_at
file_size_bytes
chunk_count
status (pending/processing/
        indexed/failed)
uploaded_by (FK -> users.id)
error_message
created_at / indexed_at
```

**Vector store (ChromaDB) schema** — one collection (`camplex_knowledge_base`), each
record = `{embedding, page_content, metadata: {document_id, source, category, chunk_index}}`.
`document_id` links every chunk back to its Postgres `documents` row, so deleting a
document cleanly removes all its vectors (`VectorStoreService.delete_by_document_id`).

---

## 7. API Architecture (REST, versioned)

Base path: `/api/v1` · Interactive docs: `/docs` (Swagger) and `/redoc`.

| Endpoint | Method | Auth | Purpose |
|---|---|---|---|
| `/auth/register` | POST | Public | Create account |
| `/auth/login` | POST | Public | Get access + refresh JWT |
| `/auth/me` | GET | User | Current profile |
| `/chat/message` | POST | User | Grounded RAG chat turn |
| `/chat/message/stream` | POST | User | Streaming (typing-indicator) variant |
| `/chat/conversations` | GET | User | List own conversations |
| `/chat/conversations/{id}` | GET | User | Full transcript |
| `/chat/feedback` | POST | User | Thumbs up/down + comment |
| `/documents/upload` | POST | Staff/Admin | Upload + auto-index a file |
| `/documents` | GET | Staff/Admin | List indexed documents |
| `/documents/{id}` | DELETE | Staff/Admin | Remove doc + its vectors |
| `/analytics/summary` | GET | Staff/Admin | Usage KPIs |
| `/admin/users` | GET | Admin | List all users |
| `/admin/users/{id}/deactivate` | PATCH | Admin | Disable account |
| `/health` | GET | Public | Liveness probe |

Every endpoint is a thin controller: parse request → call one service method →
return schema. No business logic lives in `endpoints/`.

---

## 8. Security Architecture

- **Authentication:** JWT access tokens (60 min) + refresh tokens (7 days), `python-jose`, `HS256`.
- **Password storage:** bcrypt via `passlib`, never reversible, never logged.
- **Authorization:** Role-based access control (`student, parent, faculty, staff, admin`) enforced via FastAPI dependency guards (`require_roles`) — decorators, not scattered `if` checks.
- **Input validation:** Every request body is a Pydantic model with length/type constraints; SQLAlchemy parameterized queries eliminate SQL injection.
- **Rate limiting:** `slowapi`, IP-keyed, configurable per-minute ceiling, returns `429` cleanly.
- **CORS:** Explicit allow-list of frontend origins via `ALLOWED_ORIGINS`, never `*` in production.
- **Secrets:** 100% environment-based (`.env`, never committed — see `.gitignore`), no hardcoded keys.
- **Error handling:** Domain exceptions (`CampLexException` hierarchy) mapped centrally to safe HTTP responses — internal stack traces never leak to clients.
- **Logging:** Structured request/response + error logs (`loguru`) with rotation; no PII (passwords, tokens) ever logged.
- **Grounding as a security property:** the fallback-on-empty-context guarantee also functions as a prompt-injection mitigant — since the LLM only ever sees vetted, retrieved chunks plus the fixed system prompt, injected instructions hidden in an uploaded document can't easily override the "answer only from context" rule for *other* documents.

---

## 9. Deployment Architecture

```
docker-compose.yml
 ├── postgres   (persistent volume, healthchecked)
 ├── redis      (rate limiting / cache)
 ├── backend    (FastAPI + Uvicorn, depends_on postgres/redis, mounts ./data)
 └── frontend   (Streamlit, depends_on backend)
```

**Local/dev:** `docker compose up --build` — one command, four services, hot-reloadable.

**Production path (recommended progression):**
1. Push images to a registry (GHCR/ECR/ACR) via the CI pipeline (`.github/workflows/ci.yml` builds both images after tests pass).
2. Deploy `backend` behind a managed load balancer with 2+ replicas (stateless, horizontally scalable — sessions live in JWT, not server memory).
3. Move Postgres to a managed instance (RDS/Cloud SQL/Azure Database) and Redis to a managed cache (ElastiCache/Memorystore).
4. For heavier traffic, move ChromaDB to its dedicated server mode (or swap `VectorStoreService`'s implementation for a managed vector DB) — the rest of the codebase is unaffected because retrieval is accessed only through that one abstraction.
5. Put Streamlit and FastAPI behind a reverse proxy / API gateway (NGINX, Traefik, or cloud-native ALB) with TLS termination.
6. Wire structured logs to a central sink (CloudWatch/ELK) and add uptime checks against `/health`.

**Scalability & extensibility by design:**
- Adding a document → no retraining, ingestion is incremental (`IngestionService`).
- Swapping the LLM → one `.env` line (`LLM_PROVIDER=openai`), zero code changes.
- Swapping the vector DB or relational DB → reimplement one repository/service class behind its existing interface.
- Adding a new document format → implement `BaseLoader`, register in `loader_factory.py`.

---

## 10. Getting Started

```bash
git clone <repo> && cd CampLex
cp .env.example .env         # then fill in your GROQ_API_KEY (or other provider) and JWT_SECRET_KEY
docker compose up --build
```

- Backend API + Swagger docs → http://localhost:8000/docs
- Streamlit UI → http://localhost:8501

### Running tests
```bash
cd backend
pip install -r requirements.txt
pytest tests/unit tests/integration -v --cov=app
```

### Switching the LLM provider
Edit `.env`:
```
LLM_PROVIDER=openai        # groq | openai | gemini | claude | azure_openai | ollama | mistral
LLM_MODEL_NAME=gpt-4o-mini
OPENAI_API_KEY=sk-...
```
No code changes required — `app/services/llm/factory.py` handles the switch.

---

## 11. What's scaffolded vs. what to extend

This repo gives you a fully wired, running Clean Architecture skeleton with a real
grounded RAG pipeline, multi-format ingestion (incl. OCR), pluggable LLMs, JWT/RBAC
auth, and a working Streamlit UI. Before a real production launch, plan to also add:
Alembic migrations for schema changes, a persisted streaming-chat message save path,
refresh-token rotation/blacklisting, per-category retrieval routing, and load testing
against your expected concurrent user count.
