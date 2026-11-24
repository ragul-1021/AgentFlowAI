# AgentFlow AI — backend

FastAPI backend for the AgentFlow AI multi-agent research platform.

**Current status: Phase 1** — foundation and configuration only. The database
probe, SQLAlchemy models, and Alembic migrations arrive in Phase 2.

## Requirements

- Python 3.11+ (verified on Python 3.13)
- A Supabase project with a PostgreSQL connection string (Phase 2+)

## Setup

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt -r requirements-dev.txt
Copy-Item .env.example .env
# edit .env and set DATABASE_URL
```

`DATABASE_URL` must use the `postgresql+asyncpg://` scheme and TLS for remote
hosts. Prefer the Supabase **session pooler** URL (port 5432); do not use the
transaction pooler (port 6543).

## Run

```powershell
.\.venv\Scripts\python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- Health: <http://127.0.0.1:8000/api/health>
- Docs: <http://127.0.0.1:8000/docs>

## Test

```powershell
.\.venv\Scripts\python -m pytest
```

Tests make no network or database calls; provider and database requirements
are checked in later phases.

## Security notes

- `.env` is gitignored; only `.env.example` is committed.
- Database credentials, Gemini/Groq/Tavily keys, and Supabase service-role keys
  are never exposed to the frontend.
- Logs use a JSON formatter with secret redaction.
