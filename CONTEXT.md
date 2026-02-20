# CountyIQ — Session Context Document
Last Updated: February 19, 2026

## What This Project Is
CountyIQ is a production RAG SaaS platform that crawls all publicly available 
US county data (property, legal, demographics, permits, zoning, courts, tax) 
plus user-uploaded documents, served through a conversational AI interface.
Consumer-first, enterprise-ready. 3,235 US counties.

## Critical Rules
- ALWAYS work in F:\Projects\countyiq — NEVER C:\Users\AIMLI\countyiq
- ALWAYS activate venv first: .\.venv\Scripts\Activate.ps1
- ALWAYS use PowerShell syntax
- DocumentCategory enum values are LOWERCASE: .property not .PROPERTY
- Never commit .env — secrets stay local only

## Azure Resources
- Subscription: b094e7ca-556e-47b0-b604-c6a435139129
- Tenant: 8ad254d8-6edb-4086-8512-380bf17d8aed
- Resource Group: AIMLI_Default (East US)
- ML Workspace: connected via infra/connect_workspace.py
- Compute: cpu-cluster (Standard_DS3_v2, min 0 / max 4)
- AI Search: countyiq-search (https://countyiq-search.search.windows.net)
- OpenAI: countyiq-openai (https://countyiq-openai.openai.azure.com/)
- Embedding model: text-embedding-ada-002

## Tech Stack
- Python 3.12.10 at C:\Program Files\Python312\python.exe
- Azure ML SDK v2
- FastAPI + uvicorn
- Pydantic v2
- MLflow 3.9.0
- Anthropic SDK 0.81.0 (claude-sonnet-4-6)
- Azure AI Search SDK
- sentence-transformers (embedding fallback)
- scikit-learn (document classifier)
- loguru (all logging)

## GitHub
- Repo: https://github.com/travmcwilliams/countyiq
- Main branch: main
- Current branch: 2026-02-19-2net (merge to main when stable)

## MCP Servers (in Cursor Settings → Tools & MCP)
- filesystem: F:\Projects\countyiq
- github: travmcwilliams PAT
- azure: subscription + tenant IDs
- claude: Anthropic API key

## Project Structure
countyiq/
├── api/            # FastAPI app, routers, auth, middleware
├── crawlers/       # BaseCrawler, county_registry.json (3235 counties)
├── data/           # schemas/, processed/, raw/ (not committed)
├── evaluation/     # accuracy evaluation
├── frontend/       # consumer UI (index.html)
├── infra/          # Azure ML workspace, compute, environment YAMLs
├── monitoring/     # drift detection, pipeline monitor, cost monitor
├── notebooks/      # exploration
├── pipelines/      # ingest (PDF, CSV, upload, crawl orchestrator)
│                   # transform (document classifier)
├── rag/            # embeddings, retrieval, prompts, pipeline, guardrails
├── scripts/        # run_crawl.py, build_county_registry.py
├── tests/          # 190+ tests, all passing
└── .env            # secrets (never commit)

## Build Phases Status
- ✅ Phase 0 — Problem framing
- ✅ Phase 1 — Azure ML infrastructure
- ✅ Phase 2 — Data engineering (crawlers, PDF, CSV, upload, validation)
- ✅ Phase 3 — Document classifier (TF-IDF + LR, MLflow, model registry)
- ✅ Phase 4+6 — RAG pipeline (embeddings, retrieval, Claude API, search)
- ✅ Phase 5 — MLOps (crawl orchestrator, drift detection, CI/CD)
- ✅ Phase 7 — Responsible AI (confidence, bias, legal, PII guardrails)
- ⏳ Phase 8 — SaaS & scale (auth, UI, cost monitoring) — IN PROGRESS

## Current Status (Phase 8 In Progress)
Files created but need to be verified in C:\Projects\countyiq:
- api/auth.py — API key management, rate limiting
- api/middleware.py — request logging, cost tracking
- api/routers/admin.py — health, costs, crawl status endpoints
- frontend/index.html — consumer UI
- monitoring/cost_monitor.py — cost estimation and alerting
- tests/test_auth.py
- tests/test_middleware.py

## How To Resume
1. Open F:\Projects\countyiq in Cursor (File → Open Folder)
2. Run: .\.venv\Scripts\Activate.ps1
3. Run: python -m pytest tests/ -v --tb=short -q
4. If tests pass, continue Phase 8
5. Start server: uvicorn api.main:app --reload
6. Open UI: http://127.0.0.1:8000

## Next Steps When Resuming
1. Verify all Phase 8 files are in F:\Projects\countyiq
2. Run full test suite — target 190+ tests passing
3. Open consumer UI at http://127.0.0.1:8000
4. Commit: git add . && git commit -m "phase8: complete" && git push origin main
5. Merge branch 2026-02-19-2net to main
6. Run live crawl on 5 pilot counties: 
   python scripts/run_crawl.py --state AL --max-workers 3
7. Test full end-to-end: crawl → index → query → guardrails → UI

## DP-100 Exam Coverage Completed
- Azure ML Workspace and compute targets
- Environment reproducibility
- Supervised learning (document classifier)
- Feature engineering (TF-IDF)
- Model evaluation metrics
- MLflow experiment tracking
- Model registration and versioning
- Data pipeline design and orchestration
- Data drift detection
- Responsible AI (fairness, confidence, compliance)
- CI/CD for ML pipelines
- Online endpoint / inference pipeline pattern

## Key Commands
# Activate venv
.\.venv\Scripts\Activate.ps1

# Run all tests
python -m pytest tests/ -v --tb=short -q

# Start API server
uvicorn api.main:app --reload

# Dry run full crawl
python scripts/run_crawl.py --all --dry-run

# Run crawl for one state
python scripts/run_crawl.py --state AL --max-workers 3

# Run crawl for one county
python scripts/run_crawl.py --fips 01001

# Check crawl health
python scripts/crawl_health_summary.py

# Connect to Azure ML
python -m infra.connect_workspace

# Provision compute
python -m infra.ensure_compute