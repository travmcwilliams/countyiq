# CountyIQ

Production-grade RAG SaaS platform that crawls, indexes, and serves US county data (property, legal, demographics, permits, zoning, courts, tax records) and user-uploaded documents through a conversational AI interface.

## Project Structure

```
countyiq/
├── .github/workflows/   # CI/CD (GitHub Actions)
├── api/                 # FastAPI application and route handlers
│   └── routers/
├── crawlers/            # Web crawling logic (one subfolder per data category)
│   ├── property/
│   ├── legal/
│   ├── demographics/
│   ├── permits/
│   ├── zoning/
│   ├── courts/
│   └── tax/
├── data/                # Schemas and configs only (no raw data committed)
│   ├── raw/
│   └── processed/
├── evaluation/          # Accuracy and groundedness evaluation
│   └── accuracy/
├── infra/               # Azure ML workspace, compute, environment YAMLs
├── monitoring/          # Drift detection and performance monitoring
├── pipelines/           # Azure ML pipeline definitions (YAML + Python)
│   ├── ingest/
│   ├── transform/
│   └── index/
├── rag/                 # Embeddings, retrieval, and prompt logic
│   ├── embeddings/
│   ├── retrieval/
│   └── prompts/
└── tests/               # Unit and integration tests
```

## Setup

1. Create a virtual environment:

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and set your keys:

   ```powershell
   Copy-Item .env.example .env
   ```

3. Run tests:

   ```powershell
   pytest tests/
   ```

## Usage

- **API:** `uvicorn api.main:app --reload`
- **Crawlers:** Inherit from `crawlers.base_crawler.BaseCrawler`; implement `crawl()`.

## Tech Stack

- Azure (ML Studio, OpenAI, AI Search), Python 3.14, FastAPI, Pydantic, loguru, Anthropic SDK.

## License

Proprietary.
