# CountyIQ Testing GUI

Streamlit app for testing crawls, health, and guardrails.

## Run

From project root (`F:\Projects\countyiq`). Use `python -m streamlit` so the command works without `streamlit` on PATH:

```powershell
# Option 1: Run with project Python (recommended)
python -m streamlit run gui/app.py

# Option 2: Helper script (does the same)
python scripts/run_gui.py
```

Browser opens at `http://localhost:8501`.

## Pages

- **Crawl test** — Dry-run or run crawl for a single county (FIPS) or all counties in a state. Choose categories (property, legal, demographics, etc.).
- **Crawl health** — Refresh and view crawl health (total/crawled/pending/failed counties, document count, last crawl time).
- **RAG / Guardrails test** — Build a synthetic RAG response and run it through the guardrail pipeline (confidence, bias, PII, disclaimers).

## Requirements

- `streamlit` (in `requirements.txt`)
- Project dependencies installed: `pip install -r requirements.txt`
