"""
CountyIQ Testing GUI — crawl, health, and pipeline testing.

Run from project root: streamlit run gui/app.py
"""

import sys
from pathlib import Path

# Ensure project root is on path when running via streamlit
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

st.set_page_config(
    page_title="CountyIQ Testing",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🗺️ CountyIQ Testing GUI")
st.caption("Crawl, health, and pipeline testing from one place.")

sidebar = st.sidebar
sidebar.header("Navigation")
page = sidebar.radio(
    "Go to",
    ["Crawl test", "Crawl health", "RAG / Guardrails test"],
    index=0,
)

# --- Crawl test page ---
if page == "Crawl test":
    st.header("Crawl test")
    st.markdown("Dry-run or run the county crawl pipeline for a state or single county.")

    mode = st.radio(
        "Mode",
        ["Dry run (preview only)", "Run crawl"],
        horizontal=True,
    )
    scope = st.radio(
        "Scope",
        ["Single county (FIPS)", "All counties in a state"],
        horizontal=True,
    )

    if scope == "Single county (FIPS)":
        fips = st.text_input(
            "FIPS code (5 digits)",
            value="01001",
            help="e.g. 01001 = Autauga County, AL",
        )
        state_abbr = None
    else:
        state_abbr = st.text_input(
            "State abbreviation",
            value="AL",
            max_chars=2,
            help="e.g. AL, CA, TX",
        ).strip().upper()
        fips = None

    categories = st.multiselect(
        "Categories",
        ["property", "legal", "demographics", "tax", "zoning", "permits", "courts"],
        default=["property", "legal", "demographics"],
    )

    if st.button("Run", type="primary"):
        dry_run = mode == "Dry run (preview only)"
        if not categories:
            st.error("Select at least one category.")
        elif scope == "Single county (FIPS)" and (not fips or len(fips) != 5 or not fips.isdigit()):
            st.error("Enter a valid 5-digit FIPS code.")
        elif scope == "All counties in a state" and not state_abbr:
            st.error("Enter a state abbreviation.")
        else:
            try:
                from pipelines.ingest.crawl_orchestrator import CrawlOrchestrator
                from pipelines.ingest.county_registry import (
                    get_counties_by_state,
                    get_county_by_fips,
                    load_counties,
                )

                orch = CrawlOrchestrator(max_workers=10)
                cat_list = categories

                if dry_run:
                    if scope == "Single county (FIPS)":
                        c = get_county_by_fips(fips)
                        if c:
                            st.info(f"Dry run: would crawl 1 county — {c.county_name} ({c.fips})")
                        else:
                            st.warning(f"Unknown FIPS: {fips}")
                    else:
                        counties = get_counties_by_state(state_abbr)
                        st.info(f"Dry run: would crawl {len(counties)} counties in {state_abbr}")
                        with st.expander("County list"):
                            for c in counties[:50]:
                                st.text(f"  {c.fips}  {c.county_name}")
                            if len(counties) > 50:
                                st.text(f"  ... and {len(counties) - 50} more")
                    st.success("Dry run complete. No crawl was executed.")
                else:
                    progress = st.progress(0, text="Crawling…")
                    if scope == "Single county (FIPS)":
                        summary = orch.run_county(fips, cat_list)
                        progress.progress(1.0, text="Done")
                        st.metric("Documents crawled", summary.documents_crawled)
                        st.json({
                            "fips": summary.fips,
                            "county_name": summary.county_name,
                            "categories_succeeded": summary.categories_succeeded,
                            "categories_attempted": summary.categories_attempted,
                            "errors": summary.errors,
                        })
                        st.success("Single-county crawl finished.")
                    else:
                        summaries = orch.run_state(state_abbr, cat_list)
                        progress.progress(1.0, text="Done")
                        total_docs = sum(s.documents_crawled for s in summaries)
                        st.metric("Counties crawled", len(summaries))
                        st.metric("Total documents", total_docs)
                        with st.expander("Per-county results"):
                            for s in summaries[:30]:
                                st.text(f"{s.fips} {s.county_name}: {s.documents_crawled} docs, errors: {len(s.errors)}")
                            if len(summaries) > 30:
                                st.text(f"... and {len(summaries) - 30} more")
                        st.success("State crawl finished.")
            except Exception as e:
                st.exception(e)

# --- Crawl health page ---
elif page == "Crawl health":
    st.header("Crawl health")
    st.markdown("Summary from the latest crawl report and registry.")

    if st.button("Refresh health"):
        try:
            from monitoring.pipeline_monitor import PipelineMonitor
            monitor = PipelineMonitor()
            report = monitor.get_crawl_health()
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total counties", report.total_counties)
            col2.metric("Crawled", report.crawled_counties)
            col3.metric("Pending", report.pending_counties)
            col4.metric("Failed", report.failed_counties)
            st.metric("Total documents", report.total_documents)
            st.metric("Avg RAG confidence", round(report.avg_confidence, 4))
            st.text("Last crawl time: " + (report.last_crawl_time or "N/A"))
            st.success("Health report loaded.")
        except Exception as e:
            st.exception(e)
    else:
        st.info("Click **Refresh health** to load the latest crawl health.")

# --- RAG / Guardrails test page ---
elif page == "RAG / Guardrails test":
    st.header("RAG / Guardrails test")
    st.markdown("Run the guardrail pipeline on a synthetic response to verify confidence, bias, and legal checks.")

    confidence = st.slider("Confidence score", 0.0, 1.0, 0.7, 0.1)
    source_count = st.slider("Source count", 0, 10, 3)
    answer_text = st.text_area(
        "Sample answer text",
        value="The property value in this county is approximately $250,000. Verify with the assessor.",
        height=100,
    )
    fips = st.text_input("County FIPS", value="01001")

    if st.button("Run guardrails"):
        try:
            from rag.models import RAGResponse, SourceCitation, DocumentCategory
            from rag.guardrails import GuardrailPipeline
            from datetime import datetime

            sources = [
                SourceCitation(
                    document_id=f"doc_{i}",
                    county_fips=fips,
                    source_url=f"https://example.com/doc_{i}",
                    excerpt="Sample excerpt.",
                    relevance_score=confidence,
                    category=DocumentCategory.PROPERTY,
                    timestamp=datetime.now(),
                )
                for i in range(source_count)
            ]
            response = RAGResponse(
                answer=answer_text,
                sources=sources,
                confidence_score=confidence,
                county_fips=fips,
                hallucination_detected=False,
                category=DocumentCategory.PROPERTY,
            )
            pipeline = GuardrailPipeline(min_confidence=0.3, min_source_count=1)
            guarded = pipeline.apply(response)

            st.metric("Safe to serve", guarded.safe_to_serve)
            st.metric("Confidence sufficient", guarded.confidence_sufficient)
            st.metric("Bias detected", guarded.bias_report.bias_detected)
            st.metric("PII detected", guarded.compliance_result.pii_detected)
            st.text("Final answer (preview):")
            st.text(guarded.final_answer[:500] + ("…" if len(guarded.final_answer) > 500 else ""))
            if guarded.disclaimers:
                st.caption("Disclaimers: " + "; ".join(guarded.disclaimers))
            st.success("Guardrail run complete.")
        except Exception as e:
            st.exception(e)
