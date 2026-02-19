"""
System prompts for RAG pipeline.
Instructs Claude to answer only from provided context and cite sources.
"""

from rag.retrieval.retriever import RetrievedDocument

# DP-100: Prompt engineering - System prompt for grounded generation
SYSTEM_PROMPT = """You are a helpful assistant that answers questions about US county data using only the provided context documents.

CRITICAL RULES:
1. Answer ONLY from the provided context documents. Never use external knowledge.
2. Always cite source documents by URL and county name.
3. If the context doesn't contain enough information, say "I don't have that information in the provided documents."
4. Never speculate or make up information beyond what's in the documents.
5. Format your response as: answer first, then a "Sources:" section listing all cited documents.

Your goal is to provide accurate, traceable answers grounded in the county data provided."""


def build_context(documents: list[RetrievedDocument]) -> str:
    """
    Format retrieved documents into prompt context.

    Args:
        documents: List of RetrievedDocument instances.

    Returns:
        Formatted context string.
    """
    if not documents:
        return "No documents provided."

    context_parts: list[str] = []
    context_parts.append("CONTEXT DOCUMENTS:\n")

    for i, doc in enumerate(documents, 1):
        context_parts.append(f"--- Document {i} ---")
        context_parts.append(f"County: {doc.county_name}, {doc.fips}")
        context_parts.append(f"Category: {doc.category}")
        if doc.source_url:
            context_parts.append(f"Source: {doc.source_url}")
        context_parts.append(f"Content:\n{doc.content}\n")

    return "\n".join(context_parts)


def build_query_prompt(query: str, context: str, fips: str | None = None) -> str:
    """
    Build full user prompt with query and context.

    Args:
        query: User query.
        context: Formatted context from retrieved documents.
        fips: Optional county FIPS (for context).

    Returns:
        Full prompt string.
    """
    prompt_parts: list[str] = []

    if fips:
        prompt_parts.append(f"Question about county FIPS {fips}:\n")

    prompt_parts.append(f"QUESTION: {query}\n\n")
    prompt_parts.append(context)
    prompt_parts.append("\n\nPlease answer the question using only the context documents above. Cite your sources.")

    return "\n".join(prompt_parts)
