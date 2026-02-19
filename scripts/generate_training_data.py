"""
Generate synthetic labeled training data for document classifier.
Creates 50 examples per category with realistic keyword patterns.
"""

import json
import random
from pathlib import Path

from loguru import logger

# Category-specific keyword patterns
CATEGORY_PATTERNS: dict[str, list[str]] = {
    "property": [
        "parcel",
        "assessed value",
        "owner",
        "acreage",
        "deed",
        "property tax",
        "lot size",
        "zoning",
        "appraised value",
        "land use",
    ],
    "legal": [
        "plaintiff",
        "defendant",
        "case number",
        "judgment",
        "court",
        "lawsuit",
        "filing",
        "motion",
        "settlement",
        "attorney",
    ],
    "demographics": [
        "population",
        "census",
        "median income",
        "household",
        "demographics",
        "age distribution",
        "education level",
        "employment rate",
        "housing units",
        "per capita income",
    ],
    "permits": [
        "permit number",
        "contractor",
        "inspection",
        "approved",
        "building permit",
        "electrical permit",
        "plumbing permit",
        "construction",
        "code compliance",
        "permit fee",
    ],
    "zoning": [
        "zone",
        "land use",
        "variance",
        "setback",
        "ordinance",
        "zoning code",
        "zoning district",
        "conditional use",
        "zoning board",
        "zoning map",
    ],
    "courts": [
        "docket",
        "filing",
        "motion",
        "hearing",
        "verdict",
        "court case",
        "judge",
        "courtroom",
        "trial",
        "court order",
    ],
    "tax": [
        "tax rate",
        "levy",
        "millage",
        "exemption",
        "delinquent",
        "property tax",
        "tax assessment",
        "tax bill",
        "tax collector",
        "tax lien",
    ],
}

# Template sentences for generating realistic text
TEMPLATES: list[str] = [
    "The {keyword} for this {entity} is {value}.",
    "This document contains information about {keyword} and {keyword2}.",
    "The {entity} has a {keyword} of {value}.",
    "Please review the {keyword} details for {entity}.",
    "The {keyword} was {action} on {date}.",
    "This record shows {keyword} information for the {entity}.",
    "The {keyword} indicates that {description}.",
    "For questions about {keyword}, contact {contact}.",
    "The {entity} {keyword} has been {status}.",
    "This {keyword} document is related to {related}.",
]


def generate_document_text(category: str, num_sentences: int = 3) -> str:
    """
    Generate synthetic document text for a category.

    Args:
        category: Document category.
        num_sentences: Number of sentences to generate.

    Returns:
        Generated text string.
    """
    patterns = CATEGORY_PATTERNS.get(category, [])
    if not patterns:
        return f"This is a {category} document."

    sentences: list[str] = []
    for _ in range(num_sentences):
        template = random.choice(TEMPLATES)
        keywords = random.sample(patterns, min(2, len(patterns)))
        entity = random.choice(["property", "record", "document", "file", "case"])
        value = random.choice(["$100,000", "2024", "approved", "pending", "completed"])
        action = random.choice(["filed", "submitted", "processed", "reviewed"])
        date = random.choice(["2024-01-15", "2024-02-20", "2024-03-10"])
        description = f"the {keywords[0]} is {value}"
        contact = random.choice(["county office", "clerk", "administrator"])
        status = random.choice(["approved", "pending", "completed", "filed"])
        related = random.choice(["property records", "legal matters", "tax information"])

        sentence = (
            template.replace("{keyword}", keywords[0])
            .replace("{keyword2}", keywords[1] if len(keywords) > 1 else keywords[0])
            .replace("{entity}", entity)
            .replace("{value}", value)
            .replace("{action}", action)
            .replace("{date}", date)
            .replace("{description}", description)
            .replace("{contact}", contact)
            .replace("{status}", status)
            .replace("{related}", related)
        )
        sentences.append(sentence)

    # Add some category-specific keywords naturally
    additional_keywords = random.sample(patterns, min(3, len(patterns)))
    sentences.extend([f"The {kw} is documented in this record." for kw in additional_keywords[:2]])

    return " ".join(sentences)


def generate(
    output_path: str | Path | None = None,
    examples_per_category: int = 50,
    categories: list[str] | None = None,
) -> list[dict[str, str]]:
    """
    Generate synthetic training data.

    Args:
        output_path: Path to save JSON file (default: data/processed/training_data.json).
        examples_per_category: Number of examples per category (default: 50).
        categories: List of categories to generate (default: all 7 categories).

    Returns:
        List of training records with text, category, source fields.
    """
    if categories is None:
        categories = list(CATEGORY_PATTERNS.keys())

    if output_path is None:
        output_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "training_data.json"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, str]] = []

    logger.info("Generating {} examples per category for {} categories", examples_per_category, len(categories))

    for category in categories:
        for i in range(examples_per_category):
            text = generate_document_text(category, num_sentences=random.randint(2, 5))
            records.append(
                {
                    "text": text,
                    "category": category,
                    "source": "synthetic",
                }
            )

    # Shuffle records
    random.shuffle(records)

    # Save to JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    logger.success("Generated {} training examples, saved to {}", len(records), output_path)

    return records


if __name__ == "__main__":
    generate()
