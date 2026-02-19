"""
Document classifier for CountyIQ.
Automatically categorizes crawled documents into DocumentCategory using scikit-learn.
# DP-100: Supervised learning - Text classification using TF-IDF and LogisticRegression.
"""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from data.schemas.document import CountyDocument, DocumentCategory

# DP-100: Model persistence - Saving and loading trained models
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "classifier"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "classifier_model.joblib"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.joblib"


class ClassifierMetrics(BaseModel):
    """Metrics from classifier evaluation."""

    accuracy: float = Field(..., ge=0.0, le=1.0, description="Overall accuracy")
    precision: float = Field(..., ge=0.0, le=1.0, description="Macro-averaged precision")
    recall: float = Field(..., ge=0.0, le=1.0, description="Macro-averaged recall")
    f1: float = Field(..., ge=0.0, le=1.0, description="Macro-averaged F1 score")
    confusion_matrix: list[list[int]] = Field(..., description="Confusion matrix (categories x categories)")
    category_scores: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="Per-category precision, recall, f1",
    )


# DP-100: Supervised learning - Text classification pipeline
class DocumentClassifier:
    """
    Classifies county documents into DocumentCategory using TF-IDF + LogisticRegression.
    Saves/loads model artifacts using joblib.
    """

    def __init__(self, max_features: int = 5000, random_state: int = 42) -> None:
        """
        Initialize classifier.

        Args:
            max_features: Maximum TF-IDF features (default: 5000).
            random_state: Random seed for reproducibility (default: 42).
        """
        self.max_features = max_features
        self.random_state = random_state
        self.vectorizer: TfidfVectorizer | None = None
        self.model: LogisticRegression | None = None
        self.category_labels: list[str] | None = None

        # DP-100: Model loading - Loading saved model artifacts
        self._load_model_if_exists()

    def _load_model_if_exists(self) -> None:
        """Load saved model and vectorizer if they exist."""
        if MODEL_PATH.exists() and VECTORIZER_PATH.exists():
            try:
                self.model = joblib.load(MODEL_PATH)
                self.vectorizer = joblib.load(VECTORIZER_PATH)
                # Load category labels from metadata if available
                metadata_path = MODEL_DIR / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, encoding="utf-8") as f:
                        metadata = json.load(f)
                        self.category_labels = metadata.get("category_labels", None)
                logger.info("Loaded saved classifier model from {}", MODEL_PATH)
            except Exception as e:
                logger.warning("Failed to load saved model: {}", e)
                self.model = None
                self.vectorizer = None

    def _get_text(self, doc: CountyDocument) -> str:
        """Extract text from document (prefer processed_content, fallback to raw_content)."""
        if doc.processed_content and doc.processed_content.strip():
            return doc.processed_content.strip()
        return (doc.raw_content or "").strip()

    def train(self, documents: list[CountyDocument]) -> None:
        """
        Train classifier on labeled documents.

        Args:
            documents: List of CountyDocument instances with category labels.
        """
        if not documents:
            raise ValueError("Cannot train on empty document list")

        # Extract texts and labels
        texts: list[str] = []
        labels: list[str] = []

        for doc in documents:
            text = self._get_text(doc)
            if text and len(text) >= 10:  # Minimum text length
                texts.append(text)
                # Handle both enum and string (CountyDocument uses use_enum_values=True)
                category_val = doc.category.value if hasattr(doc.category, "value") else str(doc.category)
                labels.append(category_val)

        if not texts:
            raise ValueError("No valid text content found in documents")

        logger.info("Training classifier on {} documents", len(texts))

        # DP-100: Feature engineering - TF-IDF vectorization for text classification
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words="english",
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=2,  # Minimum document frequency
            max_df=0.95,  # Maximum document frequency
        )

        X = self.vectorizer.fit_transform(texts)
        y = np.array(labels)

        # Store category labels for prediction
        self.category_labels = sorted(set(labels))

        # DP-100: Model training - LogisticRegression for multi-class classification
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
            solver="lbfgs",
        )
        self.model.fit(X, y)

        logger.success("Classifier trained on {} samples, {} features", X.shape[0], X.shape[1])

        # Save model artifacts
        self._save_model()

    def _save_model(self) -> None:
        """Save model, vectorizer, and metadata to disk."""
        if self.model is None or self.vectorizer is None:
            return

        try:
            joblib.dump(self.model, MODEL_PATH)
            joblib.dump(self.vectorizer, VECTORIZER_PATH)

            # Save metadata
            metadata = {
                "category_labels": self.category_labels,
                "max_features": self.max_features,
                "random_state": self.random_state,
            }
            metadata_path = MODEL_DIR / "metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            logger.info("Saved classifier model to {}", MODEL_PATH)
        except Exception as e:
            logger.error("Failed to save model: {}", e)

    def predict(self, text: str) -> DocumentCategory:
        """
        Predict category for a single text.

        Args:
            text: Text content to classify.

        Returns:
            Predicted DocumentCategory.
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained. Call train() first.")

        if not text or not text.strip():
            logger.warning("Empty text provided, returning user_upload as default")
            return DocumentCategory.user_upload

        # DP-100: Model inference - Single prediction
        X = self.vectorizer.transform([text.strip()])
        predicted_label = self.model.predict(X)[0]

        try:
            return DocumentCategory(predicted_label)
        except (ValueError, TypeError):
            logger.warning("Unknown predicted label: {}, returning user_upload", predicted_label)
            return DocumentCategory.user_upload

    def predict_batch(self, texts: list[str]) -> list[DocumentCategory]:
        """
        Predict categories for multiple texts.

        Args:
            texts: List of text contents to classify.

        Returns:
            List of predicted DocumentCategory instances.
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained. Call train() first.")

        if not texts:
            return []

        # Filter empty texts
        valid_texts = [t.strip() if t else "" for t in texts]
        valid_indices = [i for i, t in enumerate(valid_texts) if t]

        if not valid_indices:
            return [DocumentCategory.user_upload] * len(texts)

        # DP-100: Batch inference - Predicting multiple samples efficiently
        X = self.vectorizer.transform([valid_texts[i] for i in valid_indices])
        predicted_labels = self.model.predict(X)

        results: list[DocumentCategory] = []
        valid_idx = 0
        for i in range(len(texts)):
            if i in valid_indices:
                try:
                    results.append(DocumentCategory(predicted_labels[valid_idx]))
                except (ValueError, TypeError):
                    results.append(DocumentCategory.user_upload)
                valid_idx += 1
            else:
                results.append(DocumentCategory.user_upload)

        return results

    def evaluate(self, test_docs: list[CountyDocument]) -> ClassifierMetrics:
        """
        Evaluate classifier on test documents.

        Args:
            test_docs: List of CountyDocument instances with true category labels.

        Returns:
            ClassifierMetrics with accuracy, precision, recall, f1, confusion matrix, category scores.
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained. Call train() first.")

        if not test_docs:
            raise ValueError("Cannot evaluate on empty document list")

        # Extract texts and true labels
        texts: list[str] = []
        true_labels: list[str] = []

        for doc in test_docs:
            text = self._get_text(doc)
            if text and len(text) >= 10:
                texts.append(text)
                # Handle both enum and string (CountyDocument uses use_enum_values=True)
                category_val = doc.category.value if hasattr(doc.category, "value") else str(doc.category)
                true_labels.append(category_val)

        if not texts:
            raise ValueError("No valid text content found in test documents")

        # Predict
        X = self.vectorizer.transform(texts)
        predicted_labels = self.model.predict(X)

        # DP-100: Model evaluation - Computing classification metrics
        accuracy = float(accuracy_score(true_labels, predicted_labels))
        precision = float(precision_score(true_labels, predicted_labels, average="macro", zero_division=0))
        recall = float(recall_score(true_labels, predicted_labels, average="macro", zero_division=0))
        f1 = float(f1_score(true_labels, predicted_labels, average="macro", zero_division=0))

        # Confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels, labels=self.category_labels)
        cm_list = cm.tolist()

        # Per-category scores
        report = classification_report(
            true_labels,
            predicted_labels,
            labels=self.category_labels,
            output_dict=True,
            zero_division=0,
        )

        category_scores: dict[str, dict[str, float]] = {}
        for label in self.category_labels or []:
            if label in report:
                category_scores[label] = {
                    "precision": float(report[label].get("precision", 0.0)),
                    "recall": float(report[label].get("recall", 0.0)),
                    "f1": float(report[label].get("f1-score", 0.0)),
                }

        logger.info(
            "Evaluation complete: accuracy={:.3f}, f1={:.3f}, precision={:.3f}, recall={:.3f}",
            accuracy,
            f1,
            precision,
            recall,
        )

        return ClassifierMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            confusion_matrix=cm_list,
            category_scores=category_scores,
        )
