"""
Azure ML experiment for document classifier training.
Tracks training in MLflow, registers model in Azure ML Model Registry.
# DP-100: Experiment tracking - MLflow integration with Azure ML for model training and registration.
"""

import json
from pathlib import Path
from typing import Any

import joblib
import mlflow
from loguru import logger

from data.schemas.document import CountyDocument
from infra.connect_workspace import get_ml_client
from pipelines.transform.document_classifier import ClassifierMetrics, DocumentClassifier

# DP-100: Model Registry - Registering models with versioning and metadata
MODEL_NAME = "county-doc-classifier"


def run_classifier_experiment(
    training_docs: list[CountyDocument],
    test_docs: list[CountyDocument] | None = None,
    experiment_name: str = "document-classifier",
    run_name: str | None = None,
) -> dict[str, Any]:
    """
    Run classifier training experiment with MLflow tracking and model registration.

    Args:
        training_docs: Training documents with category labels.
        test_docs: Optional test documents for evaluation (if None, uses training_docs split).
        experiment_name: MLflow experiment name (default: "document-classifier").
        run_name: Optional run name (default: auto-generated).

    Returns:
        Dictionary with run_id, metrics, and model registration info.
    """
    # DP-100: MLflow Tracking - Setting experiment context
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        logger.info("Starting classifier training experiment: {}", run.info.run_id)

        # Train classifier
        classifier = DocumentClassifier()
        classifier.train(training_docs)

        # Evaluate
        if test_docs:
            metrics = classifier.evaluate(test_docs)
        else:
            # Use training docs for evaluation if no test set provided
            metrics = classifier.evaluate(training_docs)

        # DP-100: Metrics logging - Logging training metrics to MLflow
        mlflow.log_metric("accuracy", metrics.accuracy)
        mlflow.log_metric("precision", metrics.precision)
        mlflow.log_metric("recall", metrics.recall)
        mlflow.log_metric("f1", metrics.f1)
        mlflow.log_metric("training_doc_count", len(training_docs))

        # Log per-category metrics
        for category, scores in metrics.category_scores.items():
            mlflow.log_metric(f"precision_{category}", scores["precision"])
            mlflow.log_metric(f"recall_{category}", scores["recall"])
            mlflow.log_metric(f"f1_{category}", scores["f1"])

        # Log parameters
        mlflow.log_param("max_features", classifier.max_features)
        mlflow.log_param("random_state", classifier.random_state)
        mlflow.log_param("num_categories", len(metrics.category_scores))

        # Log confusion matrix as artifact
        confusion_matrix_path = Path("confusion_matrix.json")
        with open(confusion_matrix_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "confusion_matrix": metrics.confusion_matrix,
                    "category_labels": list(metrics.category_scores.keys()),
                },
                f,
                indent=2,
            )
        mlflow.log_artifact(str(confusion_matrix_path))

        # DP-100: Model logging - Logging model artifact to MLflow
        if classifier.model is not None and classifier.vectorizer is not None:
            # Log sklearn model
            mlflow.sklearn.log_model(
                classifier.model,
                "model",
                registered_model_name=MODEL_NAME,
            )

            # Log vectorizer separately (as artifact)
            vectorizer_path = Path("vectorizer.joblib")
            joblib.dump(classifier.vectorizer, vectorizer_path)
            mlflow.log_artifact(str(vectorizer_path))

        logger.success(
            "Experiment complete: run_id={}, accuracy={:.3f}, f1={:.3f}",
            run.info.run_id,
            metrics.accuracy,
            metrics.f1,
        )

        # DP-100: Model Registration - Registering model in Azure ML Model Registry
        try:
            ml_client = get_ml_client()
            # Model is already registered via mlflow.sklearn.log_model above
            # But we can also explicitly register it for additional metadata
            from azure.ai.ml.entities import Model
            from azure.ai.ml.constants import AssetTypes

            model = Model(
                path=f"runs:/{run.info.run_id}/model",
                name=MODEL_NAME,
                description="Document classifier for CountyIQ - categorizes documents into property, legal, demographics, etc.",
                type=AssetTypes.MLFLOW_MODEL,
                tags={
                    "task": "text_classification",
                    "model_type": "logistic_regression",
                    "feature_type": "tfidf",
                    "num_categories": str(len(metrics.category_scores)),
                },
            )

            registered_model = ml_client.models.create_or_update(model)
            logger.info("Registered model: {} version {}", registered_model.name, registered_model.version)

            return {
                "run_id": run.info.run_id,
                "experiment_name": experiment_name,
                "metrics": {
                    "accuracy": metrics.accuracy,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1": metrics.f1,
                },
                "model_name": MODEL_NAME,
                "model_version": registered_model.version,
            }
        except Exception as e:
            logger.warning("Model registration failed (model may already be registered via MLflow): {}", e)
            return {
                "run_id": run.info.run_id,
                "experiment_name": experiment_name,
                "metrics": {
                    "accuracy": metrics.accuracy,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1": metrics.f1,
                },
                "model_name": MODEL_NAME,
                "model_version": None,
            }
