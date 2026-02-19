"""
Azure ML workspace connection utilities.
Provides MLClient for Azure ML SDK v2 operations.
# DP-100: ML Client - Connecting to Azure ML workspace for job submission and model registration.
"""

import os
from typing import Any

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from loguru import logger

# Azure configuration from environment or defaults
SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID", "b094e7ca-556e-47b0-b604-c6a435139129")
RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP", "countyiq-rg")
WORKSPACE_NAME = os.getenv("AZURE_ML_WORKSPACE_NAME", "countyiq-workspace")

_ml_client: MLClient | None = None


def get_ml_client() -> MLClient:
    """
    Get or create Azure ML Client (singleton).

    Returns:
        MLClient instance connected to Azure ML workspace.
    """
    global _ml_client

    if _ml_client is None:
        try:
            # DP-100: Authentication - Using DefaultAzureCredential for Azure ML access
            credential = DefaultAzureCredential()

            # DP-100: ML Client - Initializing connection to Azure ML workspace
            _ml_client = MLClient(
                credential=credential,
                subscription_id=SUBSCRIPTION_ID,
                resource_group_name=RESOURCE_GROUP,
                workspace_name=WORKSPACE_NAME,
            )

            logger.info(
                "Connected to Azure ML workspace: {} (subscription: {}, resource group: {})",
                WORKSPACE_NAME,
                SUBSCRIPTION_ID,
                RESOURCE_GROUP,
            )
        except Exception as e:
            logger.error("Failed to connect to Azure ML workspace: {}", e)
            raise

    return _ml_client
