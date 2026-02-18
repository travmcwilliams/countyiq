"""
Azure ML workspace connection using SDK v2.
Loads configuration from environment variables via python-dotenv.
"""

from pathlib import Path

from dotenv import load_dotenv
import os

# Load .env from project root (parent of infra/)
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_env_path)

# DP-100: Workspace connection - MLClient is the SDK v2 entry point for workspace
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# Required env vars (set in .env from .env.example)
SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID", "b094e7ca-556e-47b0-b604-c6a435139129")
TENANT_ID = os.getenv("AZURE_TENANT_ID", "8ad254d8-6edb-4086-8512-380bf17d8aed")
WORKSPACE_NAME = os.getenv("AZURE_ML_WORKSPACE_NAME", "")
RESOURCE_GROUP = os.getenv("AZURE_RESOURCE_GROUP", "")


def get_ml_client() -> MLClient:
    """
    Return an MLClient connected to the Azure ML workspace.
    Uses DefaultAzureCredential (CLI, managed identity, or env auth).
    """
    if not WORKSPACE_NAME or not RESOURCE_GROUP:
        raise ValueError(
            "Set AZURE_ML_WORKSPACE_NAME and AZURE_RESOURCE_GROUP in .env"
        )
    # DP-100: Authentication - DefaultAzureCredential supports CLI, MI, service principal
    credential = DefaultAzureCredential()
    # DP-100: ML Client - Connects to workspace for jobs, models, data, compute
    return MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME,
    )
