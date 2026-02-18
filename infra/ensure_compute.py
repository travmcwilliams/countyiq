"""
Create or update Azure ML compute cluster from compute.yml.
Loads workspace config from .env via connect_workspace (python-dotenv).
Run from repo root: python -m infra.ensure_compute
"""

from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.entities import AmlCompute

# Uses dotenv via get_ml_client
from infra.connect_workspace import get_ml_client


def main() -> None:
    # DP-100: Compute target - Attaching AmlCompute cluster to workspace
    ml_client = get_ml_client()
    # DP-100: Cluster - Min/max nodes and VM size from infra/compute.yml
    compute = AmlCompute(
        name="cpu-cluster",
        size="Standard_DS3_v2",
        min_instances=0,
        max_instances=4,
        idle_time_before_scale_down=120,
    )
    ml_client.compute.begin_create_or_update(compute)
    print("Compute cluster cpu-cluster create/update started.")


if __name__ == "__main__":
    main()
