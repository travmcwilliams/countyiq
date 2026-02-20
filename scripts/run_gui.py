"""
Launch the CountyIQ testing GUI.

Run from project root:
  python scripts/run_gui.py

Or directly with Streamlit:
  streamlit run gui/app.py
"""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
app_path = PROJECT_ROOT / "gui" / "app.py"

if not app_path.exists():
    print(f"GUI app not found: {app_path}")
    sys.exit(1)

# Use same Python as script so venv is used; no need for "streamlit" on PATH
result = subprocess.run(
    [sys.executable, "-m", "streamlit", "run", str(app_path)],
    cwd=str(PROJECT_ROOT),
)
sys.exit(result.returncode)
