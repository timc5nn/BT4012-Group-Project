from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RUNS_DIR = DATA_DIR / "agent_runs"

THRESHOLDS_FILE = PROCESSED_DIR / "xgb_thresholds.json"
