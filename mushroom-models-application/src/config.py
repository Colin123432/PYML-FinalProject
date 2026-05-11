from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
REPORTS_DIR = OUTPUT_DIR / "reports"
MODELS_DIR = OUTPUT_DIR / "models"

DEFAULT_DATA_PATH = DATA_DIR / "mushrooms.csv"
TARGET_COLUMN = "class"
POSITIVE_LABEL = "p"
NEGATIVE_LABEL = "e"
RANDOM_STATE = 42

for path in [FIGURES_DIR, METRICS_DIR, PREDICTIONS_DIR, REPORTS_DIR, MODELS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".mplconfig"))
