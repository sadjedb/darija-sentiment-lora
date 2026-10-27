"""Project-wide constants: paths, dataset, labels and the base model."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
RESULTS_DIR = ROOT / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
OUTPUTS_DIR = ROOT / "outputs"
ADAPTER_DIR = OUTPUTS_DIR / "qwen3-4b-darija-sentiment-lora"
MODEL_CARD_DIR = ROOT / "model_card"

# Benmounah et al. (2023), 50k Algerian YouTube comments, MIT licence.
DATASET_REPO = "Abdou/dz-sentiment-yt-comments"
DATASET_FILE = "ADArabic-3labels-50016.csv"

LABELS = ["negative", "neutral", "positive"]  # index == dataset label id
LABEL2ID = {name: i for i, name in enumerate(LABELS)}
ID2LABEL = dict(enumerate(LABELS))

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

SEED = 42
