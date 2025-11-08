"""Constants and configuration for hard-negative mining."""
from pathlib import Path

# Directory paths
DEFAULT_INPUT_DIR = Path("scripts") / "input"
DEFAULT_OUTPUT_ROOT = Path("scripts") / "curated"
EXPERIMENTS_ROOT = Path("experiments")
RESULTS_DIR_NAME = "low_quality_inference"

# Model configuration
MODEL_EXTENSIONS = ('.tflite', '.h5', '.pt')
MODEL_SOURCE_OPTIONS = [
    "Use experiment (canonical analyzer args)",
    "Use a model file from an experiment",
    "Upload a .tflite file"
]

# UI configuration
UI_UPDATE_INTERVAL = 0.1  # seconds between UI updates during streaming
MAX_PREVIEW_FILES = 8
MAX_LOG_LINES_DISPLAYED = 100
MAX_LOG_LINES_FINAL = 200
MAX_INDEX_PICKER_ROWS = 200

# File operations
LINK_METHODS = ["copy", "hardlink", "symlink"]
LINK_METHOD_LABELS = {
    "copy": "Copy",
    "hardlink": "Hard link",
    "symlink": "Symlink"
}

# CSV configuration
CONFIDENCE_COLUMN_NAME = "radr_max_confidence"
FILE_COLUMN_NAME = "File"
REQUIRED_COLUMNS = [FILE_COLUMN_NAME, CONFIDENCE_COLUMN_NAME]

# Selection modes
SELECTION_MODES = ["top_k", "top_pct", "by_index", "threshold", "manual"]
SELECTION_MODE_LABELS = {
    "top_k": "Top K",
    "top_pct": "Top %",
    "by_index": "By index list",
    "threshold": "By confidence threshold",
    "manual": "Manual select"
}

# Default values
DEFAULT_TOP_K = 50
DEFAULT_TOP_PERCENTILE = 10
DEFAULT_THRESHOLD = 0.5
DEFAULT_OUTPUT_LABEL = "hardneg_manual"
DEFAULT_ZIP_NAME = "hardneg_selection.zip"
