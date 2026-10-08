from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "datasets"
ARTIFACT_DIR = ROOT / "artifacts"

# Artifact subdirectories
MODELS_DIR = ARTIFACT_DIR / "models"
METRICS_DIR = ARTIFACT_DIR / "metrics"
FIGURES_DIR = ARTIFACT_DIR / "figures"
TABLES_DIR = ARTIFACT_DIR / "tables"
LOGS_DIR = ARTIFACT_DIR / "logs"

# HAM10000 (training)
HAM_DIR = DATA_DIR / "HAM10000"
HAM_METADATA = HAM_DIR / "HAM10000_metadata.csv"
HAM_IMAGE_DIRS = [
    HAM_DIR / "images" / "HAM10000_images_part_1",
    HAM_DIR / "images" / "HAM10000_images_part_2",
]

# DDI (evaluation)
DDI_DIR = DATA_DIR / "DDI"
DDI_METADATA = DDI_DIR / "ddi_metadata.csv"
DDI_IMAGE_DIR = DDI_DIR / "images"

# Labels
BENIGN, MALIGNANT = 0, 1
CLASS_NAMES = ["Benign", "Malignant"]

# DDI skin-tone code -> group
SKIN_TONE_MAP = {12: "Light", 34: "Medium", 56: "Dark"}

# Preprocessing
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Training / loaders
SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 15
VAL_SPLIT = 0.2
NUM_WORKERS = 0
