from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
DATA_DIR=ROOT/"data"
ARTIFACT_DIR=ROOT/"artifacts"
SEED=42
IMAGE_SIZE=224
BATCH_SIZE=32
LEARNING_RATE=1e-4
EPOCHS=20
