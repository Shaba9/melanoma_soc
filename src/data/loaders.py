"""Metadata loading, label/skin-tone mapping, lesion-safe splitting, and DataLoader
factories per DATA-SPEC.md.
"""
import logging
from functools import lru_cache

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .. import config
from .datasets import DDIDataset, HAM10000Dataset

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _ham_image_index() -> dict:
    """Map HAM image_id (file stem) -> absolute image path across both part folders."""
    index = {}
    for folder in config.HAM_IMAGE_DIRS:
        if not folder.exists():
            continue
        for path in folder.glob("*.jpg"):
            index[path.stem] = path
    return index


def load_ham_metadata() -> pd.DataFrame:
    """Load HAM10000 metadata with binary label and resolved image paths."""
    df = pd.read_csv(config.HAM_METADATA)
    df["label"] = (df["dx"] == "mel").astype(int)
    index = _ham_image_index()
    df["image_path"] = df["image_id"].map(index)
    missing = df["image_path"].isna()
    if missing.any():
        logger.warning("HAM10000: skipping %d rows with missing images", int(missing.sum()))
        df = df[~missing]
    return df.reset_index(drop=True)


def load_ddi_metadata() -> pd.DataFrame:
    """Load DDI metadata with binary label, skin-tone group, and image paths."""
    df = pd.read_csv(config.DDI_METADATA, index_col=0)
    df["label"] = df["malignant"].astype(str).str.lower().eq("true").astype(int)
    df["skin_tone"] = df["skin_tone"].astype(int)
    df["skin_tone_group"] = df["skin_tone"].map(config.SKIN_TONE_MAP)
    df["image_path"] = df["DDI_file"].map(lambda f: config.DDI_IMAGE_DIR / f)
    exists = df["image_path"].map(lambda p: p.exists())
    if not exists.all():
        logger.warning("DDI: skipping %d rows with missing images", int((~exists).sum()))
        df = df[exists]
    return df.reset_index(drop=True)


def split_ham(df: pd.DataFrame, val_split: float = config.VAL_SPLIT,
              seed: int = config.SEED):
    """Stratified 80/20 split by lesion_id (no lesion leakage across splits)."""
    lesion = df.groupby("lesion_id")["label"].max().reset_index()
    train_ids, val_ids = train_test_split(
        lesion["lesion_id"],
        test_size=val_split,
        stratify=lesion["label"],
        random_state=seed,
    )
    train_ids, val_ids = set(train_ids), set(val_ids)
    train_df = df[df["lesion_id"].isin(train_ids)].reset_index(drop=True)
    val_df = df[df["lesion_id"].isin(val_ids)].reset_index(drop=True)
    return train_df, val_df


def compute_class_weights(df: pd.DataFrame) -> torch.Tensor:
    """Inverse-frequency class weights for [Benign, Malignant]."""
    counts = df["label"].value_counts().reindex([config.BENIGN, config.MALIGNANT]).fillna(0)
    total = counts.sum()
    weights = total / (len(counts) * counts.clip(lower=1))
    return torch.tensor(weights.values, dtype=torch.float32)


def build_ham_loaders(train_transform, val_transform,
                      batch_size: int = config.BATCH_SIZE):
    """Return (train_loader, val_loader) for HAM10000."""
    df = load_ham_metadata()
    train_df, val_df = split_ham(df)
    train_loader = DataLoader(
        HAM10000Dataset(train_df, train_transform),
        batch_size=batch_size, shuffle=True, num_workers=config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        HAM10000Dataset(val_df, val_transform),
        batch_size=batch_size, shuffle=False, num_workers=config.NUM_WORKERS,
    )
    return train_loader, val_loader


def build_ddi_loader(transform, batch_size: int = config.BATCH_SIZE):
    """Return a DataLoader over the full DDI evaluation set."""
    df = load_ddi_metadata()
    return DataLoader(
        DDIDataset(df, transform),
        batch_size=batch_size, shuffle=False, num_workers=config.NUM_WORKERS,
    )
