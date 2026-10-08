"""PyTorch datasets for HAM10000 and DDI per DATA-SPEC.md.

Each dataset yields ``(image_tensor, label, metadata_dict)``.
Dataframes are prepared by ``loaders.py`` (label mapping, image paths, skin-tone group).
"""
from typing import Optional, Callable

from PIL import Image
from torch.utils.data import Dataset


class LesionDataset(Dataset):
    """Base dataset over pre-built records: ``{path, label, meta}``."""

    def __init__(self, records: list, transform: Optional[Callable] = None):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        image = Image.open(rec["path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, rec["label"], rec["meta"]


class HAM10000Dataset(LesionDataset):
    """HAM10000 training dataset. Expects columns: image_id, image_path, label."""

    def __init__(self, df, transform: Optional[Callable] = None):
        records = [
            {
                "path": row["image_path"],
                "label": int(row["label"]),
                "meta": {"image_id": row["image_id"], "dataset": "HAM10000"},
            }
            for _, row in df.iterrows()
        ]
        super().__init__(records, transform)


class DDIDataset(LesionDataset):
    """DDI evaluation dataset. Expects columns: DDI_ID, image_path, label,
    skin_tone, skin_tone_group."""

    def __init__(self, df, transform: Optional[Callable] = None):
        records = [
            {
                "path": row["image_path"],
                "label": int(row["label"]),
                "meta": {
                    "ddi_id": int(row["DDI_ID"]),
                    "dataset": "DDI",
                    "skin_tone": int(row["skin_tone"]),
                    "skin_tone_group": row["skin_tone_group"],
                },
            }
            for _, row in df.iterrows()
        ]
        super().__init__(records, transform)
