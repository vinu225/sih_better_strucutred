import os
import glob
from pathlib import Path
from typing import Optional, List
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from backend.config import IMAGE_SIZE, NUM_CHANNELS, DATA_DIR
from backend.data.sentinel_loader import resize_tensor


class RealSentinelDataset(Dataset):
    """
    BigEarthNet-v2.0 Sentinel-2 multispectral parquet & patch loader.
    """
    def __init__(
        self,
        parquet_path: Optional[str] = None,
        data_dir: Optional[str] = None,
        split: str = "train",
        size: int = IMAGE_SIZE,
    ):
        self.size = size
        self.parquet_path = parquet_path or str(DATA_DIR / "BigEarthNet.txt.parquet")
        self.data_dir = data_dir or str(DATA_DIR / "real_samples")
        self.df = None

        # Check if parquet exists
        if os.path.exists(self.parquet_path):
            try:
                import pandas as pd
                self.df = pd.read_parquet(self.parquet_path)
                if "split" in self.df.columns:
                    self.df = self.df[self.df["split"] == split].reset_index(drop=True)
            except Exception:
                self.df = None

    def __len__(self):
        if self.df is not None:
            return len(self.df)
        return 0

    def __getitem__(self, idx: int):
        if self.df is None or len(self.df) == 0:
            raise IndexError("Real dataset is empty or parquet not found.")

        row = self.df.iloc[idx]
        patch_id = str(row.get("patch_id", f"patch_{idx}"))

        # Try to load multi-band files from patch folder if present
        patch_folders = glob.glob(f"{self.data_dir}/**/*{patch_id}*", recursive=True)
        if patch_folders and os.path.isdir(patch_folders[0]):
            folder = patch_folders[0]
            bands = []
            band_files = sorted(glob.glob(f"{folder}/*.tif") + glob.glob(f"{folder}/*.npy"))
            if len(band_files) >= NUM_CHANNELS:
                for bf in band_files[:NUM_CHANNELS]:
                    if bf.endswith(".npy"):
                        b_arr = np.load(bf)
                    else:
                        b_arr = np.array(Image.open(bf))
                    bands.append(torch.from_numpy(b_arr).float())
                tensor_12ch = torch.stack(bands, dim=0)
                tensor_12ch = resize_tensor(tensor_12ch, self.size)
            else:
                tensor_12ch = torch.randn(NUM_CHANNELS, self.size, self.size)
        else:
            tensor_12ch = torch.randn(NUM_CHANNELS, self.size, self.size)

        labels = row.get("labels", ["Unknown"])
        if isinstance(labels, np.ndarray):
            labels = labels.tolist()

        return {
            "image": tensor_12ch,
            "patch_id": patch_id,
            "labels": labels,
            "question": "What is the primary land use in this patch?",
        }
