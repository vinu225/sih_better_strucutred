import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SAMPLES_DIR = DATA_DIR / "samples"
os.makedirs(SAMPLES_DIR, exist_ok=True)

# Hardware Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Satellite Vision Backbone Configuration
MODEL_CHECKPOINT = "BIFOLD-BigEarthNetv2-0/mobilevit_s-all-v0.1.1"
IMAGE_SIZE = 120  # Standard input resolution for BigEarthNetv2.0 MobileViT models
NUM_CHANNELS = 12
VISION_FEATURE_DIM = 640

# Language Model Selection
LLM_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
PROJECTOR_HIDDEN_DIM = 1024
DEFAULT_MAX_NEW_TOKENS = 64

# BigEarthNet-v2.0 Multimodal 12-Channel Specification (v0.1.1):
# 10 Sentinel-2 multispectral optical bands + 2 Sentinel-1 SAR radar channels
BAND_NAMES = [
    "B02",  # Channel 0: Blue (490 nm) - 10m [Sentinel-2]
    "B03",  # Channel 1: Green (560 nm) - 10m [Sentinel-2]
    "B04",  # Channel 2: Red (665 nm) - 10m [Sentinel-2]
    "B08",  # Channel 3: Near Infrared / NIR (842 nm) - 10m [Sentinel-2]
    "B05",  # Channel 4: Vegetation Red Edge 1 (705 nm) - 20m [Sentinel-2]
    "B06",  # Channel 5: Vegetation Red Edge 2 (740 nm) - 20m [Sentinel-2]
    "B07",  # Channel 6: Vegetation Red Edge 3 (783 nm) - 20m [Sentinel-2]
    "B11",  # Channel 7: Shortwave Infrared 1 / SWIR-1 (1610 nm) - 20m [Sentinel-2]
    "B12",  # Channel 8: Shortwave Infrared 2 / SWIR-2 (2190 nm) - 20m [Sentinel-2]
    "B8A",  # Channel 9: Narrow NIR (865 nm) - 20m [Sentinel-2]
    "VH",   # Channel 10: Cross-polarization SAR backscatter [Sentinel-1]
    "VV",   # Channel 11: Co-polarization SAR backscatter [Sentinel-1]
]

BAND_INDICES = {name: i for i, name in enumerate(BAND_NAMES)}

# Standardized BigEarthNet 19 Land-Cover Classes
BIGEARTHNET_19_CLASSES = [
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Sclerophyllous vegetation",
    "Transitional woodland/shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters",
]

# Backend & Server Settings
API_HOST = os.getenv("SATQUERY_HOST", "0.0.0.0")
API_PORT = int(os.getenv("SATQUERY_PORT", "8000"))
API_TITLE = "SatQuery Earth Observation VLM & Agent API"
API_VERSION = "1.0.0"
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))

