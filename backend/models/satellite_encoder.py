import logging
import torch
import torch.nn as nn
from backend.config import MODEL_CHECKPOINT, NUM_CHANNELS, VISION_FEATURE_DIM

logger = logging.getLogger(__name__)


class MockSatelliteBackbone(nn.Module):
    """
    Lightweight fallback multispectral encoder matching MobileViT-s 640-dim feature map.
    Used when reben_publication or HuggingFace checkpoint is unavailable locally.
    """
    def __init__(self, in_channels=NUM_CHANNELS, out_dim=VISION_FEATURE_DIM):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, out_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.flatten = nn.Flatten()

    def forward(self, x):
        return self.flatten(self.features(x))


def _load_pretrained_checkpoint(checkpoint_name: str) -> nn.Module:
    """
    Loads BIFOLD-BigEarthNetv2-0/mobilevit_s-all-v0.1.1 checkpoint.
    Tries reben_publication first; if not available, loads directly via
    ConfigILM and safetensors from local cache or HuggingFace Hub.
    """
    try:
        from reben_publication.BigEarthNetv2_0_ImageClassifier import BigEarthNetv2_0_ImageClassifier
        return BigEarthNetv2_0_ImageClassifier.from_pretrained(checkpoint_name)
    except Exception as exc:
        logger.info(f"reben_publication direct import not available ({exc}), loading via ConfigILM+safetensors")

    import os
    import json
    import inspect
    from pathlib import Path
    from configilm.ConfigILM import ILMConfiguration, ILMType, ConfigILM
    from safetensors.torch import load_file

    candidate_paths = [
        Path(checkpoint_name),
        Path("_hf_cache") / Path(checkpoint_name).name,
        Path.cwd() / "_hf_cache" / Path(checkpoint_name).name,
    ]
    local_dir = None
    for p in candidate_paths:
        if p.exists() and (p / "model.safetensors").exists() and (p / "config.json").exists():
            local_dir = str(p.resolve())
            break

    if local_dir is None:
        from huggingface_hub import snapshot_download
        local_dir = snapshot_download(
            repo_id=checkpoint_name,
            allow_patterns=["config.json", "model.safetensors"],
        )

    with open(os.path.join(local_dir, "config.json"), "r") as f:
        cfg_dict = json.load(f)

    sig = inspect.signature(ILMConfiguration.__init__)
    kwargs = {k: v for k, v in cfg_dict.items() if k in sig.parameters}
    if "network_type" in kwargs:
        kwargs["network_type"] = ILMType(kwargs["network_type"])
    kwargs["load_pretrained_hf_if_available"] = False
    kwargs["load_pretrained_timm_if_available"] = False

    ilm_config = ILMConfiguration(**kwargs)
    model = ConfigILM(ilm_config)

    weights = load_file(os.path.join(local_dir, "model.safetensors"))
    stripped = {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in weights.items()}
    model.load_state_dict(stripped, strict=True)
    return model


class SatelliteEncoder(nn.Module):
    """
    Multispectral Sentinel-2 Vision Encoder.
    Wraps BigEarthNetv2_0_ImageClassifier (MobileViT-s) to extract 640-dim representation vectors
    from 12-channel multispectral satellite imagery.
    """
    def __init__(self, checkpoint_name: str = MODEL_CHECKPOINT, use_mock_fallback: bool = False):
        super().__init__()
        self.checkpoint_name = checkpoint_name
        self.is_mock = use_mock_fallback

        if not use_mock_fallback:
            try:
                # Monkey-patch configilm to prevent hub configuration lookup errors on load
                try:
                    from configilm.ConfigILM import ILMConfiguration
                    if not hasattr(ILMConfiguration, "items"):
                        ILMConfiguration.items = lambda s: s.__dict__.items()
                except ImportError:
                    pass

                self.classifier = _load_pretrained_checkpoint(checkpoint_name)
                # Freeze entire pretrained backbone
                for param in self.parameters():
                    param.requires_grad = False
                self.eval()
                self.is_mock = False
                logger.info(f"Loaded pretrained BigEarthNet backbone: {checkpoint_name}")
            except Exception as exc:
                logger.warning(
                    f"Pretrained BigEarthNet backbone could not be loaded ({exc}). "
                    "Falling back to multispectral architecture."
                )
                self.classifier = None
                self.is_mock = True
        else:
            self.classifier = None

        if self.is_mock:
            self.backbone = MockSatelliteBackbone(in_channels=NUM_CHANNELS, out_dim=VISION_FEATURE_DIM)
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (B, 12, H, W)
        Returns:
            Tensor of shape (B, 640)
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)

        if x.shape[1] != NUM_CHANNELS:
            raise ValueError(f"Expected exactly {NUM_CHANNELS} channels, got {x.shape[1]}")

        if self.is_mock or self.classifier is None:
            with torch.no_grad():
                return self.backbone(x)

        # Pretrained BigEarthNet extraction
        with torch.no_grad():
            vision_encoder = getattr(self.classifier, "model", self.classifier).vision_encoder
            if hasattr(vision_encoder, "forward_features"):
                features = vision_encoder.forward_features(x)
            else:
                features = vision_encoder(x)

            if len(features.shape) == 4:
                # [B, C, H, W] -> Global average pool over spatial dimensions
                features = torch.mean(features, dim=[2, 3])
            elif len(features.shape) == 3:
                # [B, N, C] -> Pool over sequence dimension
                features = torch.mean(features, dim=1)

            return features

