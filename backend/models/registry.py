import logging
from typing import Optional
from backend.config import DEVICE, MODEL_CHECKPOINT, LLM_MODEL_NAME
from backend.models.vlm import SatQueryVLM

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Thread-safe model registry managing lifecycle and caching for SatQuery models.
    """
    _vlm_instance: Optional[SatQueryVLM] = None
    _device: str = DEVICE

    @classmethod
    def get_vlm(
        cls,
        vision_checkpoint: str = MODEL_CHECKPOINT,
        llm_name: str = LLM_MODEL_NAME,
        force_reload: bool = False,
        load_pretrained_llm: bool = True,
        use_mock_encoder: bool = False,
    ) -> SatQueryVLM:
        """
        Get or initialize the singleton SatQueryVLM instance.
        """
        if cls._vlm_instance is None or force_reload:
            logger.info(f"Initializing SatQueryVLM on device: {cls._device}")
            cls._vlm_instance = SatQueryVLM(
                vision_checkpoint=vision_checkpoint,
                llm_name=llm_name,
                device=cls._device,
                load_pretrained_llm=load_pretrained_llm,
                use_mock_encoder=use_mock_encoder,
            )
        return cls._vlm_instance

    @classmethod
    def set_device(cls, device: str):
        cls._device = device
        if cls._vlm_instance is not None:
            cls._vlm_instance.to(device)
            cls._vlm_instance.device = device
