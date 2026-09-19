from typing import Dict, Any, Union, Optional
import numpy as np
import torch
from backend.models.registry import ModelRegistry
from backend.services.spectral_service import SpectralService
from backend.services.classification_service import ClassificationService


class VLMService:
    """
    Multimodal Visual Question Answering Service for Earth Observation.
    """

    @classmethod
    def query(
        cls,
        tensor_12ch: Union[torch.Tensor, np.ndarray],
        question: str,
        max_new_tokens: int = 64,
        enrich_with_spectral_context: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute visual question answering with optional remote-sensing context enrichment.
        """
        if isinstance(tensor_12ch, np.ndarray):
            tensor_12ch = torch.from_numpy(tensor_12ch).float()
        if tensor_12ch.dim() == 3:
            tensor_12ch = tensor_12ch.unsqueeze(0)

        vlm = ModelRegistry.get_vlm()
        tensor_12ch = tensor_12ch.to(vlm.device)

        # Retrieve spectral indices and classification for grounding
        spectral_data = SpectralService.analyze_tile(tensor_12ch)
        classification = ClassificationService.classify_tile(tensor_12ch, top_k=3)

        if enrich_with_spectral_context and (vlm.llm is not None):
            context_hint = (
                f"[Sentinel-2 Context: Detected primary class '{classification['primary_class']}' "
                f"with {classification['primary_confidence']:.2f} confidence. "
                f"NDVI: {spectral_data['ndvi_stats']['mean']:.2f}, "
                f"NDWI: {spectral_data['ndwi_stats']['mean']:.2f}]. "
            )
            prompt = f"{context_hint}Question: {question} Answer:"
        else:
            prompt = f"Question: {question} Answer:"

        # Generate answer from multimodal model
        raw_answer = vlm.generate_answer(
            satellite_img=tensor_12ch,
            question_text=prompt,
            max_new_tokens=max_new_tokens,
        )

        # If fallback mock VLM produced default message, formulate a grounded answer
        if "Rule-based fallback" in raw_answer or "[SatQuery VLM Analysis]" in raw_answer:
            answer = (
                f"Based on multispectral Sentinel-2 analysis, this region is predominantly "
                f"{classification['primary_class']} (confidence: {classification['primary_confidence']*100:.1f}%). "
                f"{spectral_data['assessment']}"
            )
        else:
            answer = raw_answer

        return {
            "question": question,
            "answer": answer,
            "primary_landcover": classification["primary_class"],
            "confidence": classification["primary_confidence"],
            "spectral_summary": spectral_data["assessment"],
            "coverage_estimates": spectral_data["coverage_estimates"],
        }
