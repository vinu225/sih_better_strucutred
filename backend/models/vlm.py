import logging
from typing import Optional, Tuple, Union, List
import torch
import torch.nn as nn
from backend.config import (
    DEVICE,
    MODEL_CHECKPOINT,
    LLM_MODEL_NAME,
    VISION_FEATURE_DIM,
    DEFAULT_MAX_NEW_TOKENS,
)
from backend.models.satellite_encoder import SatelliteEncoder
from backend.models.projector import VisionProjector

logger = logging.getLogger(__name__)


class SatQueryVLM(nn.Module):
    """
    Unified Satellite Vision-Language Model framework.
    Maps 12-band multispectral Sentinel-2 representations into causal language model
    text spaces using projected inputs_embeds prefix tokens.
    """
    def __init__(
        self,
        vision_checkpoint: str = MODEL_CHECKPOINT,
        llm_name: str = LLM_MODEL_NAME,
        device: str = DEVICE,
        load_pretrained_llm: bool = True,
        use_mock_encoder: bool = False,
    ):
        super().__init__()
        self.device = device
        self.llm_name = llm_name
        self.load_pretrained_llm = load_pretrained_llm

        # 1. Initialize Visual Backbone
        self.vision_encoder = SatelliteEncoder(
            checkpoint_name=vision_checkpoint,
            use_mock_fallback=use_mock_encoder,
        ).to(device)

        # 2. Initialize Language Model & Tokenizer
        self.tokenizer = None
        self.llm = None
        self.llm_hidden_size = 896  # Default hidden size for Qwen2.5-0.5B

        if load_pretrained_llm:
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                logger.info(f"Loading LLM tokenizer & weights: {llm_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                self.llm = AutoModelForCausalLM.from_pretrained(llm_name).to(device)
                self.llm_hidden_size = self.llm.config.hidden_size

                # Freeze language model backbone
                for param in self.llm.parameters():
                    param.requires_grad = False
                self.llm.eval()
                logger.info(f"Successfully loaded LLM ({llm_name}) with hidden size {self.llm_hidden_size}")
            except Exception as exc:
                logger.warning(
                    f"Could not load HuggingFace LLM {llm_name} ({exc}). Initializing lightweight mock LLM."
                )
                self._init_fallback_llm()
        else:
            self._init_fallback_llm()

        # 3. Initialize Visual Alignment Projector (Trainable)
        self.projector = VisionProjector(
            input_dim=VISION_FEATURE_DIM,
            output_dim=self.llm_hidden_size,
        ).to(device)

    def _init_fallback_llm(self):
        """Fallback lightweight embedding layer for offline / test environments."""
        self.llm_hidden_size = 896
        self.mock_embedding = nn.Embedding(1000, self.llm_hidden_size).to(self.device)
        self.tokenizer = None
        self.llm = None

    def get_multimodal_embeddings(
        self,
        satellite_img: torch.Tensor,
        question_text: Union[str, List[str]],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Combines visual prefix token with question text token embeddings.
        Returns:
            multimodal_emb: (B, num_tokens + 1, hidden_size)
            attention_mask: (B, num_tokens + 1)
        """
        if satellite_img.device != torch.device(self.device):
            satellite_img = satellite_img.to(self.device)

        # Extract visual feature: (B, 640)
        vision_features = self.vision_encoder(satellite_img)

        # Project to LLM hidden dimension: (B, 1, hidden_size)
        projected_emb = self.projector(vision_features).unsqueeze(1)
        batch_size = projected_emb.size(0)

        if self.llm is not None and self.tokenizer is not None:
            if isinstance(question_text, str):
                question_text = [question_text]

            text_inputs = self.tokenizer(
                question_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(self.device)

            text_emb = self.llm.get_input_embeddings()(text_inputs.input_ids)
            multimodal_emb = torch.cat([projected_emb.to(dtype=text_emb.dtype), text_emb], dim=1)

            # Build extended attention mask (prepend 1 for the visual token)
            img_mask = torch.ones((batch_size, 1), dtype=text_inputs.attention_mask.dtype, device=self.device)
            attention_mask = torch.cat([img_mask, text_inputs.attention_mask], dim=1)
            return multimodal_emb, attention_mask
        else:
            # Synthetic / fallback representation
            dummy_tokens = torch.randint(0, 500, (batch_size, 10), device=self.device)
            text_emb = self.mock_embedding(dummy_tokens)
            multimodal_emb = torch.cat([projected_emb, text_emb], dim=1)
            attention_mask = torch.ones(multimodal_emb.shape[:2], device=self.device)
            return multimodal_emb, attention_mask

    def forward(
        self,
        satellite_img: torch.Tensor,
        question_text: Union[str, List[str]],
    ):
        """
        Forward pass producing language logits.
        """
        multimodal_emb, attention_mask = self.get_multimodal_embeddings(satellite_img, question_text)

        if self.llm is not None:
            outputs = self.llm(inputs_embeds=multimodal_emb, attention_mask=attention_mask)
            return outputs, multimodal_emb.shape
        else:
            # Fallback output
            dummy_logits = torch.randn(
                multimodal_emb.size(0),
                multimodal_emb.size(1),
                1000,
                device=self.device,
            )
            return type("Outputs", (), {"logits": dummy_logits})(), multimodal_emb.shape

    def generate_answer(
        self,
        satellite_img: torch.Tensor,
        question_text: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate answer for the satellite question.
        """
        if self.llm is not None and self.tokenizer is not None:
            multimodal_emb, attention_mask = self.get_multimodal_embeddings(satellite_img, question_text)
            with torch.no_grad():
                generated_ids = self.llm.generate(
                    inputs_embeds=multimodal_emb,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            answer = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            return answer.strip()
        else:
            # Rule-based fallback response if weights aren't downloaded
            return (
                f"[SatQuery VLM Analysis] Multispectral region evaluated for: '{question_text}'. "
                f"Spectral signatures indicate dominant terrain features and vegetative balance."
            )
