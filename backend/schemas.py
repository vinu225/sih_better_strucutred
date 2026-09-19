from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Core health / tile schemas
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    device: str
    model_checkpoint: str
    llm_name: str


class TileSummary(BaseModel):
    tile_id: str
    terrain: str
    has_npy: bool
    has_rgb: bool


class TileListResponse(BaseModel):
    total_tiles: int
    tiles: List[TileSummary]


# ---------------------------------------------------------------------------
# Modality schemas
# ---------------------------------------------------------------------------

class ModalityInfoResponse(BaseModel):
    """Detected or user-specified modality for a tile."""
    tile_id: str
    modality: str
    channel_count: int
    detection_basis: str
    basis_label: str
    display_label: str
    band_names: Optional[List[str]] = None
    available_analyses: List[str]


class ModalityHint(BaseModel):
    """Optional user-provided modality override for any analysis request."""
    modality_hint: Optional[str] = Field(
        default=None,
        description=(
            "Explicit modality override. One of: MULTIMODAL_S1_S2, "
            "SENTINEL2_MULTISPECTRAL, SAR_ONLY, NIR_INFRARED, RGB_OPTICAL. "
            "If None, modality is inferred from metadata or shape."
        ),
    )


# ---------------------------------------------------------------------------
# VLM schemas
# ---------------------------------------------------------------------------

class VLMQueryRequest(BaseModel):
    tile_id: str = Field(default="sample_forest_tile", description="Sample tile name or identifier")
    question: str = Field(
        default="What is the dominant landscape in this satellite image?",
        description="Natural language question",
    )
    max_new_tokens: int = Field(default=64, ge=16, le=256)
    modality_hint: Optional[str] = Field(default=None)


class VLMQueryResponse(BaseModel):
    tile_id: str
    question: str
    answer: str
    primary_landcover: str
    confidence: float
    spectral_summary: str
    coverage_estimates: Dict[str, float]


# ---------------------------------------------------------------------------
# Classification schemas
# ---------------------------------------------------------------------------

class ClassificationRequest(BaseModel):
    tile_id: str = Field(default="sample_forest_tile")
    top_k: int = Field(default=5, ge=1, le=19)
    threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    modality_hint: Optional[str] = Field(default=None)


class ClassPrediction(BaseModel):
    class_name: str
    confidence: float


class ClassificationResponse(BaseModel):
    tile_id: str
    primary_class: str
    primary_confidence: float
    top_predictions: List[ClassPrediction]
    detected_classes: List[ClassPrediction]


# ---------------------------------------------------------------------------
# Spectral schemas
# ---------------------------------------------------------------------------

class SpectralRequest(BaseModel):
    tile_id: str = Field(default="sample_forest_tile")
    modality_hint: Optional[str] = Field(default=None)


class IndexStats(BaseModel):
    mean: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    std: Optional[float] = None
    status: Optional[str] = None    # "UNAVAILABLE" when bands absent
    reason: Optional[str] = None


class CoverageEstimates(BaseModel):
    dense_vegetation_percent: Optional[float] = None
    water_body_percent: Optional[float] = None
    builtup_percent: Optional[float] = None
    status: Optional[str] = None
    reason: Optional[str] = None


class SpectralResponse(BaseModel):
    tile_id: str
    modality: str
    ndvi_stats: Dict[str, Any]
    ndwi_stats: Dict[str, Any]
    ndbi_stats: Dict[str, Any]
    coverage_estimates: Dict[str, Any]
    assessment: str


# ---------------------------------------------------------------------------
# SAR analysis schemas
# ---------------------------------------------------------------------------

class SARRequest(BaseModel):
    tile_id: str = Field(default="sample_forest_tile")
    modality_hint: Optional[str] = Field(default=None)


class SARChannelStats(BaseModel):
    label: str
    mean: float
    std: float
    min: float
    max: float
    median: float


class SARResponse(BaseModel):
    tile_id: str
    scope_note: str
    num_channels: int
    channel_labels: List[str]
    channels: List[Dict[str, Any]]
    polarization_ratio: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Change detection schemas
# ---------------------------------------------------------------------------

class ChangeDetectionRequest(BaseModel):
    tile_id_before: str = Field(..., description="Tile ID for the 'before' image")
    tile_id_after: str = Field(..., description="Tile ID for the 'after' image")
    co_registered: bool = Field(
        default=False,
        description=(
            "Set to true ONLY when both tiles are confirmed to be spatially "
            "co-registered (same grid, resolution, and extent). "
            "Without co-registration, pixel-wise differences reflect alignment "
            "errors, not geographic change."
        ),
    )
    modality_hint_before: Optional[str] = Field(default=None)
    modality_hint_after: Optional[str] = Field(default=None)
    label_before: str = Field(default="before")
    label_after: str = Field(default="after")


class ChangeDetectionResponse(BaseModel):
    tile_id_before: str
    tile_id_after: str
    pair_type: str
    co_registered: bool
    warning: Optional[str] = None
    image_before: Dict[str, Any]
    image_after: Dict[str, Any]
    pixel_change: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Agent / chat schemas
# ---------------------------------------------------------------------------

class AgentChatRequest(BaseModel):
    tile_id: str = Field(default="sample_forest_tile")
    query: str = Field(..., description="User question or analysis command for SatQuery AI Agent")
    modality_hint: Optional[str] = Field(default=None)


class AgentChatResponse(BaseModel):
    tile_id: str
    query: str
    response: str
    plan: List[str]
    tool_artifacts: Dict[str, Any]
    history_length: int
    answer: Optional[str] = None
    confidence: Optional[float] = None
    visual_evidence: Optional[Any] = None
    selected_task: Optional[str] = None
    selected_model_or_tool: Optional[str] = None
    detected_modality: Optional[str] = None
    execution_trace: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Structured Agentic /analyze Schemas
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    tile_id: Optional[str] = Field(default="sample_forest_tile", description="Primary tile ID or sample name")
    tile_ids: Optional[List[str]] = Field(default=None, description="Tile IDs for two-image pairs (e.g. before/after or optical+SAR)")
    query: str = Field(..., description="Natural language question or analysis command")
    modality_hint: Optional[str] = Field(default=None, description="Optional modality hint for primary tile")
    modality_hints: Optional[List[str]] = Field(default=None, description="Optional modality hints for image pair")


class AnalyzeResponse(BaseModel):
    answer: str
    confidence: float
    visual_evidence: Optional[Any] = None
    selected_task: str
    selected_model_or_tool: str
    detected_modality: str
    execution_trace: List[str]

