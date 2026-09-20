import io
import uuid
from pathlib import Path
from typing import Optional, List, Union, Tuple, Any
import numpy as np

import torch
from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form, Request
from fastapi.responses import Response

from backend.config import (
    DEVICE,
    MODEL_CHECKPOINT,
    LLM_MODEL_NAME,
    SAMPLES_DIR,
    API_VERSION,
    MAX_UPLOAD_MB,
)
from backend.schemas import (
    HealthResponse,
    TileListResponse,
    TileSummary,
    UploadTileResponse,
    VLMQueryRequest,
    VLMQueryResponse,
    ClassificationRequest,
    ClassificationResponse,
    SpectralRequest,
    SpectralResponse,
    SARRequest,
    SARResponse,
    ChangeDetectionRequest,
    ChangeDetectionResponse,
    ModalityInfoResponse,
    AgentChatRequest,
    AgentChatResponse,
    AnalyzeRequest,
    AnalyzeResponse,
    MessageHistoryItem,
    SessionHistoryResponse,
    SessionDeleteResponse,
)
from backend.data.modality import (
    InputModality,
    ModalityError,
    detect_modality,
    standardize_image_input,
    to_rgb_image,
)
from backend.data.sample_generator import generate_sample_tiles
from backend.data.sentinel_loader import (
    render_rgb_composite,
    render_false_color_cir,
    render_swir_composite,
)
from backend.services.vlm_service import VLMService
from backend.services.classification_service import ClassificationService
from backend.services.spectral_service import SpectralService
from backend.services.sar_service import SARService
from backend.services.change_detection_service import ChangeDetectionService
from backend.agent.sat_agent import SatQueryAgent

router = APIRouter()
_global_agent = SatQueryAgent()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_tile_tensor(tile_id: str) -> np.ndarray:
    """Resolve and load a numpy array from the samples directory."""
    if not SAMPLES_DIR.exists() or len(list(SAMPLES_DIR.glob("*.npy"))) == 0:
        generate_sample_tiles(SAMPLES_DIR)

    file_path = SAMPLES_DIR / f"{tile_id}.npy"
    if not file_path.exists():
        matches = list(SAMPLES_DIR.glob(f"*{tile_id}*.npy"))
        if matches:
            file_path = matches[0]
        else:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Tile '{tile_id}' not found in {SAMPLES_DIR}. "
                    f"Available: {[f.stem for f in SAMPLES_DIR.glob('*.npy')]}"
                ),
            )
    return np.load(str(file_path))


def _resolve_modality(arr: np.ndarray, hint: Optional[str]):
    """Detect modality from array + optional user hint string."""
    return detect_modality(arr, user_hint=hint)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok",
        version=API_VERSION,
        device=DEVICE,
        model_checkpoint=MODEL_CHECKPOINT,
        llm_name=LLM_MODEL_NAME,
    )


# ---------------------------------------------------------------------------
# Tile list
# ---------------------------------------------------------------------------

@router.get("/api/v1/tiles", response_model=TileListResponse)
def list_tiles():
    if not SAMPLES_DIR.exists() or len(list(SAMPLES_DIR.glob("*.npy"))) == 0:
        generate_sample_tiles(SAMPLES_DIR)

    npy_files = list(SAMPLES_DIR.glob("*.npy"))
    tiles = []
    for f in npy_files:
        stem = f.stem
        terrain = stem.replace("sample_", "").replace("_tile", "")
        has_rgb = (SAMPLES_DIR / f"{stem}_rgb.png").exists()
        tiles.append(
            TileSummary(tile_id=stem, terrain=terrain, has_npy=True, has_rgb=has_rgb)
        )
    return TileListResponse(total_tiles=len(tiles), tiles=tiles)


# ---------------------------------------------------------------------------
# Modality detection endpoint
# ---------------------------------------------------------------------------

@router.get("/api/v1/tiles/{tile_id}/modality", response_model=ModalityInfoResponse)
def get_tile_modality(
    tile_id: str,
    modality_hint: Optional[str] = Query(default=None),
):
    """
    Detect (or confirm) the modality of a stored tile.
    Pass modality_hint to override shape-based detection.
    """
    arr = _load_tile_tensor(tile_id)
    spec = _resolve_modality(arr, modality_hint)
    d = spec.to_dict()
    return ModalityInfoResponse(
        tile_id=tile_id,
        modality=d["modality"],
        channel_count=d["channel_count"],
        detection_basis=d["detection_basis"],
        basis_label=d["basis_label"],
        display_label=d["display_label"],
        band_names=d["band_names"],
        available_analyses=d["available_analyses"],
    )


# ---------------------------------------------------------------------------
# Composites (RGB/CIR/SWIR)
# ---------------------------------------------------------------------------

@router.get("/api/v1/tiles/{tile_id}/composite")
def get_tile_composite(
    tile_id: str,
    mode: str = Query(default="rgb", pattern="^(rgb|cir|swir)$"),
):
    arr = _load_tile_tensor(tile_id)
    if mode == "rgb":
        img = render_rgb_composite(arr)
    elif mode == "cir":
        img = render_false_color_cir(arr)
    else:
        img = render_swir_composite(arr)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


# ---------------------------------------------------------------------------
# VLM query
# ---------------------------------------------------------------------------

@router.post("/api/v1/vlm/query", response_model=VLMQueryResponse)
def query_vlm(req: VLMQueryRequest):
    arr = _load_tile_tensor(req.tile_id)
    res = VLMService.query(arr, req.question, max_new_tokens=req.max_new_tokens)
    return VLMQueryResponse(
        tile_id=req.tile_id,
        question=res["question"],
        answer=res["answer"],
        primary_landcover=res["primary_landcover"],
        confidence=res["confidence"],
        spectral_summary=res["spectral_summary"],
        coverage_estimates=res["coverage_estimates"],
    )


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

@router.post("/api/v1/classify", response_model=ClassificationResponse)
def classify_landcover(req: ClassificationRequest):
    arr = _load_tile_tensor(req.tile_id)
    spec = _resolve_modality(arr, req.modality_hint)

    # Classification requires at least SENTINEL2_MULTISPECTRAL or MULTIMODAL_S1_S2
    blocked = {InputModality.SAR_ONLY, InputModality.RGB_OPTICAL, InputModality.NIR_INFRARED}
    if spec.modality in blocked:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Land-cover classification is not supported for modality "
                f"'{spec.modality.value}'. "
                "Requires Sentinel-2 multispectral or S1+S2 multimodal data."
            ),
        )

    res = ClassificationService.classify_tile(arr, top_k=req.top_k, threshold=req.threshold)
    return ClassificationResponse(
        tile_id=req.tile_id,
        primary_class=res["primary_class"],
        primary_confidence=res["primary_confidence"],
        top_predictions=res["top_predictions"],
        detected_classes=res["detected_classes"],
    )


# ---------------------------------------------------------------------------
# Spectral indices
# ---------------------------------------------------------------------------

@router.post("/api/v1/spectral", response_model=SpectralResponse)
def analyze_spectral(req: SpectralRequest):
    arr = _load_tile_tensor(req.tile_id)
    spec = _resolve_modality(arr, req.modality_hint)

    try:
        res = SpectralService.analyze_tile(arr, spec=spec)
    except ModalityError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return SpectralResponse(
        tile_id=req.tile_id,
        modality=spec.modality.value,
        ndvi_stats=res["ndvi_stats"],
        ndwi_stats=res["ndwi_stats"],
        ndbi_stats=res["ndbi_stats"],
        coverage_estimates=res["coverage_estimates"],
        assessment=res.get("assessment", ""),
    )


# ---------------------------------------------------------------------------
# SAR analysis
# ---------------------------------------------------------------------------

@router.post("/api/v1/sar", response_model=SARResponse)
def analyze_sar(req: SARRequest):
    arr = _load_tile_tensor(req.tile_id)
    spec = _resolve_modality(arr, req.modality_hint)

    # Determine SAR channels to pass
    if spec.modality == InputModality.MULTIMODAL_S1_S2:
        sar_arr = SARService.extract_sar_channels(arr)
        channel_labels = ["VH", "VV"]
    elif spec.modality == InputModality.SAR_ONLY:
        sar_arr = arr
        channel_labels = (
            spec.band_names
            if spec.band_names
            else [f"ch_{i}" for i in range(arr.shape[0])]
        )
    else:
        raise HTTPException(
            status_code=422,
            detail=(
                f"SAR analysis requires SAR_ONLY or MULTIMODAL_S1_S2 modality. "
                f"Detected: '{spec.modality.value}'. "
                "Use modality_hint to override if the detection is incorrect."
            ),
        )

    try:
        res = SARService.analyze(sar_arr, spec=spec, channel_labels=channel_labels)
    except ModalityError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return SARResponse(
        tile_id=req.tile_id,
        scope_note=res["scope_note"],
        num_channels=res["num_channels"],
        channel_labels=res["channel_labels"],
        channels=res["channels"],
        polarization_ratio=res.get("polarization_ratio"),
    )


# ---------------------------------------------------------------------------
# Change detection
# ---------------------------------------------------------------------------

async def _process_upload_file(
    file: Union[UploadFile, Any],
    modality_hint: Optional[str] = None,
    max_mb: Optional[int] = None,
    tile_id_prefix: Optional[str] = None,
) -> Tuple[str, np.ndarray]:
    """
    Ingest, validate, standardize, and save an uploaded satellite image tile.
    Returns (tile_id, numpy_array).
    """
    if max_mb is None:
        max_mb = MAX_UPLOAD_MB

    if not file or not hasattr(file, "filename") or not file.filename:
        raise HTTPException(status_code=422, detail="Missing required image file.")

    fname = file.filename
    fname_lower = fname.lower()
    valid_exts = (".npy", ".npz", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")

    if not fname_lower.endswith(valid_exts):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported format for '{fname}'. Supported formats: {list(valid_exts)}",
        )

    content = await file.read() if hasattr(file, "read") else b""
    if len(content) == 0:
        raise HTTPException(status_code=400, detail=f"Uploaded file '{fname}' is empty.")

    if len(content) > max_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded file '{fname}' ({len(content)/(1024*1024):.1f}MB) exceeds maximum limit of {max_mb}MB.",
        )


    # Magic bytes verification
    valid_magic = False
    if fname_lower.endswith(".png"):
        valid_magic = content.startswith(b"\x89PNG\r\n\x1a\n")
    elif fname_lower.endswith((".jpg", ".jpeg")):
        valid_magic = content.startswith(b"\xff\xd8\xff")
    elif fname_lower.endswith(".npy"):
        valid_magic = content.startswith(b"\x93NUMPY")
    elif fname_lower.endswith(".npz"):
        valid_magic = content.startswith(b"PK\x03\x04")
    elif fname_lower.endswith((".tif", ".tiff")):
        valid_magic = (
            content.startswith(b"II*\x00")
            or content.startswith(b"MM\x00*")
            or content.startswith(b"II\x2b\x00")
        )
    elif fname_lower.endswith(".bmp"):
        valid_magic = content.startswith(b"BM")
    elif fname_lower.endswith(".webp"):
        valid_magic = content.startswith(b"RIFF") and b"WEBP" in content[:16]

    if not valid_magic:
        raise HTTPException(
            status_code=415,
            detail=f"Uploaded file '{fname}' header content does not match expected format magic bytes.",
        )

    try:
        arr = standardize_image_input(content)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Corrupt or unreadable image file '{fname}': {exc}")

    stem = Path(fname).stem
    tile_id = f"{tile_id_prefix}{stem}" if tile_id_prefix else stem
    save_path = SAMPLES_DIR / f"{tile_id}.npy"
    np.save(str(save_path), arr)

    try:
        rgb_preview = to_rgb_image(arr)
        rgb_preview.save(str(SAMPLES_DIR / f"{tile_id}_rgb.png"))
    except Exception:
        pass

    return tile_id, arr


@router.post(
    "/api/v1/change-detection",
    response_model=ChangeDetectionResponse,
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {
                    "schema": {
                        "$ref": "#/components/schemas/ChangeDetectionRequest"
                    }
                },
                "multipart/form-data": {
                    "schema": {
                        "type": "object",
                        "required": ["image_before", "image_after"],
                        "properties": {
                            "image_before": {
                                "type": "string",
                                "format": "binary",
                                "description": "Before image file (.tif, .tiff, .png, .jpg, .jpeg, .npy)"
                            },
                            "image_after": {
                                "type": "string",
                                "format": "binary",
                                "description": "After image file (.tif, .tiff, .png, .jpg, .jpeg, .npy)"
                            },
                            "co_registered": {
                                "type": "boolean",
                                "default": True,
                                "description": "Set to true if images are spatially co-registered"
                            },
                            "modality_hint_before": {
                                "type": "string",
                                "description": "Optional modality hint for before image"
                            },
                            "modality_hint_after": {
                                "type": "string",
                                "description": "Optional modality hint for after image"
                            },
                            "label_before": {
                                "type": "string",
                                "default": "before"
                            },
                            "label_after": {
                                "type": "string",
                                "default": "after"
                            }
                        }
                    }
                }
            }
        }
    },
)
async def change_detection(request: Request):
    content_type = request.headers.get("content-type", "").lower()

    if content_type.startswith("application/json"):
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Invalid JSON body: {exc}")

        try:
            req = ChangeDetectionRequest(**body)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Validation error: {exc}")

        arr_before = _load_tile_tensor(req.tile_id_before)
        arr_after = _load_tile_tensor(req.tile_id_after)

        spec_before = _resolve_modality(arr_before, req.modality_hint_before)
        spec_after = _resolve_modality(arr_after, req.modality_hint_after)

        res = ChangeDetectionService.compare(
            arr_before,
            arr_after,
            co_registered=req.co_registered,
            spec_before=spec_before,
            spec_after=spec_after,
            label_before=req.label_before,
            label_after=req.label_after,
        )

        return ChangeDetectionResponse(
            tile_id_before=req.tile_id_before,
            tile_id_after=req.tile_id_after,
            pair_type=res["pair_type"],
            co_registered=res["co_registered"],
            warning=res.get("warning"),
            image_before=res["image_before"],
            image_after=res["image_after"],
            pixel_change=res.get("pixel_change"),
        )

    elif content_type.startswith("multipart/form-data"):
        try:
            form = await request.form()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to parse multipart form data: {exc}")

        file_before = form.get("image_before")
        file_after = form.get("image_after")

        if not file_before or not hasattr(file_before, "filename") or not file_before.filename:
            raise HTTPException(status_code=422, detail="Missing required file field 'image_before'.")
        if not file_after or not hasattr(file_after, "filename") or not file_after.filename:
            raise HTTPException(status_code=422, detail="Missing required file field 'image_after'.")

        co_reg_raw = form.get("co_registered")
        if co_reg_raw is None or co_reg_raw == "":
            co_registered = True
        elif isinstance(co_reg_raw, str):
            co_registered = co_reg_raw.lower() in ("true", "1", "yes")
        else:
            co_registered = bool(co_reg_raw)

        hint_before = form.get("modality_hint_before") or None
        hint_after = form.get("modality_hint_after") or None
        label_before = str(form.get("label_before") or "before")
        label_after = str(form.get("label_after") or "after")

        prefix_b = f"upload_{uuid.uuid4().hex[:8]}_"
        prefix_a = f"upload_{uuid.uuid4().hex[:8]}_"

        tile_id_b, arr_b = await _process_upload_file(file_before, hint_before, tile_id_prefix=prefix_b)
        tile_id_a, arr_a = await _process_upload_file(file_after, hint_after, tile_id_prefix=prefix_a)

        spec_before = _resolve_modality(arr_b, hint_before)
        spec_after = _resolve_modality(arr_a, hint_after)

        res = ChangeDetectionService.compare(
            arr_b,
            arr_a,
            co_registered=co_registered,
            spec_before=spec_before,
            spec_after=spec_after,
            label_before=label_before,
            label_after=label_after,
        )

        return ChangeDetectionResponse(
            tile_id_before=tile_id_b,
            tile_id_after=tile_id_a,
            pair_type=res["pair_type"],
            co_registered=res["co_registered"],
            warning=res.get("warning"),
            image_before=res["image_before"],
            image_after=res["image_after"],
            pixel_change=res.get("pixel_change"),
        )

    else:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported Content-Type '{content_type}'. Expected application/json or multipart/form-data.",
        )


@router.post(
    "/api/v1/change-detection/upload",
    response_model=ChangeDetectionResponse,
    summary="Bi-Temporal Change Detection (Upload Image Files)",
    description="Upload two satellite/aerial images (Before and After) to compute temporal change metrics.",
)
async def change_detection_upload(
    image_before: UploadFile = File(..., description="Before image file (.tif, .tiff, .png, .jpg, .jpeg, .npy)"),
    image_after: UploadFile = File(..., description="After image file (.tif, .tiff, .png, .jpg, .jpeg, .npy)"),
    co_registered: bool = Form(default=True, description="Set to true if images are spatially co-registered"),
    modality_hint_before: Optional[str] = Form(default=None, description="Optional modality hint for before image"),
    modality_hint_after: Optional[str] = Form(default=None, description="Optional modality hint for after image"),
    label_before: str = Form(default="before", description="Display label for before image"),
    label_after: str = Form(default="after", description="Display label for after image"),
):
    prefix_b = f"upload_{uuid.uuid4().hex[:8]}_"
    prefix_a = f"upload_{uuid.uuid4().hex[:8]}_"

    tile_id_b, arr_b = await _process_upload_file(image_before, modality_hint_before, tile_id_prefix=prefix_b)
    tile_id_a, arr_a = await _process_upload_file(image_after, modality_hint_after, tile_id_prefix=prefix_a)

    spec_before = _resolve_modality(arr_b, modality_hint_before)
    spec_after = _resolve_modality(arr_a, modality_hint_after)

    res = ChangeDetectionService.compare(
        arr_b,
        arr_a,
        co_registered=co_registered,
        spec_before=spec_before,
        spec_after=spec_after,
        label_before=label_before,
        label_after=label_after,
    )

    return ChangeDetectionResponse(
        tile_id_before=tile_id_b,
        tile_id_after=tile_id_a,
        pair_type=res["pair_type"],
        co_registered=res["co_registered"],
        warning=res.get("warning"),
        image_before=res["image_before"],
        image_after=res["image_after"],
        pixel_change=res.get("pixel_change"),
    )



@router.post("/api/v1/agent/chat", response_model=AgentChatResponse)
def agent_chat(req: AgentChatRequest):
    arr = _load_tile_tensor(req.tile_id)
    res = _global_agent.chat(
        arr,
        req.query,
        tile_id=req.tile_id,
        session_id=req.session_id,
    )
    return AgentChatResponse(
        tile_id=req.tile_id,
        session_id=res.get("session_id"),
        query=res["query"],
        response=res["response"],
        answer=res.get("answer", res["response"]),
        confidence=res.get("confidence", 1.0),
        visual_evidence=res.get("visual_evidence"),
        selected_task=res.get("selected_task", "analysis"),
        selected_model_or_tool=res.get("selected_model_or_tool", "agent"),
        detected_modality=res.get("detected_modality", "RGB_OPTICAL"),
        execution_trace=res.get("execution_trace", []),
        plan=res.get("plan", []),
        tool_artifacts=res.get("tool_artifacts", {}),
        history_length=res.get("history_length", 1),
    )


# ---------------------------------------------------------------------------
# Structured Agentic /analyze Endpoint (Multipart for Direct Testing & Swagger)
# ---------------------------------------------------------------------------

@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    tags=["Primary Analysis"],
    summary="Unified Earth Observation Analysis (Single Entrypoint)",
)
async def analyze_multipart(
    files: List[UploadFile] = File(
        ...,
        description="One or more satellite or aerial image files (.jpg, .jpeg, .png, .tif, .tiff, .npy, .npz)",
    ),
    query: str = Form(
        ...,
        description="Natural-language query about the satellite image (e.g. 'Where are the water bodies?')",
    ),
    session_id: Optional[str] = Form(
        default=None,
        description="Optional session ID to maintain conversation memory",
    ),
):
    """
    Main SatQuery AI Analysis Endpoint (multipart/form-data):
    Upload one or more EO images (.jpg, .png, .tif, .npy) and enter a natural-language query.
    Routes to the appropriate specialist tool, validates sensor physical capabilities,
    maintains per-session conversational memory, and returns answer + confidence + visual evidence + auditable execution trace.
    """
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="At least one image file must be provided.")

    clean_query = query.strip() if query else ""
    if not clean_query:
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    valid_exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".npy", ".npz", ".bmp", ".webp")
    arrays = []

    for f in files:
        fname = f.filename or "upload.png"
        fname_lower = fname.lower()
        if not fname_lower.endswith(valid_exts):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format for '{fname}'. Supported: {list(valid_exts)}",
            )

        content = await f.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail=f"Uploaded file '{fname}' is empty.")

        try:
            arr = standardize_image_input(content)
            arrays.append(arr)
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Failed to process and standardize image '{fname}': {exc}",
            )

    try:
        if len(arrays) == 1:
            res = _global_agent.chat(
                tensor_12ch=arrays[0],
                query=clean_query,
                tile_id="uploaded_tile",
                session_id=session_id,
            )
        else:
            active_sid, active_mem = _global_agent.session_store.get_or_create(session_id)
            active_mem.add_user_message(clean_query, tile_id="uploaded_pair")
            res = _global_agent.analyze(
                images=arrays,
                query=clean_query,
            )
            res["session_id"] = active_sid
            selected_task = res.get("selected_task", "analysis")
            active_mem.add_agent_message(
                text=res["answer"],
                tools_used=[selected_task],
                artifacts=res.get("visual_evidence") or {},
                tile_id="uploaded_pair",
            )
            res["history_length"] = len(active_mem.get_history())
    except ModalityError as me:
        raise HTTPException(status_code=422, detail=str(me))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal analysis failure: {exc}")

    return AnalyzeResponse(
        answer=res["answer"],
        confidence=res["confidence"],
        visual_evidence=res.get("visual_evidence") or {},
        selected_task=res["selected_task"],
        selected_model_or_tool=res["selected_model_or_tool"],
        detected_modality=res["detected_modality"],
        execution_trace=res["execution_trace"],
        session_id=res.get("session_id"),
        history_length=res.get("history_length"),
    )


@router.post("/api/v1/analyze", response_model=AnalyzeResponse)
def analyze_json(req: AnalyzeRequest):
    """
    Programmatic JSON analysis endpoint for stored sample tiles with conversational memory support.
    """
    if req.tile_ids and len(req.tile_ids) >= 2:
        arrays = [_load_tile_tensor(t) for t in req.tile_ids[:2]]
        hints = req.modality_hints
        tile_label = f"{req.tile_ids[0]}+{req.tile_ids[1]}"
        active_sid, active_mem = _global_agent.session_store.get_or_create(req.session_id)
        active_mem.add_user_message(req.query, tile_id=tile_label)
        res = _global_agent.analyze(
            images=arrays,
            query=req.query,
            modality_hints=hints,
            tile_id=tile_label,
        )
        res["session_id"] = active_sid
        selected_task = res.get("selected_task", "analysis")
        active_mem.add_agent_message(
            text=res["answer"],
            tools_used=[selected_task],
            artifacts=res.get("visual_evidence") or {},
            tile_id=tile_label,
        )
        res["history_length"] = len(active_mem.get_history())
    elif req.tile_id:
        arrays = [_load_tile_tensor(req.tile_id)]
        res = _global_agent.chat(
            tensor_12ch=arrays[0],
            query=req.query,
            tile_id=req.tile_id,
            session_id=req.session_id,
        )
    else:
        raise HTTPException(status_code=400, detail="Must provide either 'tile_id' or 'tile_ids'.")

    return AnalyzeResponse(
        answer=res["answer"],
        confidence=res["confidence"],
        visual_evidence=res.get("visual_evidence") or {},
        selected_task=res["selected_task"],
        selected_model_or_tool=res["selected_model_or_tool"],
        detected_modality=res["detected_modality"],
        execution_trace=res["execution_trace"],
        session_id=res.get("session_id"),
        history_length=res.get("history_length"),
    )


# ---------------------------------------------------------------------------
# Upload tile (accepts standard images: PNG, JPG, JPEG, TIFF or NPY/NPZ arrays)
# ---------------------------------------------------------------------------

@router.post("/api/v1/upload-tile", response_model=UploadTileResponse)
async def upload_tile(
    file: UploadFile = File(...),
    modality_hint: Optional[str] = Query(default=None),
    session_id: Optional[str] = Query(default=None),
):
    """
    Upload a satellite tile or aerial photo. Accepts .npy, .npz, .png, .jpg, .jpeg, .tif.
    Automatically standardizes input and detects sensor modality.
    """
    tile_id, arr = await _process_upload_file(file, modality_hint=modality_hint, tile_id_prefix=None)
    spec = detect_modality(arr, user_hint=modality_hint)

    active_sid, _ = _global_agent.session_store.get_or_create(session_id)

    return UploadTileResponse(
        status="success",
        tile_id=tile_id,
        shape=list(arr.shape),
        modality=spec.modality.value,
        detection_basis=spec.detection_basis,
        basis_label=spec.basis_label,
        display_label=spec.display_label,
        available_analyses=spec.available_analyses,
        session_id=active_sid,
    )


# ---------------------------------------------------------------------------
# Session & Memory Management Endpoints
# ---------------------------------------------------------------------------

@router.get("/api/v1/agent/session/{session_id}/history", response_model=SessionHistoryResponse)
def get_session_history(session_id: str):
    """
    Retrieve stored conversational message history for a given session ID (without heavy binary artifacts).
    """
    mem = _global_agent.session_store.get(session_id)
    if not mem:
        return SessionHistoryResponse(
            session_id=session_id,
            history_length=0,
            messages=[],
        )
    raw_history = mem.get_history_clean()
    return SessionHistoryResponse(
        session_id=session_id,
        history_length=len(raw_history),
        messages=[MessageHistoryItem(**m) for m in raw_history],
    )


@router.delete("/api/v1/agent/session/{session_id}", response_model=SessionDeleteResponse)
def delete_session(session_id: str):
    """
    Clear and remove the conversational memory associated with a session ID.
    """
    deleted = _global_agent.session_store.delete(session_id)
    return SessionDeleteResponse(
        session_id=session_id,
        status="deleted" if deleted else "not_found",
        message="Session memory cleared." if deleted else "Session not found or already expired.",
    )


