"""
satquery/agent/sat_agent.py
───────────────────────────
Autonomous Earth Observation Intelligence Agent.
Features query-driven routing across multimodal specialists, capability guards,
grounding/bounding box localization, temporal change detection, and auditable execution tracing.
"""

from typing import Dict, Any, Union, List, Optional
import numpy as np
import torch

from backend.data.modality import (
    InputModality,
    ModalitySpec,
    detect_modality,
    resolve_pair_modality,
    standardize_image_input,
)
from backend.agent.tools import ToolRegistry, Tool
from backend.agent.memory import ConversationMemory


class SatQueryAgent:
    """
    Agentic Router & Multi-Modal Intelligence Engine for Earth Observation.
    Routes queries to dedicated specialist tools based on detected modality,
    sensor bands, and query intent while enforcing physical sensor constraints.
    """

    def __init__(self, memory: Optional[ConversationMemory] = None):
        self.registry = ToolRegistry()
        self.memory = memory or ConversationMemory()

    def analyze(
        self,
        images: Union[Any, List[Any]],
        query: str,
        modality_hints: Optional[Union[str, List[str]]] = None,
        tile_id: str = "active_tile",
    ) -> Dict[str, Any]:
        """
        Unified Agentic Analysis Pipeline:
        IMAGE(S) + NATURAL LANGUAGE QUERY -> AGENTIC ROUTER -> SPECIALIST TOOL -> ANSWER + VISUAL EVIDENCE + TRACE
        """
        trace: List[str] = []

        # ----------------------------------------------------------------------
        # 1. Input Inspection & Standardization
        # ----------------------------------------------------------------------
        if isinstance(images, list):
            raw_list = images
        else:
            raw_list = [images]

        parsed_arrays = [standardize_image_input(img) for img in raw_list]
        image_count = len(parsed_arrays)
        shapes = [list(a.shape) for a in parsed_arrays]
        trace.append(f"Input Validation: Received {image_count} image(s) with tensor shape(s): {shapes}.")

        # ----------------------------------------------------------------------
        # 2. Modality Verification & Validation
        # ----------------------------------------------------------------------
        hint_a = modality_hints[0] if isinstance(modality_hints, list) and len(modality_hints) > 0 else (modality_hints if isinstance(modality_hints, str) else None)
        hint_b = modality_hints[1] if isinstance(modality_hints, list) and len(modality_hints) > 1 else None

        specs = [detect_modality(arr, user_hint=(hint_b if i == 1 else hint_a)) for i, arr in enumerate(parsed_arrays)]

        if image_count == 2:
            pair_modality = resolve_pair_modality(specs[0], specs[1])
            detected_modality = pair_modality.value
            trace.append(
                f"Modality Verification: Multi-image input resolved to '{pair_modality.value}' "
                f"(Image A: {specs[0].modality.value}, Image B: {specs[1].modality.value})."
            )
        else:
            detected_modality = specs[0].modality.value
            trace.append(
                f"Modality Verification: Detected modality '{detected_modality}' "
                f"via {specs[0].basis_label}."
            )

        # ----------------------------------------------------------------------
        # 3. True Query-Driven Intent Extraction
        # ----------------------------------------------------------------------
        q_lower = query.lower().strip()
        selected_task, target_concept = self._classify_intent(q_lower, image_count, specs)
        trace.append(f"Intent Extraction: Parsed query intent as '{selected_task}' (target concept: '{target_concept}').")

        # ----------------------------------------------------------------------
        # 4. Modality Capability & Physical Sensor Integrity Check
        # ----------------------------------------------------------------------
        primary_arr = parsed_arrays[0]

        # Rule: Physical spectral indices (NDVI/NDWI/NDBI) cannot be run on standard RGB
        if selected_task == "spectral_analysis":
            if specs[0].modality == InputModality.RGB_OPTICAL:
                trace = [
                    "Input validated",
                    f"{detected_modality} detected",
                    "Query classified as spectral index analysis",
                    "SensorIntegrityGuard executed: NDVI/NDWI/NDBI rejected on RGB optical imagery",
                ]
                return {
                    "answer": (
                        "⚠️ Capability Limitation: Calculating physical remote sensing indices (NDVI for vegetation, "
                        "NDWI for water, NDBI for built-up) requires Sentinel-2 Near-Infrared (B08, 842 nm) and Shortwave Infrared "
                        "(B11, 1610 nm) bands. Standard 3-channel RGB optical imagery does not capture these physical wavelengths. "
                        "SatQuery enforces scientific sensor integrity and refuses to fabricate fake spectral indices from ordinary RGB. "
                        "To assess vegetation or water in this RGB image, ask for object grounding, green canopy coverage, or upload a multispectral Sentinel-2 tile."
                    ),
                    "confidence": 1.0,
                    "visual_evidence": {},
                    "selected_task": "spectral_analysis",
                    "selected_model_or_tool": "SensorIntegrityGuard",
                    "detected_modality": detected_modality,
                    "execution_trace": trace,
                }
            if specs[0].modality == InputModality.SAR_ONLY:
                trace = [
                    "Input validated",
                    f"{detected_modality} detected",
                    "Query classified as spectral index analysis",
                    "SensorIntegrityGuard executed: Optical spectral indices rejected on SAR radar imagery",
                ]
                return {
                    "answer": (
                        "⚠️ Capability Limitation: SAR (Synthetic Aperture Radar) measures microwave backscatter (VH/VV), "
                        "which cannot calculate optical NDVI/NDWI indices. Use SAR backscatter analysis instead."
                    ),
                    "confidence": 1.0,
                    "visual_evidence": {},
                    "selected_task": "spectral_analysis",
                    "selected_model_or_tool": "SensorIntegrityGuard",
                    "detected_modality": detected_modality,
                    "execution_trace": trace,
                }

        # Rule: Standalone SAR cannot be fed into 12-channel S1+S2 model
        if selected_task == "classification" and (specs[0].modality == InputModality.SAR_ONLY or primary_arr.shape[0] in (1, 2)):
            trace = [
                "Input validated",
                f"{detected_modality} detected",
                "Query classified as land-cover classification",
                "SensorIntegrityGuard executed: Standalone SAR cannot be fed into 12-channel optical+SAR classifier",
            ]
            return {
                "answer": (
                    "⚠️ Capability Limitation: Standalone SAR (Synthetic Aperture Radar) data cannot be processed by the "
                    "12-channel BigEarthNet land-cover classification model, which requires Sentinel-2 optical/multispectral bands (B02–B12). "
                    "SatQuery enforces sensor integrity and refuses to pass standalone SAR data into an S1+S2 12-channel model. "
                    "For standalone SAR, please request radar backscatter and polarization analysis, or provide paired optical imagery."
                ),
                "confidence": 1.0,
                "visual_evidence": {},
                "selected_task": "classification",
                "selected_model_or_tool": "SensorIntegrityGuard",
                "detected_modality": detected_modality,
                "execution_trace": trace,
            }

        # Rule: SAR analysis on pure optical
        if selected_task == "sar_analysis" and specs[0].modality in (InputModality.RGB_OPTICAL, InputModality.SENTINEL2_MULTISPECTRAL):
            trace = [
                "Input validated",
                f"{detected_modality} detected",
                "Query classified as SAR radar backscatter analysis",
                "SensorIntegrityGuard executed: Radar analysis rejected on optical imagery",
            ]
            return {
                "answer": (
                    "⚠️ Capability Limitation: SAR backscatter analysis requires Sentinel-1 radar channels (VH/VV). "
                    "This image is optical and contains no microwave radar data."
                ),
                "confidence": 1.0,
                "visual_evidence": {},
                "selected_task": "sar_analysis",
                "selected_model_or_tool": "SensorIntegrityGuard",
                "detected_modality": detected_modality,
                "execution_trace": trace,
            }

        # ----------------------------------------------------------------------
        # 5. Specialist Tool Selection & Execution
        # ----------------------------------------------------------------------
        primary_arr = parsed_arrays[0]

        if selected_task == "change_detection":
            tool_name = "change_detection"
            if image_count < 2:
                trace = [
                    "Input validated",
                    f"{detected_modality} detected",
                    f"Query classified as {selected_task}",
                    "ChangeDetectionService executed: Image pair required",
                ]
                return {
                    "answer": "⚠️ Temporal change detection requires two images (Before and After). Please upload an image pair to analyze change.",
                    "confidence": 1.0,
                    "visual_evidence": {},
                    "selected_task": selected_task,
                    "selected_model_or_tool": tool_name,
                    "detected_modality": detected_modality,
                    "execution_trace": trace,
                }
            change_res = self.registry.get_tool(tool_name).execute(parsed_arrays[0], parsed_arrays[1], co_registered=True)
            answer_text = (
                f"Temporal change analysis between Image A and Image B completed. "
                f"Evaluated {change_res['image_before']['shape'][0]}-channel images. "
            )
            if change_res.get("pixel_change"):
                cm = change_res["pixel_change"]["change_magnitude"]
                answer_text += f"Mean change magnitude: {cm['mean']:.4f}, maximum change: {cm['max']:.4f}."

            change_vis = {"type": "change_metrics", "metrics": change_res.get("pixel_change")}
            try:
                arr_b, arr_a = parsed_arrays[0], parsed_arrays[1]
                if arr_b.shape == arr_a.shape:
                    import io, base64
                    from PIL import Image, ImageDraw
                    from backend.data.modality import to_rgb_image

                    pil_b = to_rgb_image(arr_b).resize((200, 200))
                    pil_a = to_rgb_image(arr_a).resize((200, 200))

                    diff_raw = np.mean(np.abs(arr_a.astype(np.float32) - arr_b.astype(np.float32)), axis=0)
                    p98 = float(np.percentile(diff_raw, 98)) or 1.0
                    diff_norm = np.clip(diff_raw / p98, 0.0, 1.0)

                    heat = np.zeros((diff_norm.shape[0], diff_norm.shape[1], 3), dtype=np.uint8)
                    heat[..., 0] = (diff_norm * 255).astype(np.uint8)
                    heat[..., 1] = ((1.0 - np.abs(diff_norm - 0.5) * 2) * 180).astype(np.uint8)
                    heat[..., 2] = ((1.0 - diff_norm) * 220).astype(np.uint8)
                    pil_diff = Image.fromarray(heat).resize((200, 200))

                    composite = Image.new("RGB", (620, 230), (24, 24, 28))
                    composite.paste(pil_b, (10, 25))
                    composite.paste(pil_a, (215, 25))
                    composite.paste(pil_diff, (415, 25))

                    draw = ImageDraw.Draw(composite)
                    draw.text((10, 8), "Before Image", fill=(200, 200, 200))
                    draw.text((215, 8), "After Image", fill=(200, 200, 200))
                    draw.text((415, 8), "Change Magnitude Map", fill=(255, 100, 100))

                    buf = io.BytesIO()
                    composite.save(buf, format="PNG")
                    change_vis["image_base64"] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
            except Exception:
                pass

            trace = [
                "Input validated",
                "Two temporally related images detected",
                "Query classified as temporal change detection",
                "ChangeDetectionService executed",
            ]
            return {
                "answer": answer_text,
                "confidence": 0.90,
                "visual_evidence": change_vis,
                "selected_task": selected_task,
                "selected_model_or_tool": tool_name,
                "detected_modality": detected_modality,
                "execution_trace": trace,
            }

        elif selected_task == "cross_modal_analysis":
            tool_name = "cross_modal_analysis"
            if image_count < 2:
                trace = [
                    "Input validated",
                    f"{detected_modality} detected",
                    "Query classified as cross-modal analysis",
                    "CrossModalService executed: Pair required",
                ]
                return {
                    "answer": "⚠️ Cross-modal analysis requires two images (one Optical and one SAR). Please provide both modalities to perform joint reasoning.",
                    "confidence": 1.0,
                    "visual_evidence": {},
                    "selected_task": selected_task,
                    "selected_model_or_tool": tool_name,
                    "detected_modality": detected_modality,
                    "execution_trace": trace,
                }
            if specs[0].modality in (InputModality.RGB_OPTICAL, InputModality.SENTINEL2_MULTISPECTRAL):
                opt_idx, sar_idx = 0, 1
            else:
                opt_idx, sar_idx = 1, 0
            cm_res = self.registry.get_tool(tool_name).execute(parsed_arrays[opt_idx], parsed_arrays[sar_idx], query=query)
            trace = [
                "Input validated",
                "Optical + SAR pair detected",
                "Query classified as cross-modal analysis",
                "CrossModalService executed",
            ]
            return {
                "answer": f"### Cross-Modal Optical + SAR Analysis\n{cm_res['joint_reasoning']}",
                "confidence": cm_res["confidence"],
                "visual_evidence": cm_res["visual_evidence"],
                "selected_task": selected_task,
                "selected_model_or_tool": tool_name,
                "detected_modality": detected_modality,
                "execution_trace": trace,
            }

        elif selected_task == "grounding":
            tool_name = "grounding_localization"
            ground_res = self.registry.get_tool(tool_name).execute(primary_arr, target_concept=target_concept)
            cnt = ground_res["count"]
            cov = ground_res["coverage_percent"]
            answer_text = (
                f"Grounding analysis for **'{target_concept}'** identified **{cnt}** candidate region(s) "
                f"accounting for **{cov}%** of total surface area with mean confidence of **{ground_res['confidence']*100:.1f}%**."
            )
            trace = [
                "Input validated",
                f"{detected_modality} detected",
                f"Query classified as {target_concept} detection",
                "GroundingService executed",
            ]
            return {
                "answer": answer_text,
                "confidence": ground_res["confidence"],
                "visual_evidence": ground_res["visual_evidence"],
                "selected_task": selected_task,
                "selected_model_or_tool": tool_name,
                "detected_modality": detected_modality,
                "execution_trace": trace,
            }

        elif selected_task == "spectral_analysis":
            tool_name = "spectral_analysis"
            spec_res = self.registry.get_tool(tool_name).execute(primary_arr)
            cov = spec_res["coverage"]
            def _fmt_stat(v):
                return f"{v:.3f}" if v is not None else "N/A"
            def _fmt_pct(v):
                return f"{v}%" if v is not None else "N/A"

            answer_text = (
                f"### Multispectral Indices\n"
                f"- **NDVI (Vegetation)**: Mean `{_fmt_stat(spec_res.get('ndvi_mean'))}` | Dense canopy: `{_fmt_pct(cov.get('dense_vegetation_percent'))}`\n"
                f"- **NDWI (Water)**: Mean `{_fmt_stat(spec_res.get('ndwi_mean'))}` | Surface moisture/water: `{_fmt_pct(cov.get('water_body_percent'))}`\n"
                f"- **NDBI (Built-up)**: Mean `{_fmt_stat(spec_res.get('ndbi_mean'))}` | Impervious surface: `{_fmt_pct(cov.get('builtup_percent'))}`\n\n"
                f"**Ecosystem Evaluation**: {spec_res.get('assessment', 'N/A')}"
            )
            trace = [
                "Input validated",
                f"{detected_modality} detected",
                "Query classified as multispectral index analysis",
                "SpectralService executed",
            ]
            return {
                "answer": answer_text,
                "confidence": 0.94,
                "visual_evidence": {"type": "spectral_metrics", "metrics": spec_res},
                "selected_task": selected_task,
                "selected_model_or_tool": "SpectralService",
                "detected_modality": detected_modality,
                "execution_trace": trace,
            }

        elif selected_task == "sar_analysis":
            tool_name = "sar_analysis"
            sar_res = self.registry.get_tool(tool_name).execute(primary_arr)
            ch_info = [f"Channel {ch['label']}: mean {ch['mean']:.4f} (std {ch['std']:.4f})" for ch in sar_res["channels"]]
            answer_text = (
                f"### Sentinel-1 SAR Backscatter Analysis\n"
                + "\n".join([f"- {c}" for c in ch_info])
            )
            if sar_res.get("polarization_ratio"):
                pr = sar_res["polarization_ratio"]
                answer_text += f"\n- **Polarization Ratio (VH/VV)**: Mean {pr['mean_ratio']:.4f} (depolarization indicator of volumetric canopy roughness)."
            trace = [
                "Input validated",
                f"{detected_modality} detected",
                "Query classified as SAR radar backscatter analysis",
                "SARService executed",
            ]
            return {
                "answer": answer_text,
                "confidence": 0.91,
                "visual_evidence": {"type": "sar_statistics", "metrics": sar_res},
                "selected_task": selected_task,
                "selected_model_or_tool": "SARService",
                "detected_modality": detected_modality,
                "execution_trace": trace,
            }

        elif selected_task == "classification":
            tool_name = "landcover_classification"
            cls_res = self.registry.get_tool(tool_name).execute(primary_arr)
            top_preds = ", ".join([f"{p['class_name']} ({p['confidence']*100:.1f}%)" for p in cls_res["top_5"][:3]])
            answer_text = (
                f"Evaluated land-cover as **{cls_res['primary_class']}** "
                f"(confidence: **{cls_res['confidence']*100:.1f}%**; Model: `{cls_res.get('model_used', 'Classifier')}`). "
                f"Top predictions: {top_preds}."
            )
            model_label = cls_res.get("model_used", tool_name)
            trace = [
                "Input validated",
                f"{detected_modality} detected",
                "Query classified as land-cover classification",
                f"{model_label} executed",
            ]
            return {
                "answer": answer_text,
                "confidence": cls_res["confidence"],
                "visual_evidence": {"type": "class_probabilities", "classes": cls_res["top_5"]},
                "selected_task": selected_task,
                "selected_model_or_tool": model_label,
                "detected_modality": detected_modality,
                "execution_trace": trace,
            }

        elif selected_task == "rgb_scene_analysis":
            tool_name = "rgb_scene_analysis"
            rgb_res = self.registry.get_tool(tool_name).execute(primary_arr)
            trace = [
                "Input validated",
                f"{detected_modality} detected",
                "Query classified as RGB scene description",
                "RGBVisionService executed",
            ]
            return {
                "answer": rgb_res["scene_description"],
                "confidence": rgb_res["confidence"],
                "visual_evidence": {"type": "feature_coverages", "coverages": rgb_res["feature_coverages"]},
                "selected_task": selected_task,
                "selected_model_or_tool": "RGBVisionService",
                "detected_modality": detected_modality,
                "execution_trace": trace,
            }

        else:
            # VQA & open-ended scene reasoning fallback
            tool_name = "visual_qa"
            vlm_res = self.registry.get_tool(tool_name).execute(primary_arr, question=query)
            ans = vlm_res.get("answer", "Multimodal evaluation complete.")
            trace = [
                "Input validated",
                f"{detected_modality} detected",
                "Query classified as visual QA reasoning",
                "VLMService executed",
            ]
            return {
                "answer": ans,
                "confidence": vlm_res.get("confidence", 0.85),
                "visual_evidence": {},
                "selected_task": "vqa",
                "selected_model_or_tool": "VLMService",
                "detected_modality": detected_modality,
                "execution_trace": trace,
            }

    def _classify_intent(
        self,
        query: str,
        image_count: int,
        specs: List[ModalitySpec],
    ) -> (str, str):
        """
        True query-driven intent router for arbitrary concepts and questions.
        Considers image count, detected modality, and query text.
        """
        q_clean = query.lower().replace("-", " ").strip()

        # 1. Multi-image pair intents
        if image_count >= 2:
            pair_mod = resolve_pair_modality(specs[0], specs[1])
            if any(w in q_clean for w in ["change", "diff", "before", "after", "temporal", "growth", "loss", "changed"]):
                return "change_detection", "temporal_pair"
            if pair_mod == InputModality.CROSS_MODAL_PAIR or any(w in q_clean for w in ["sar", "radar", "optical", "cross", "fuse", "joint"]):
                return "cross_modal_analysis", "optical_sar_joint"
            return "change_detection", "temporal_pair"

        # 2. Physical spectral indices intent
        if any(w in q_clean for w in ["ndvi", "ndwi", "ndbi", "vegetation index", "water index", "built up index", "spectral index", "spectral indices"]):
            return "spectral_analysis", "multispectral_indices"

        # 3. Standalone SAR-specific queries
        if specs[0].modality == InputModality.SAR_ONLY:
            if any(w in q_clean for w in ["land cover", "what class", "classify", "classification", "terrain type", "category"]):
                return "classification", "land_cover"
            if any(w in q_clean for w in ["sar", "radar", "polarization", "backscatter", "vh", "vv", "surface roughness", "structure", "structures", "analyze", "describe"]):
                return "sar_analysis", "radar_backscatter"

        # 4. General SAR radar intent
        if any(w in q_clean for w in ["sar", "radar", "polarization", "backscatter", "vh", "vv", "surface roughness"]):
            return "sar_analysis", "radar_backscatter"

        # 5. Spatial grounding / localization / counting intent for ANY object or region
        grounding_triggers = [
            "where", "locate", "find", "count", "how many", "box", "boxes",
            "bounding", "highlight", "show me", "detect", "region", "regions",
            "position", "positions", "presence of", "spot"
        ]
        if any(w in q_clean for w in grounding_triggers):
            concept = self._extract_target_concept(q_clean)
            return "grounding", concept

        # 6. Land cover classification intent
        if any(w in q_clean for w in ["land cover", "what class", "classify", "classification", "terrain type", "what is this land", "category", "land types"]):
            return "classification", "land_cover"

        # 7. Specific feature inquiries that imply grounding or detection
        feature_words = ["water", "river", "lake", "tree", "forest", "building", "house", "road", "crop", "agriculture", "field"]
        for f in feature_words:
            if f in q_clean and any(q_word in q_clean for q_word in ["is there", "are there", "any", "look for", "identify"]):
                return "grounding", f

        # 8. RGB scene description if 3-channel
        if specs[0].modality == InputModality.RGB_OPTICAL and any(w in q_clean for w in ["describe", "scene", "overview", "what is in", "summary"]):
            return "rgb_scene_analysis", "scene_description"

        # 9. Default to VQA
        return "vqa", "scene_reasoning"

    def _extract_target_concept(self, query: str) -> str:
        """Extract the target entity/concept from a grounding query."""
        q = query.lower()
        candidates = [
            ("water bodies", ["water bodies", "water body", "water", "river", "lake", "ocean", "sea", "pond", "canal"]),
            ("trees/forest", ["tree", "trees", "forest", "forests", "woodland", "canopy", "vegetation", "greenery"]),
            ("buildings", ["building", "buildings", "house", "houses", "structure", "structures", "urban", "roof", "roofs"]),
            ("roads", ["road", "roads", "highway", "highways", "street", "streets", "path", "corridor", "runway"]),
            ("agriculture", ["agriculture", "agricultural", "crop", "crops", "farm", "farms", "field", "fields", "farmland"]),
            ("bare soil", ["bare soil", "sand", "dune", "dunes", "soil"]),
            ("vehicles", ["vehicle", "vehicles", "car", "cars", "truck", "trucks", "ship", "ships", "boat", "boats"]),
        ]
        for name, keywords in candidates:
            if any(k in q for k in keywords):
                return name

        # Fallback: clean prompt to find the noun
        for remove_word in ["where are the", "where is the", "locate the", "find the", "count the", "how many", "highlight the", "show me"]:
            if remove_word in q:
                return q.split(remove_word)[-1].strip(" ?.,!")

        return "regions of interest"

    # --------------------------------------------------------------------------
    # Backward compatibility helpers
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # Backward compatibility helpers
    # --------------------------------------------------------------------------
    def determine_plan(self, query: str) -> List[str]:
        q = query.lower()
        tools_to_run = []
        if any(w in q for w in ["ndvi", "ndwi", "ndbi", "vegetation", "water", "drought", "green", "health", "spectral"]):
            tools_to_run.append("spectral_analysis")
        if any(w in q for w in ["class", "land cover", "what is this", "identify", "type", "forest", "urban", "crop"]):
            tools_to_run.append("landcover_classification")
        if any(w in q for w in ["band", "wavelength", "b0", "b1", "b8a"]):
            tools_to_run.append("band_metadata")
        if not tools_to_run or any(w in q for w in ["why", "explain", "question", "how", "describe"]):
            tools_to_run.append("visual_qa")
        if "spectral_analysis" not in tools_to_run and "landcover_classification" not in tools_to_run:
            tools_to_run.insert(0, "spectral_analysis")
        return tools_to_run

    def execute_plan(
        self,
        tensor_12ch: Union[torch.Tensor, np.ndarray],
        query: str,
        plan: List[str],
    ) -> Dict[str, Any]:
        results = {}
        for tool_name in plan:
            tool = self.registry.get_tool(tool_name)
            if not tool:
                continue

            if tool_name == "visual_qa":
                results[tool_name] = tool.execute(tensor_12ch, query)
            elif tool_name == "band_metadata":
                band_match = None
                for b in ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]:
                    if b.lower() in query.lower():
                        band_match = b
                        break
                results[tool_name] = tool.execute(band_match)
            else:
                results[tool_name] = tool.execute(tensor_12ch)

        return results

    def synthesize_response(self, query: str, tool_outputs: Dict[str, Any]) -> str:
        sections = []
        if "landcover_classification" in tool_outputs:
            cls_data = tool_outputs["landcover_classification"]
            primary_class = cls_data["primary_class"]
            conf = cls_data["confidence"] * 100.0
            sections.append(
                f"### Executive Finding\n"
                f"The evaluated Sentinel-2 tile is characterized predominantly as **{primary_class}** "
                f"(confidence: **{conf:.1f}%**)."
            )

        if "spectral_analysis" in tool_outputs:
            spec = tool_outputs["spectral_analysis"]
            cov = spec.get("coverage", {})
            def _fmt_val(v):
                return f"{v:.3f}" if v is not None else "N/A"
            def _fmt_p(v):
                return f"{v}%" if v is not None else "N/A"

            sections.append(
                f"### Spectral Index Analysis\n"
                f"- **NDVI (Vegetation Index)**: Mean `{_fmt_val(spec.get('ndvi_mean'))}` | Dense Canopy: `{_fmt_p(cov.get('dense_vegetation_percent'))}`\n"
                f"- **NDWI (Water Index)**: Mean `{_fmt_val(spec.get('ndwi_mean'))}` | Hydrological Surface: `{_fmt_p(cov.get('water_body_percent'))}`\n"
                f"- **NDBI (Built-up Index)**: Mean `{_fmt_val(spec.get('ndbi_mean'))}` | Urban/Impervious: `{_fmt_p(cov.get('builtup_percent'))}`\n"
                f"- **Ecosystem Assessment**: {spec.get('assessment', '')}"
            )

        if "visual_qa" in tool_outputs:
            vlm_res = tool_outputs["visual_qa"]
            sections.append(
                f"### SatQuery VLM Insights\n"
                f"{vlm_res.get('answer', '')}"
            )

        if "band_metadata" in tool_outputs:
            b_info = tool_outputs["band_metadata"]
            sections.append(
                f"### Optical Band Reference\n"
                f"```json\n{b_info}\n```"
            )

        return "\n\n".join(sections)

    def chat(
        self,
        tensor_12ch: Union[torch.Tensor, np.ndarray],
        query: str,
        tile_id: str = "current_tile",
    ) -> Dict[str, Any]:
        """
        Agent chat method executing unified analysis pipeline and maintaining conversation history.
        """
        self.memory.add_user_message(query, tile_id=tile_id)
        res = self.analyze(images=tensor_12ch, query=query, tile_id=tile_id)

        selected_task = res.get("selected_task", "analysis")
        selected_tool = res.get("selected_model_or_tool", "agent")

        tool_artifacts = {
            selected_task: res.get("visual_evidence") or {},
            "visual_evidence": res.get("visual_evidence") or {},
            "trace": res.get("execution_trace", []),
            "confidence": res.get("confidence", 1.0),
            "selected_tool": selected_tool,
        }
        if isinstance(res.get("visual_evidence"), dict):
            tool_artifacts.update(res["visual_evidence"])

        self.memory.add_agent_message(
            text=res["answer"],
            tools_used=[selected_task],
            artifacts=tool_artifacts,
        )

        return {
            "tile_id": tile_id,
            "query": query,
            "response": res["answer"],
            "answer": res["answer"],
            "confidence": res.get("confidence", 1.0),
            "visual_evidence": res.get("visual_evidence"),
            "selected_task": selected_task,
            "selected_model_or_tool": selected_tool,
            "detected_modality": res.get("detected_modality", "RGB_OPTICAL"),
            "execution_trace": res.get("execution_trace", []),
            "plan": [selected_task],
            "tool_artifacts": tool_artifacts,
            "history_length": len(self.memory.get_history()),
        }
