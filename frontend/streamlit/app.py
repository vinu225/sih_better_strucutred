import io
import base64
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
import streamlit as st
import httpx

from backend.config import SAMPLES_DIR, API_HOST, API_PORT
from backend.data.sample_generator import generate_sample_tiles
from backend.data.modality import standardize_image_input, to_rgb_image
from backend.agent.sat_agent import SatQueryAgent

st.set_page_config(
    page_title="SatQuery AI | Unified EO Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main-header {
        font-size: 2.2rem; font-weight: 700;
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header { font-size: 1.05rem; color: #aaa; margin-bottom: 1.4rem; }
    .status-card {
        background: #181c24; border: 1px solid #2d3748; border-radius: 8px;
        padding: 14px 18px; margin-bottom: 16px;
    }
    .badge {
        display: inline-block; padding: 4px 12px; border-radius: 16px;
        font-weight: 600; font-size: 0.82rem; margin-right: 8px;
    }
    .badge-modality { background: #1a3a4a; color: #00C9FF; border: 1px solid #00C9FF; }
    .badge-task     { background: #2a1a3a; color: #DA70D6; border: 1px solid #DA70D6; }
    .badge-tool     { background: #1a3a2a; color: #92FE9D; border: 1px solid #92FE9D; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "agent" not in st.session_state:
    st.session_state.agent = SatQueryAgent()

if not SAMPLES_DIR.exists() or len(list(SAMPLES_DIR.glob("*.npy"))) == 0:
    generate_sample_tiles(SAMPLES_DIR)

tile_files = sorted(list(SAMPLES_DIR.glob("*.npy")))
tile_names = [f.stem for f in tile_files]

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🛰️ SatQuery AI: Unified Earth Observation Intelligence</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Single Unified Backend API: <code>POST /analyze</code>. '
    'Upload one or two satellite images and enter any natural-language query. '
    'The agentic router automatically determines modality, validates sensor capability, and invokes the specialist.</div>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Upload Section (1 or 2 images)
# ─────────────────────────────────────────────────────────────────────────────
col_img1, col_img2 = st.columns(2)

with col_img1:
    st.subheader("1. Primary Image (Required)")
    img1_src = st.radio("Source 1", ["Upload File", "Select Stored Sample"], horizontal=True, key="src1")
    upload_1 = None
    bytes_1 = None
    fname_1 = "image_1.png"

    if img1_src == "Upload File":
        upload_1 = st.file_uploader("Upload Image (.jpg, .png, .tif, .npy)", type=["jpg", "jpeg", "png", "tif", "tiff", "npy", "npz"], key="file1")
        if upload_1:
            bytes_1 = upload_1.getvalue()
            fname_1 = upload_1.name
    else:
        sample_1 = st.selectbox("Sample Tile", tile_names, index=0, key="samp1")
        if sample_1:
            arr1 = np.load(str(SAMPLES_DIR / f"{sample_1}.npy"))
            buf = io.BytesIO()
            np.save(buf, arr1)
            bytes_1 = buf.getvalue()
            fname_1 = f"{sample_1}.npy"

    if bytes_1:
        try:
            arr_preview1 = standardize_image_input(bytes_1)
            st.image(to_rgb_image(arr_preview1), caption=f"{fname_1} (Shape: {list(arr_preview1.shape)})", use_container_width=True)
        except Exception:
            pass

with col_img2:
    st.subheader("2. Secondary Image (Optional)")
    use_second = st.checkbox("Include second image (for Before/After Temporal Change or Optical+SAR Pair)", value=False)
    bytes_2 = None
    fname_2 = "image_2.png"

    if use_second:
        img2_src = st.radio("Source 2", ["Upload File", "Select Stored Sample"], horizontal=True, key="src2")
        if img2_src == "Upload File":
            upload_2 = st.file_uploader("Upload Second Image", type=["jpg", "jpeg", "png", "tif", "tiff", "npy", "npz"], key="file2")
            if upload_2:
                bytes_2 = upload_2.getvalue()
                fname_2 = upload_2.name
        else:
            sample_2 = st.selectbox("Sample Tile B", tile_names, index=min(1, len(tile_names)-1), key="samp2")
            if sample_2:
                arr2 = np.load(str(SAMPLES_DIR / f"{sample_2}.npy"))
                buf2 = io.BytesIO()
                np.save(buf2, arr2)
                bytes_2 = buf2.getvalue()
                fname_2 = f"{sample_2}.npy"

        if bytes_2:
            try:
                arr_preview2 = standardize_image_input(bytes_2)
                st.image(to_rgb_image(arr_preview2), caption=f"{fname_2} (Shape: {list(arr_preview2.shape)})", use_container_width=True)
            except Exception:
                pass

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Query Input & Quick Demo Queries
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("3. Natural-Language Query")

st.caption("Quick Demo Query Suggestions:")
btn_cols = st.columns(4)
btn_cols2 = st.columns(3)

preset_query = ""
if btn_cols[0].button("💧 Water Bodies (RGB)", use_container_width=True):
    preset_query = "Where are the water bodies?"
if btn_cols[1].button("🔍 Describe Scene (RGB)", use_container_width=True):
    preset_query = "Describe the major objects visible in this image."
if btn_cols[2].button("🏷️ Land Cover (Sentinel-2)", use_container_width=True):
    preset_query = "What land-cover types are present?"
if btn_cols[3].button("📡 SAR Structures (SAR)", use_container_width=True):
    preset_query = "Analyze the major structures in this SAR image."

if btn_cols2[0].button("🔀 Optical + SAR Fusion", use_container_width=True):
    preset_query = "Use both images to identify built-up and water-covered regions."
if btn_cols2[1].button("⏱️ Temporal Change", use_container_width=True):
    preset_query = "What changed between these two images?"
if btn_cols2[2].button("⚠️ Test NDVI Guard (RGB)", use_container_width=True):
    preset_query = "Calculate NDVI."

query_input = st.text_input(
    "Query",
    value=preset_query if preset_query else "",
    placeholder="e.g., Where are the water bodies? / What changed between these dates?",
    label_visibility="collapsed",
)

analyze_clicked = st.button("🚀 Analyze", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Execution & Unified Result Display
# ─────────────────────────────────────────────────────────────────────────────
if analyze_clicked:
    if not bytes_1:
        st.error("Please upload or select at least one primary image.")
    elif not query_input.strip():
        st.error("Please enter a natural-language query.")
    else:
        # Prepare multipart files for POST /analyze
        files_payload = [("files", (fname_1, bytes_1, "application/octet-stream"))]
        if bytes_2:
            files_payload.append(("files", (fname_2, bytes_2, "application/octet-stream")))

        backend_url = f"http://{API_HOST}:{API_PORT}/analyze"
        res_data = None

        with st.spinner("SatQuery Agent routing through unified POST /analyze..."):
            try:
                # Call unified endpoint POST /analyze via HTTP
                r = httpx.post(
                    backend_url,
                    files=files_payload,
                    data={"query": query_input.strip()},
                    timeout=60.0,
                )
                if r.status_code == 200:
                    res_data = r.json()
                else:
                    st.warning(f"Backend returned HTTP {r.status_code}: {r.text}. Executing agent directly.")
            except Exception as e:
                # Fallback to local agent instance if backend server is not running
                pass

            if res_data is None:
                # In-memory execution fallback
                arrays = [standardize_image_input(bytes_1)]
                if bytes_2:
                    arrays.append(standardize_image_input(bytes_2))
                res_data = st.session_state.agent.analyze(
                    images=arrays,
                    query=query_input.strip(),
                )

        # ── Display Unified Result ───────────────────────────────────────────
        st.success("✅ Analysis Complete")

        # Analytical Answer
        st.markdown("### 📋 Analytical Answer")
        st.markdown(res_data.get("answer", ""))

        # Metadata badges
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Confidence", f"{float(res_data.get('confidence', 0.85))*100:.1f}%")
        with m_col2:
            st.markdown(f"**Modality**<br><span class='badge badge-modality'>{res_data.get('detected_modality', 'N/A')}</span>", unsafe_allow_html=True)
        with m_col3:
            st.markdown(f"**Selected Task**<br><span class='badge badge-task'>{res_data.get('selected_task', 'N/A')}</span>", unsafe_allow_html=True)
        with m_col4:
            st.markdown(f"**Selected Specialist**<br><span class='badge badge-tool'>{res_data.get('selected_model_or_tool', 'N/A')}</span>", unsafe_allow_html=True)

        st.progress(float(res_data.get("confidence", 0.85)))

        # Visual Evidence (if present)
        vis = res_data.get("visual_evidence")
        if vis and isinstance(vis, dict):
            st.divider()
            st.markdown("### 🎯 Visual Evidence & Spatial Overlays")

            if "image_base64" in vis and vis["image_base64"]:
                b64_str = vis["image_base64"]
                if b64_str.startswith("data:image"):
                    b64_str = b64_str.split(",", 1)[1]
                try:
                    img_bytes = base64.b64decode(b64_str)
                    st.image(img_bytes, caption=f"Evidence: {res_data.get('selected_task')}", use_container_width=True)
                except Exception:
                    pass

            if "boxes" in vis and vis["boxes"]:
                with st.expander(f"Bounding Box Coordinates ({len(vis['boxes'])} detections)", expanded=False):
                    import pandas as pd
                    st.dataframe(pd.DataFrame(vis["boxes"]), use_container_width=True)

            elif "classes" in vis:
                st.write("**Predicted Class Distribution**")
                for pred in vis["classes"][:5]:
                    c1, c2 = st.columns([1, 2])
                    c1.write(f"**{pred.get('class_name')}**")
                    c2.progress(float(pred.get("confidence", 0.0)), text=f"{pred.get('confidence', 0.0)*100:.1f}%")

            elif "coverages" in vis:
                st.write("**Optical Feature Coverages**")
                cov_cols = st.columns(len(vis["coverages"]))
                for idx, (k, v) in enumerate(vis["coverages"].items()):
                    cov_cols[idx].metric(k.replace("_percent", "").title(), f"{v}%")

        # Auditable Execution Trace
        trace_steps = res_data.get("execution_trace", [])
        if trace_steps:
            st.divider()
            with st.expander("🔍 Auditable Execution Trace (Agent Routing Decisions)", expanded=True):
                st.caption("Step-by-step verifiable routing decisions logged by the SatQuery agent:")
                for step in trace_steps:
                    st.markdown(f"- {step}")
