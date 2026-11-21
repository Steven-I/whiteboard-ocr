import streamlit as st
import cv2
import numpy as np
import os

from processing import (
    normalize_resolution,
    isolate_whiteboard_roi,
    neutralize_glare_and_lighting,
    extract_text_regions,
    create_highlight_overlay
)
from ocr_engine import (
    is_tesseract_available,
    parse_with_baseline,
    parse_with_advanced
)

st.set_page_config(page_title="SmartBoard OCR System", layout="wide")

st.title("SmartBoard OCR: Whiteboard Diagram & Text Extraction")
st.markdown(
    "**Course Project:** COMP7116001 - Computer Vision | **Alignment:** LO 1 (Operators), "
    "LO 2 (Features), LO 3 (Localization/Clustering), LO 4 (Recognition Systems)"
)

# System Diagnostics
tesseract_ok = is_tesseract_available()
if not tesseract_ok:
    st.warning("⚠️ Native Tesseract binary not found on PATH. Baseline comparison will be disabled; EasyOCR will run as primary.")

# Sidebar Controls
st.sidebar.header("1. System Configuration")
gemini_key = st.sidebar.text_input("Gemini API Key (Optional VLM Engine)", type="password", value=os.environ.get("GEMINI_API_KEY", ""))

st.sidebar.header("2. Preprocessing Controls (LO 1 & LO 2)")
auto_isolate = st.sidebar.checkbox("Enable Surface Isolation & Deskew", value=True)
kernel_scale = st.sidebar.slider("Glare Filter Scale (Morphology)", 0.01, 0.06, 0.03, step=0.005)
block_size = st.sidebar.slider("Adaptive Threshold Block Size", 11, 51, 25, step=2)
c_val = st.sidebar.slider("Threshold Offset Constant (C)", 1, 25, 8)

st.sidebar.header("3. Manual Trimming Offsets (%)")
crop_top = st.sidebar.slider("Crop Top", 0, 40, 0)
crop_bottom = st.sidebar.slider("Crop Bottom", 0, 40, 0)
crop_left = st.sidebar.slider("Crop Left", 0, 40, 0)
crop_right = st.sidebar.slider("Crop Right", 0, 40, 0)

uploaded_file = st.file_uploader("Upload a whiteboard image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    raw_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if raw_image is None:
        st.error("Error decoding image file.")
    else:
        # Step 0: Standardize Resolution
        norm_image, scale_factor = normalize_resolution(raw_image, max_dim=1920)

        # Step 1: Surface Isolation (LO 1)
        isolated_img = isolate_whiteboard_roi(norm_image, enable_quad=auto_isolate) if auto_isolate else norm_image.copy()

        # Manual Trimming Apply
        h, w = isolated_img.shape[:2]
        top_px = int(h * (crop_top / 100.0))
        bot_px = int(h * (1.0 - (crop_bottom / 100.0)))
        left_px = int(w * (crop_left / 100.0))
        right_px = int(w * (1.0 - (crop_right / 100.0)))
        if bot_px > top_px and right_px > left_px:
            processed_img = isolated_img[top_px:bot_px, left_px:right_px]
        else:
            processed_img = isolated_img

        # Step 2: Illumination & Glare Normalization (LO 1 & LO 2)
        enhanced_gray, binary_mask = neutralize_glare_and_lighting(
            processed_img, kernel_scale=kernel_scale, block_size=block_size, c_val=c_val
        )

        # Step 3: Localization & Line Clustering (LO 3)
        boxes = extract_text_regions(binary_mask)

        # Step 4: Visual Highlight & Metric Coverage
        highlighted_img, coverage_pct = create_highlight_overlay(processed_img, binary_mask, boxes)

        st.info(
            f"**{coverage_pct:.2f}%** of the board area is identified as valid marker strokes. "
            f"Extracted **{len(boxes)} text regions**."
        )

        tab1, tab2 = st.tabs(["📷 Pipeline Visualizations & OCR", "📊 Academic Benchmarks (Report Data)"])

        with tab1:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Visual Pipeline Output")
                st.image(cv2.cvtColor(highlighted_img, cv2.COLOR_BGR2RGB), caption="Green Stroke & Bounding Box Overlay", use_container_width=True)
                st.image(binary_mask, caption="Noise & Smudge-Filtered Mask", use_container_width=True)

            with col2:
                st.subheader("Recognition Output (LO 4)")
                with st.spinner("Executing recognition engine..."):
                    adv_text, adv_meta, adv_time, engine_used = parse_with_advanced(
                        enhanced_gray, binary_mask, boxes=boxes, gemini_api_key=gemini_key
                    )

                st.caption(f"**Active Engine:** {engine_used} | **Inference Time:** {adv_time:.2f}s")
                st.text_area("Extracted Content", value=adv_text, height=420)

                st.download_button(
                    label="Download Notes (.txt)",
                    data=adv_text,
                    file_name="whiteboard_extracted_notes.txt",
                    mime="text/plain"
                )

        with tab2:
            st.subheader("Baseline vs. Proposed System Evaluation")
            st.markdown("Use these metrics for the **Implementation & Results** section of your 10–15 page final report.")

            if st.button("Run Baseline Comparison (PyTesseract)"):
                with st.spinner("Running baseline engine..."):
                    base_text, base_meta, base_time = parse_with_baseline(enhanced_gray, binary_mask, boxes)

                b_col1, b_col2 = st.columns(2)
                with b_col1:
                    st.markdown("### Baseline System (OpenCV + Tesseract)")
                    st.caption(f"**Execution Time:** {base_time:.2f}s | **Extracted Blocks:** {len(base_meta)}")
                    st.text_area("Baseline Output", value=base_text, height=300)

                with b_col2:
                    st.markdown(f"### Proposed System ({engine_used})")
                    st.caption(f"**Execution Time:** {adv_time:.2f}s | **Extracted Blocks:** {len(adv_meta)}")
                    st.text_area("Proposed Output", value=adv_text, height=300)

                # Quantitative Comparison Table
                st.markdown("### Metric Comparison Summary")
                st.table([
                    {"Metric": "Execution Time (s)", "Baseline (Tesseract)": f"{base_time:.2f}s", "Proposed Engine": f"{adv_time:.2f}s"},
                    {"Metric": "Detected Text Regions", "Baseline (Tesseract)": len(base_meta), "Proposed Engine": len(boxes)},
                    {"Metric": "Surface Coverage (%)", "Baseline (Tesseract)": f"{coverage_pct:.2f}%", "Proposed Engine": f"{coverage_pct:.2f}%"},
                    {"Metric": "Math / LaTeX Parsing", "Baseline (Tesseract)": "Unsupported", "Proposed Engine": "Supported"}
                ])