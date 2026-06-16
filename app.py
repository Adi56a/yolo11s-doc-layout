import os
import io
import sys
import time
from pathlib import Path
import streamlit as st
from PIL import Image, ImageDraw
import numpy as np

# Ensure local imports work regardless of working directory
sys.path.append(str(Path(__file__).parent.resolve()))

# Page config
st.set_page_config(
    page_title="YOLO Layout Inference Hub",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling matching parent app theme
st.markdown("""
<style>
    /* Global styles */
    .stApp {
        background-color: #0f121d;
        color: #e2e8f0;
    }
    
    h1, h2, h3 {
        color: #ffffff !important;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    
    /* Custom banner */
    .banner {
        background: linear-gradient(135deg, #1e1b4b 0%, #1e293b 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(99, 102, 241, 0.3);
        box-shadow: 0 10px 25px rgba(0,0,0,0.3);
    }
    .banner-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(to right, #818cf8, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .banner-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-top: 0.5rem;
    }
    
    /* Metrics panel card styling */
    .metric-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.15);
        transition: transform 0.2s ease;
        margin-bottom: 1rem;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# Class styling configurations (Dynamic colors matching the 9 layout classes)
class_colors = {
    0: (99, 102, 241),    # block_text (Indigo)
    1: (255, 234, 0),    # block_diagram (Bright Yellow)
    2: (16, 185, 129),   # block_table (Emerald Green)
    3: (245, 158, 11),   # block_rough (Amber)
    4: (107, 114, 128),  # block_empty (Gray)
    5: (236, 72, 153),   # question (Pink)
    6: (139, 92, 246),   # sub_question (Purple)
    7: (6, 182, 212),    # block_graph (Cyan)
    8: (239, 68, 68)     # block_map (Red)
}

# Dynamic scan for model weights
inferences_dir = Path(__file__).parent.resolve()
local_models_dir = inferences_dir / "models"
parent_models_dir = Path(__file__).parent.parent.resolve() / "models"

model_files = []
if local_models_dir.exists():
    for f in local_models_dir.iterdir():
        if f.suffix.lower() in ['.pt', '.onnx']:
            model_files.append(f)
if parent_models_dir.exists():
    for f in parent_models_dir.iterdir():
        if f.suffix.lower() in ['.pt', '.onnx']:
            model_files.append(f)

# Deduplicate and sort
model_files = sorted(list(set(model_files)), key=lambda x: x.name)

# Cache load functions
@st.cache_resource
def load_pytorch_model(path):
    from ultralytics import YOLO
    return YOLO(str(path))

@st.cache_resource
def load_onnx_model(path):
    from onnx_inference import YOLOONNX
    return YOLOONNX(str(path))

def draw_detections(image, boxes, class_names):
    """Draws layout boxes with translucent fill and labels on a copy of the image."""
    annotated_img = image.copy()
    draw = ImageDraw.Draw(annotated_img, "RGBA")
    
    for idx, box in enumerate(boxes, 1):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = class_names.get(cls_id, f"Class {cls_id}")
        
        color = class_colors.get(cls_id, (255, 255, 255))
        
        # Transparent overlay: 30% alpha
        fill_color = color + (76,) 
        border_color = color + (255,)
        
        # Draw filled block and solid border
        draw.rectangle([x1, y1, x2, y2], fill=fill_color, outline=border_color, width=3)
        
        # Text label tag
        label_text = f"#{idx} {class_name} ({conf:.0%})"
        draw.text((x1 + 5, y1 + 5), label_text, fill=(255, 255, 255, 255))
        
    return annotated_img

# Sidebar Router Navigation
st.sidebar.markdown("### 🧭 Navigation Router")
app_mode = st.sidebar.radio(
    "Select App Mode",
    ["Playground Mode", "Latency & Engine Benchmark"],
    help="Playground mode supports standard multi-image batch analysis. Benchmark mode compares PyTorch and ONNX latency."
)

if app_mode == "Playground Mode":
    # ------------------ PLAYGROUND MODE ------------------
    st.markdown("""
    <div class="banner">
        <h1 class="banner-title">🤖 YOLO Layout Inference Playground</h1>
        <p class="banner-subtitle">Upload multiple images to run batch inference using your custom fine-tuned YOLO model (supports 9 layout classes).</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("### ⚙️ Inference Settings")
    
    if not model_files:
        st.sidebar.error("🔴 No model files (.pt or .onnx) found in inferences/ or models/ folders.")
        st.stop()
        
    # Select model file
    model_display_names = [f.name for f in model_files]
    default_index = 0
    for preference in ["110-best.onnx", "110-best.pt", "best.onnx", "best.pt", "custom_model.onnx"]:
        if preference in model_display_names:
            default_index = model_display_names.index(preference)
            break
            
    selected_model_name = st.sidebar.selectbox(
        "Select Model File",
        options=model_display_names,
        index=default_index,
        help="Select the model checkpoint (.pt) or compiled format (.onnx)."
    )
    weights_path = model_files[model_display_names.index(selected_model_name)]
    
    # Detect model type and configure engine selection
    model_is_onnx = weights_path.suffix.lower() == ".onnx"
    engine_options = []
    if model_is_onnx:
        engine_options = ["Pure ONNX Runtime", "Ultralytics (ONNX)"]
    else:
        engine_options = ["Ultralytics (PyTorch)"]
        
    selected_engine = st.sidebar.selectbox(
        "Inference Engine",
        options=engine_options,
        help="Select execution backend. Pure ONNX Runtime is recommended for lightweight production testing."
    )
    
    # Export option for PyTorch checkpoints
    if not model_is_onnx:
        onnx_counterpart = weights_path.with_suffix(".onnx")
        if not onnx_counterpart.exists():
            st.sidebar.markdown("---")
            st.sidebar.warning(f"💡 Export `{selected_model_name}` to `.onnx` for faster runtime and production testing.")
            if st.sidebar.button("⚡ Export Model to ONNX"):
                with st.spinner(f"Compiling {selected_model_name} to ONNX format..."):
                    try:
                        from onnx_inference import export_to_onnx
                        export_to_onnx(weights_path, onnx_counterpart)
                        st.sidebar.success(f"✅ Exported to `{onnx_counterpart.name}`! Reloading models...")
                        st.rerun()
                    except Exception as e:
                        st.sidebar.error(f"Export failed: {e}")
                        
    # Sliders
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=1.00,
        value=0.25,
        step=0.05,
        help="Minimum score required to visualize a detection."
    )
    
    # Load model
    model_loaded = False
    model = None
    class_names = {}
    
    with st.spinner(f"Loading {selected_engine} engine..."):
        try:
            if selected_engine == "Pure ONNX Runtime":
                model = load_onnx_model(weights_path)
            else:
                model = load_pytorch_model(weights_path)
            class_names = model.names
            st.sidebar.success(f"🟢 Active: {selected_engine}")
            model_loaded = True
        except Exception as e:
            st.sidebar.error(f"Failed to load model: {e}")
            st.error(f"⚠️ Failed to load model `{selected_model_name}` using `{selected_engine}`:\n\n`{e}`")
            st.info("💡 Please select a different model file (e.g. `110-best.onnx` or `best.pt`) or a compatible inference engine in the sidebar configuration to recover.")
            model_loaded = False
            
    if model_loaded:
        uploaded_files = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
        
        if uploaded_files:
            images = []
            for f in uploaded_files:
                try:
                    images.append(Image.open(f).convert("RGB"))
                except Exception as e:
                    st.error(f"Error loading image {f.name}: {e}")
                    
            if images:
                with st.spinner(f"Running batch inference on {len(images)} images..."):
                    try:
                        results = model.predict(source=images, conf=conf_threshold)
                    except Exception as e:
                        st.error(f"Inference failed: {e}")
                        st.stop()
                        
                if len(images) > 1:
                    st.markdown(f"### 🗂️ Batch Processing Results ({len(images)} images)")
                    file_names = [f.name for f in uploaded_files]
                    selected_file_name = st.selectbox("Select an image to inspect details:", file_names)
                    selected_idx = file_names.index(selected_file_name)
                else:
                    selected_idx = 0
                    selected_file_name = uploaded_files[0].name
                    
                image = images[selected_idx]
                result = results[selected_idx]
                boxes = result.boxes
                
                # Count classes
                class_counts = {cls_id: 0 for cls_id in range(9)}
                detected_data = []
                
                for idx, box in enumerate(boxes, 1):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = class_names.get(cls_id, f"Class {cls_id}")
                    
                    if cls_id in class_counts:
                        class_counts[cls_id] += 1
                        
                    detected_data.append({
                        "Block ID": f"#{idx}",
                        "Class Name": class_name,
                        "Confidence": f"{conf:.2%}",
                        "Bounding Box": f"[{x1}, {y1}, {x2}, {y2}]"
                    })
                    
                # Draw detections
                annotated_img = draw_detections(image, boxes, class_names)
                
                # Metrics Panel
                st.markdown(f"### 📊 Detection Statistics for `{selected_file_name}`")
                col_total, *cols_classes = st.columns(5)
                
                with col_total:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="color: #a78bfa;">{len(boxes)}</div>
                        <div class="metric-label">Total Blocks</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                active_classes = {cls_id: count for cls_id, count in class_counts.items() if count > 0}
                if active_classes:
                    active_list = list(active_classes.items())
                    num_cols = min(len(active_list), 4)
                    class_cols = st.columns(num_cols)
                    for idx_c, (cls_id, count) in enumerate(active_list):
                        col_target = class_cols[idx_c % num_cols]
                        class_name = class_names.get(cls_id, f"Class {cls_id}")
                        rgb = class_colors.get(cls_id, (255, 255, 255))
                        hex_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
                        with col_target:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value" style="color: {hex_color};">{count}</div>
                                <div class="metric-label">{class_name}</div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info("No layout blocks detected. Try reducing the Confidence Threshold in the sidebar.")
                    
                st.markdown("---")
                
                # Visual side-by-side comparison
                col_left, col_right = st.columns(2)
                with col_left:
                    st.markdown("### 🖼️ Original Image")
                    st.image(image, use_column_width=True)
                with col_right:
                    st.markdown("### 👁️ Layout Detections")
                    st.image(annotated_img, use_column_width=True)
                    
                    buffer = io.BytesIO()
                    annotated_img.save(buffer, format="JPEG")
                    st.download_button(
                        label=f"📥 Download Annotated `{selected_file_name}`",
                        data=buffer.getvalue(),
                        file_name=f"annotated_{selected_file_name}",
                        mime="image/jpeg"
                    )
                    
                if detected_data:
                    st.markdown("### 📊 Block Coordinates Map")
                    st.dataframe(detected_data)
        else:
            st.info("👋 Upload image files to analyze them using your custom trained model.")

else:
    # ------------------ BENCHMARK MODE ------------------
    st.markdown("""
    <div class="banner">
        <h1 class="banner-title">⚡ Latency & Engine Performance Router</h1>
        <p class="banner-subtitle">Upload a test page to benchmark production ONNX Runtime versus standard PyTorch inference latency side-by-side.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("### ⚙️ Benchmark Configurations")
    
    pt_files = [f for f in model_files if f.suffix.lower() == '.pt']
    onnx_files = [f for f in model_files if f.suffix.lower() == '.onnx']
    
    if not pt_files or not onnx_files:
        st.error("🔴 Both .pt (PyTorch) and .onnx model files are required to run this benchmark.")
        st.stop()
        
    selected_pt_name = st.sidebar.selectbox("Select PyTorch Model (.pt)", [f.name for f in pt_files], index=0)
    selected_onnx_name = st.sidebar.selectbox("Select ONNX Model (.onnx)", [f.name for f in onnx_files], index=0)
    
    pt_path = pt_files[[f.name for f in pt_files].index(selected_pt_name)]
    onnx_path = onnx_files[[f.name for f in onnx_files].index(selected_onnx_name)]
    
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=1.00,
        value=0.25,
        step=0.05
    )
    
    uploaded_file = st.file_uploader("Upload Benchmarking Target Image", type=["png", "jpg", "jpeg", "webp"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        
        # Load models
        pt_model_loaded = False
        onnx_model_loaded = False
        
        with st.spinner("Preparing PyTorch model..."):
            try:
                pt_model = load_pytorch_model(pt_path)
                pt_model_loaded = True
            except Exception as e:
                st.error(f"Failed to load PyTorch model: {e}")
                
        with st.spinner("Preparing ONNX Runtime model..."):
            try:
                onnx_model = load_onnx_model(onnx_path)
                onnx_model_loaded = True
            except Exception as e:
                st.error(f"Failed to load ONNX model: {e}")
                
        if pt_model_loaded and onnx_model_loaded:
            if st.button("⚡ Run Performance Benchmark", type="primary"):
                with st.spinner("Executing warm-up passes (resolving initial execution caches)..."):
                    try:
                        # Warm up runs
                        pt_model.predict(source=image, conf=conf_threshold)
                        onnx_model.predict(source=image, conf=conf_threshold)
                    except Exception as e:
                        st.error(f"Warm-up run failed: {e}")
                        st.stop()
                        
                with st.spinner("Running timed benchmarks..."):
                    # Measure PyTorch Latency
                    t_start = time.perf_counter()
                    pt_results = pt_model.predict(source=image, conf=conf_threshold)
                    pt_time = (time.perf_counter() - t_start) * 1000
                    
                    # Measure ONNX Latency
                    t_start = time.perf_counter()
                    onnx_results = onnx_model.predict(source=image, conf=conf_threshold)
                    onnx_time = (time.perf_counter() - t_start) * 1000
                    
                # Extract results
                pt_res = pt_results[0]
                onnx_res = onnx_results[0]
                
                pt_boxes = pt_res.boxes
                onnx_boxes = onnx_res.boxes
                
                # Render outputs
                pt_annotated = draw_detections(image, pt_boxes, pt_model.names)
                onnx_annotated = draw_detections(image, onnx_boxes, onnx_model.names)
                
                # Show Benchmark dashboard
                st.markdown("### 🏁 Latency Comparison Dashboard")
                
                col_pt_stat, col_onnx_stat, col_speedup_stat = st.columns(3)
                
                with col_pt_stat:
                    st.markdown(f"""
                    <div class="metric-card" style="border-left: 5px solid #ef4444;">
                        <div class="metric-value" style="color: #f87171;">{pt_time:.1f} ms</div>
                        <div class="metric-label">PyTorch Latency</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col_onnx_stat:
                    st.markdown(f"""
                    <div class="metric-card" style="border-left: 5px solid #10b981;">
                        <div class="metric-value" style="color: #34d399;">{onnx_time:.1f} ms</div>
                        <div class="metric-label">ONNX Runtime Latency</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                speedup = pt_time / onnx_time if onnx_time > 0 else 1.0
                speedup_color = "#34d399" if speedup >= 1.5 else "#fca5a5"
                with col_speedup_stat:
                    st.markdown(f"""
                    <div class="metric-card" style="border-left: 5px solid #818cf8;">
                        <div class="metric-value" style="color: {speedup_color};">{speedup:.2f}x</div>
                        <div class="metric-label">ONNX Speedup Factor</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                # Chart
                st.markdown("#### 📈 Execution Time Comparison (Lower is Better)")
                chart_data = {
                    "Engine": ["PyTorch (Ultralytics)", "Pure ONNX Runtime"],
                    "Latency (ms)": [pt_time, onnx_time]
                }
                st.bar_chart(data=chart_data, x="Engine", y="Latency (ms)", use_container_width=True)
                
                # Layout visual side-by-side
                st.markdown("---")
                st.markdown("### 👁️ Predictions Visual Alignment check")
                col_pt_img, col_onnx_img = st.columns(2)
                
                with col_pt_img:
                    st.markdown(f"#### 🟥 PyTorch (`{selected_pt_name}`) | Detections: {len(pt_boxes)}")
                    st.image(pt_annotated, use_column_width=True)
                    
                with col_onnx_img:
                    st.markdown(f"#### 🟩 ONNX Runtime (`{selected_onnx_name}`) | Detections: {len(onnx_boxes)}")
                    st.image(onnx_annotated, use_column_width=True)
                    
                # Verification table of class coordinates
                st.markdown("---")
                st.markdown("### 📋 Detections Coordinate Matching Check")
                
                # Match boxes
                table_rows = []
                max_len = max(len(pt_boxes), len(onnx_boxes))
                for i in range(max_len):
                    row = {"Block ID": f"#{i+1}"}
                    
                    if i < len(pt_boxes):
                        pb = pt_boxes[i]
                        p_x1, p_y1, p_x2, p_y2 = map(int, pb.xyxy[0])
                        p_cls = int(pb.cls[0])
                        p_name = pt_model.names.get(p_cls, f"Class {p_cls}")
                        row["PyTorch Class"] = p_name
                        row["PyTorch Confidence"] = f"{float(pb.conf[0]):.1%}"
                        row["PyTorch Box"] = f"[{p_x1}, {p_y1}, {p_x2}, {p_y2}]"
                    else:
                        row["PyTorch Class"] = "-"
                        row["PyTorch Confidence"] = "-"
                        row["PyTorch Box"] = "-"
                        
                    if i < len(onnx_boxes):
                        ob = onnx_boxes[i]
                        o_x1, o_y1, o_x2, o_y2 = map(int, ob.xyxy[0])
                        o_cls = int(ob.cls[0])
                        o_name = onnx_model.names.get(o_cls, f"Class {o_cls}")
                        row["ONNX Class"] = o_name
                        row["ONNX Confidence"] = f"{float(ob.conf[0]):.1%}"
                        row["ONNX Box"] = f"[{o_x1}, {o_y1}, {o_x2}, {o_y2}]"
                    else:
                        row["ONNX Class"] = "-"
                        row["ONNX Confidence"] = "-"
                        row["ONNX Box"] = "-"
                        
                    table_rows.append(row)
                    
                if table_rows:
                    st.dataframe(table_rows, use_container_width=True)
                    
                    # Verify equivalence
                    boxes_match = len(pt_boxes) == len(onnx_boxes)
                    if boxes_match:
                        st.success("✅ Semantic Equivalence Verified: Both PyTorch and ONNX Runtime returned the same number of detected layout blocks.")
                    else:
                        st.warning(f"⚠️ Equivalence Warning: Detected block counts differ (PyTorch: {len(pt_boxes)} vs ONNX: {len(onnx_boxes)}). Verify confidence thresholds.")
    else:
        st.info("👋 Upload a test image and click '⚡ Run Performance Benchmark' to start latency comparison testing.")
