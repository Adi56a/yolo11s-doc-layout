import os
import io
from pathlib import Path
import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
from ultralytics import YOLO

# Page config
st.set_page_config(
    page_title="YOLO Custom Model Inference",
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

# Custom Banner
st.markdown("""
<div class="banner">
    <h1 class="banner-title">🤖 YOLO Layout Inference Playground</h1>
    <p class="banner-subtitle">Upload multiple images to run batch inference using your custom fine-tuned YOLO model (supports 9 layout classes).</p>
</div>
""", unsafe_allow_html=True)

# Sidebar configurations
st.sidebar.markdown("### ⚙️ Inference Settings")

# Check for weight file
inferences_dir = Path(__file__).parent.resolve()
local_weights = inferences_dir / "110-best.pt"
fallback_weights = Path(__file__).parent.parent.resolve() / "models" / "custom_best.pt"

weights_path = None
if local_weights.exists():
    weights_path = local_weights
    st.sidebar.success("🟢 Found custom weights `inferences/110-best.pt`!")
elif fallback_weights.exists():
    weights_path = fallback_weights
    st.sidebar.info("🟡 Using project models weights `models/custom_best.pt`.")
else:
    st.sidebar.error("🔴 No weights file found! Please place your `110-best.pt` file in the `inferences` folder.")
    st.stop()

# Sliders
conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=1.00,
    value=0.25,
    step=0.05,
    help="Minimum score required to visualize a detection."
)

# Load YOLO model
@st.cache_resource
def load_yolo_model(path):
    return YOLO(str(path))

with st.spinner("Loading YOLO engine..."):
    try:
        model = load_yolo_model(weights_path)
        class_names = model.names
    except Exception as e:
        st.error(f"Failed to load YOLO model: {e}")
        st.stop()

# Class styling configurations (Dynamic colors matching the 9 classes)
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

# Image upload supporting multiple files (batch processing)
uploaded_files = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)

if uploaded_files:
    # Load all images
    images = []
    for f in uploaded_files:
        try:
            images.append(Image.open(f).convert("RGB"))
        except Exception as e:
            st.error(f"Error loading image {f.name}: {e}")
            
    if images:
        # Run batch prediction
        with st.spinner(f"Running batch inference on {len(images)} images..."):
            try:
                results = model.predict(source=images, conf=conf_threshold)
            except Exception as e:
                st.error(f"Inference failed: {e}")
                st.stop()
        
        # Display image selector if multiple images uploaded
        if len(images) > 1:
            st.markdown(f"### 🗂️ Batch Processing Results ({len(images)} images)")
            file_names = [f.name for f in uploaded_files]
            selected_file_name = st.selectbox("Select an image to inspect details:", file_names)
            selected_idx = file_names.index(selected_file_name)
        else:
            selected_idx = 0
            selected_file_name = uploaded_files[0].name
            
        # Extract results for the selected image
        image = images[selected_idx]
        result = results[selected_idx]
        boxes = result.boxes

        # Count classes dynamically
        class_counts = {cls_id: 0 for cls_id in range(9)}
        detected_data = []

        # Prepare image copy for custom drawing with semi-transparency
        annotated_img = image.copy()
        draw = ImageDraw.Draw(annotated_img, "RGBA")

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

            color = class_colors.get(cls_id, (255, 255, 255))
            
            # Transparent overlay: 30% alpha
            fill_color = color + (76,) 
            border_color = color + (255,)

            # Draw filled block and solid border
            draw.rectangle([x1, y1, x2, y2], fill=fill_color, outline=border_color, width=3)
            
            # Text label tag
            label_text = f"#{idx} {class_name} ({conf:.0%})"
            draw.text((x1 + 5, y1 + 5), label_text, fill=(255, 255, 255, 255))

        # Metrics Panel for selected image
        st.markdown(f"### 📊 Detection Statistics for `{selected_file_name}`")
        
        # Display dynamically: 1 card for total and rest for specific classes
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
            # Recreate columns to fit the detected classes nicely
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

        # Side by side visualizer
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("### 🖼️ Original Image")
            st.image(image, use_column_width=True)
            
        with col_right:
            st.markdown("### 👁️ Layout Detections")
            st.image(annotated_img, use_column_width=True)
            
            # Download annotated image button
            buffer = io.BytesIO()
            annotated_img.save(buffer, format="JPEG")
            st.download_button(
                label=f"📥 Download Annotated `{selected_file_name}`",
                data=buffer.getvalue(),
                file_name=f"annotated_{selected_file_name}",
                mime="image/jpeg"
            )

        # Table of details
        if detected_data:
            st.markdown("### 📊 Block Coordinates Map")
            st.dataframe(detected_data)
        else:
            st.info("No layout blocks found for this image.")
else:
    st.info("👋 Upload image files to analyze them using your custom trained model.")
