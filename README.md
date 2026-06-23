# 🤖 YOLO11s Document Layout & Marksense Pipeline Hub 📄

This repository contains two primary modules:
1. **Interactive Layout Inference & Visualization Utilities** (`app.py` and `predict.py`): Visual playground tools designed to display and test YOLO model detections using drawn bounding boxes.
2. **End-to-End Production Processing Pipeline** (`run_pipeline.py`): The core backend pipeline that converts student PDF sheets to images, uploads pages to S3, crops layout elements, uploads block crops to S3, runs OCR text extraction, and maps student answers to question anchors—returning clean, structured JSON payloads.

---

## 📋 1. Core Model Class Definitions

The YOLO model segments document elements into 9 distinct semantic layout classes:

| Class ID | Class Name | Target Element | Visual Indicator | Default Color (RGB) |
| :---: | :--- | :--- | :---: | :--- |
| **0** | `block_text` | Standard paragraph text blocks | 🔵 | Indigo (`99, 102, 241`) |
| **1** | `block_diagram` | Visual illustrations, drawings | 🟡 | Yellow (`255, 234, 0`) |
| **2** | `block_table` | Tabular grids, matrices | 🟢 | Green (`16, 185, 129`) |
| **3** | `block_rough` | Hand-written scribbles, draft work | 🟠 | Amber (`245, 158, 11`) |
| **4** | `block_empty` | Empty structural padding blocks | ⚫ | Gray (`107, 114, 128`) |
| **5** | `question` | Primary question text boundaries | 💗 | Pink (`236, 72, 153`) |
| **6** | `sub_question` | Sub-parts or nested questions | 🟣 | Purple (`139, 92, 246`) |
| **7** | `block_graph` | Chart plots, line/bar/pie charts | cyan | Cyan (`6, 182, 212`) |
| **8** | `block_map` | Cartographic maps, spatial plots | 🔴 | Red (`239, 68, 68`) |

---

## 🛠️ 2. Setup & Installation

Install python dependencies from the manifest:
```bash
pip install -r requirements.txt
```
*(For headless Linux servers, `packages.txt` lists the `libgl1` and `libglib2.0-0` system packages required by OpenCV).*

---

## 🔄 3. Production Marksense Processing Pipeline (`run_pipeline.py`)

> [!IMPORTANT]
> **Production vs. Playground**
> Unlike the playground tools which overlay boxes onto a single image, the **Marksense Pipeline** is a headless backend script designed to process student answer sheets end-to-end, upload files to S3, run local OCR, and generate structured JSON outputs matching answers to question papers.

### 📶 Pipeline Architecture & Workflow

```mermaid
flowchart TD
    PDF[1. PDF Answer Sheet] -->|fitz PDF Splitting| PNGs[2. Page Images P1.png, P2.png...]
    PNGs -->|s3_helper| S3Pages[(S3 answer_sheet_pages/)]
    PNGs -->|onnx_inference| YOLO[3. YOLO Layout Block Detection]
    YOLO -->|Direct crops| Crops[4. Crop PNGs P1_B1.png, P1_B2.png...]
    Crops -->|s3_helper| S3Blocks[(S3 answer_blocks/)]
    Crops -->|ocr_helper| OCR[5. OCR Text Extraction]
    OCR -->|Answer-to-Question mapping| Maps[6. Global Lookup Mapping JSONs]
    
    style PDF fill:#1e1b4b,stroke:#818cf8,color:#fff
    style S3Pages fill:#1e293b,stroke:#94a3b8,color:#fff
    style S3Blocks fill:#1e293b,stroke:#94a3b8,color:#fff
    style Maps fill:#111827,stroke:#10b981,stroke-width:2px,color:#fff
```

### ⚙️ Pipeline Configuration (`.env` file)
Rename [.env.example](file:///d:/NextLeap/block%20detection/inferences/.env.example) to `.env` and configure your credentials.

* **AWS S3 Credentials:** Define `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, and `S3_BUCKET_NAME`.
* **Local Storage Toggle:** Set `USE_LOCAL_STORAGE=True` to run offline (it will save files in `outputs/local_s3_mock` using S3 directory structure) or `False` to run real S3 uploads.
* **OCR engine selection:** Set `OCR_ENGINE=mock` (instant offline testing), `easyocr` (local deep learning model), or `pytesseract` (local Tesseract wrapper).

### 📁 Dynamic S3 Folder Architecture
When uploading, S3 files are structured dynamically using the student parameters:
```text
{school_name}/{academic_year}/{class}/{section}/{subject}/{assessment_id}/students/{student_id}/
├── answer_sheet_pages/
│   ├── P1.png            <-- Page 1 Image
│   └── P2.png            <-- Page 2 Image
└── answer_blocks/
    ├── P1_B1.png         <-- Page 1, Crop Block 1
    ├── P1_B2.png         <-- Page 1, Crop Block 2
    └── P2_B1.png         <-- Page 2, Crop Block 1
```

### 🚀 Pipeline Execution Command
To run the production pipeline, run the following command in your terminal:
```bash
python run_pipeline.py \
  --input samples/cli_onnx_test.jpg \
  --school-name scholars_home \
  --academic-year 2025-2026 \
  --class 9th \
  --section 9th-A \
  --subject SCIENCE_CHEMISTRY \
  --assessment-id Assessment1 \
  --student-id 11 \
  --marksense-uuid ms_456 \
  --question-paper-uuid qp_789 \
  --output-dir outputs
```

### 📄 Generated JSON Outputs
The pipeline outputs four structured JSON results inside the `--output-dir` folder:
* **`res_pages.json`**: Links each page number to its uploaded S3 URL (maintaining page order).
* **`res_blocks.json`**: Lists cropped blocks containing block numbers, S3 URLs, layout types, bounding boxes (`{x, y, w, h}`), and whether the block is a question label (question anchor).
* **`res_contents.json`**: Lists block numbers, S3 URLs, and their extracted OCR text.
* **`res_lookup.json`**: Maps student answer blocks (drawings, standard text) to their preceding question anchors.

---

## 🎨 4. Layout Playground & CLI Visualizers (`app.py` & `predict.py`)

These tools are designed to verify layout detection quality visually by drawing translucent color bounding box overlays on top of the images.

### 💻 Streamlit Web Application Dashboard
Starts a local Streamlit web application on port `8501`:
```bash
streamlit run app.py
```
* **Playground Mode:** Upload single or batch images to visually inspect YOLO detections, change confidence thresholds, download drawings, and check coordinate tables.
* **Latency Benchmark Mode:** Run timed runs comparing the inference execution speed of standard PyTorch against optimized ONNX Runtime engines.

### 🖼️ Standalone CLI Visualization Script
Run a simple test on an image and save the visual drawing output:
```bash
python predict.py \
  --image samples/cli_onnx_test.jpg \
  --weights models/110-best.pt \
  --output outputs/visual_output.jpg
```
*(This script loads the weights, detects blocks, overlays translucent class colors matching the 9 layout classes, and saves the image to `--output`).*
