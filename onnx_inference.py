import os
import cv2
import numpy as np
from PIL import Image

class ONNXBox:
    """Mimics the Ultralytics Box object for downstream code compatibility."""
    def __init__(self, xyxy, cls_id, conf):
        # Coordinates: [x1, y1, x2, y2]
        self.xyxy = np.array([xyxy], dtype=np.float32)
        # Class ID
        self.cls = np.array([cls_id], dtype=np.int32)
        # Confidence score
        self.conf = np.array([conf], dtype=np.float32)

class ONNXResult:
    """Mimics the Ultralytics Result object for downstream code compatibility."""
    def __init__(self, boxes, names):
        self.boxes = boxes  # List of ONNXBox objects
        self.names = names  # Dict of class mappings: {id: name}

class YOLOONNX:
    """Pure ONNX Runtime inference wrapper for YOLOv8/v11 models."""
    def __init__(self, model_path):
        import onnxruntime as ort
        self.model_path = model_path
        
        # Load the ONNX model
        self.session = ort.InferenceSession(
            model_path, 
            providers=['CPUExecutionProvider']
        )
        
        # Get model input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name
        
        # Parse names from metadata
        self.names = self.parse_metadata()

    def parse_metadata(self):
        """Extracts class names from ONNX model metadata if present, or returns defaults."""
        import ast
        import json
        
        default_names = {
            0: "block_text",
            1: "block_diagram",
            2: "block_table",
            3: "block_rough",
            4: "block_empty",
            5: "question",
            6: "sub_question",
            7: "block_graph",
            8: "block_map"
        }
        
        try:
            meta = self.session.get_modelmeta()
            if 'names' in meta.custom_metadata_map:
                names_str = meta.custom_metadata_map['names']
                try:
                    return ast.literal_eval(names_str)
                except Exception:
                    return json.loads(names_str)
        except Exception:
            pass
            
        return default_names

    def preprocess(self, img):
        """Preprocesses input image (letterboxing, normalization, transposition)."""
        # img can be a PIL Image or numpy array
        if isinstance(img, Image.Image):
            img_np = np.array(img)
            # PIL uses RGB, convert to BGR for OpenCV processing
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            img_cv = img.copy()

        h0, w0 = img_cv.shape[:2]
        
        # Letterboxing to 640x640
        input_size = 640
        r = min(input_size / h0, input_size / w0)
        new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
        
        img_resized = cv2.resize(img_cv, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        dw = (input_size - new_w) / 2
        dh = (input_size - new_h) / 2
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        # Standard YOLO padding value is 114
        img_padded = cv2.copyMakeBorder(
            img_resized, top, bottom, left, right, 
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        # Convert BGR (OpenCV format) to RGB for YOLO ONNX model input
        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
        
        # Scale to [0.0, 1.0]
        input_tensor = img_rgb.astype(np.float32) / 255.0
        
        # Transpose to BCHW: (1, 3, 640, 640)
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor, r, (dw, dh)

    def postprocess(self, output, conf_threshold, original_shape, ratio, padding):
        """Processes raw model predictions to obtain filtered, scaled bounding boxes."""
        # Output is of shape (1, num_classes + 4, 8400)
        predictions = np.squeeze(output, axis=0) # (num_classes + 4, 8400)
        predictions = predictions.T # (8400, num_classes + 4)
        
        num_classes = predictions.shape[1] - 4
        
        bboxes = []
        scores = []
        class_ids = []
        
        # Filter predictions by confidence
        for row in predictions:
            coords = row[:4]
            class_scores = row[4:]
            
            cls_id = np.argmax(class_scores)
            conf = class_scores[cls_id]
            
            if conf >= conf_threshold:
                # Convert center xywh to top-left xywh for cv2.dnn.NMSBoxes
                x_center, y_center, w, h = coords
                x_min = x_center - w / 2
                y_min = y_center - h / 2
                
                bboxes.append([float(x_min), float(y_min), float(w), float(h)])
                scores.append(float(conf))
                class_ids.append(int(cls_id))
                
        if not bboxes:
            return []
            
        # Run Non-Maximum Suppression (NMS)
        # NMS threshold default in YOLO is 0.70
        nms_threshold = 0.70
        indices = cv2.dnn.NMSBoxes(bboxes, scores, conf_threshold, nms_threshold)
        
        onnx_boxes = []
        h0, w0 = original_shape[:2]
        dw, dh = padding
        
        if len(indices) > 0:
            # Flatten indices if needed (depends on cv2 version, though newer returns flattened list)
            indices_flat = np.array(indices).flatten()
            for idx in indices_flat:
                x_min, y_min, w, h = bboxes[idx]
                cls_id = class_ids[idx]
                conf = scores[idx]
                
                # Convert top-left xywh back to corner coordinates xyxy
                x1 = x_min
                y1 = y_min
                x2 = x_min + w
                y2 = y_min + h
                
                # Scale coordinates back to original image dimensions
                x1 = (x1 - dw) / ratio
                y1 = (y1 - dh) / ratio
                x2 = (x2 - dw) / ratio
                y2 = (y2 - dh) / ratio
                
                # Clip to image boundaries
                x1 = max(0.0, min(x1, w0))
                y1 = max(0.0, min(y1, h0))
                x2 = max(0.0, min(x2, w0))
                y2 = max(0.0, min(y2, h0))
                
                onnx_boxes.append(ONNXBox([x1, y1, x2, y2], cls_id, conf))
                
        return onnx_boxes

    def predict(self, source, conf=0.25):
        """Runs batch or single image inference on the ONNX model."""
        # Determine if source is list-like or single item
        is_list = isinstance(source, (list, tuple))
        sources = source if is_list else [source]
        
        results = []
        for src in sources:
            # Resolve image format to numpy array for dimensions
            if isinstance(src, Image.Image):
                img_np = np.array(src)
                original_shape = img_np.shape
                img_to_preprocess = src
            elif isinstance(src, (str, os.PathLike)):
                img_cv = cv2.imread(str(src))
                if img_cv is None:
                    raise ValueError(f"Could not load image file: {src}")
                img_np = img_cv
                original_shape = img_cv.shape
                img_to_preprocess = img_cv
            elif isinstance(src, np.ndarray):
                img_np = src
                original_shape = src.shape
                img_to_preprocess = src
            else:
                raise TypeError(f"Unsupported image type: {type(src)}")
                
            # Preprocess
            input_tensor, ratio, padding = self.preprocess(img_to_preprocess)
            
            # Run inference
            outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
            
            # Postprocess
            boxes = self.postprocess(
                outputs[0], 
                conf_threshold=conf, 
                original_shape=original_shape, 
                ratio=ratio, 
                padding=padding
            )
            
            results.append(ONNXResult(boxes, self.names))
            
        return results

def export_to_onnx(pytorch_model_path, onnx_model_path=None):
    """Utility to compile YOLO PyTorch checkpoints to ONNX format."""
    from ultralytics import YOLO
    model = YOLO(str(pytorch_model_path))
    if onnx_model_path is None:
        onnx_model_path = model.export(format="onnx")
    else:
        # ultralytics export saves to same directory, we can move it if target path is different
        exported_path = model.export(format="onnx")
        if exported_path != str(onnx_model_path):
            import shutil
            shutil.move(exported_path, onnx_model_path)
            onnx_model_path = str(onnx_model_path)
    return onnx_model_path
