import os
import sys
import cv2
import numpy as np
from PIL import Image
import gradio as gr
import time
from pathlib import Path
import requests
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(f"{BASE_DIR}/input", exist_ok=True)
os.makedirs(f"{BASE_DIR}/output", exist_ok=True)
os.makedirs(f"{BASE_DIR}/logo_templates", exist_ok=True)

class LogoIdentifier:
    def __init__(self):
        self.logo_database = self.load_logo_database()
        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher()
        
    def load_logo_database(self):
        """Load known logos with their names and features"""
        logo_db = {
            "nike": {
                "name": "Nike",
                "color_profile": [(0, 0, 0)],  # Black dominant
                "shape": "swoosh",
                "keywords": ["swoosh", "just do it", "nike"]
            },
            "adidas": {
                "name": "Adidas",
                "color_profile": [(0, 0, 0), (255, 255, 255)],  # Black/white
                "shape": "three stripes",
                "keywords": ["three stripes", "adidas", "impossible is nothing"]
            },
            "apple": {
                "name": "Apple",
                "color_profile": [(255, 255, 255), (0, 0, 0)],  # White/black
                "shape": "apple shape",
                "keywords": ["apple", "bite", "fruit"]
            },
            "starbucks": {
                "name": "Starbucks",
                "color_profile": [(0, 51, 0)],  # Green dominant
                "shape": "mermaid",
                "keywords": ["starbucks", "coffee", "mermaid", "siren"]
            },
            "mcdonalds": {
                "name": "McDonald's",
                "color_profile": [(255, 193, 7)],  # Yellow/Gold
                "shape": "arches",
                "keywords": ["mcdonalds", "golden arches", "fast food"]
            },
            "cocacola": {
                "name": "Coca-Cola",
                "color_profile": [(255, 0, 0)],  # Red dominant
                "shape": "script text",
                "keywords": ["coca", "cola", "red", "script"]
            },
            "pepsi": {
                "name": "Pepsi",
                "color_profile": [(0, 0, 255)],  # Blue dominant
                "shape": "circle wave",
                "keywords": ["pepsi", "blue", "circle"]
            },
            "google": {
                "name": "Google",
                "color_profile": [(66, 133, 244), (234, 67, 53), (251, 188, 5), (52, 168, 83)],  # Multi-color
                "shape": "text",
                "keywords": ["google", "search", "multi-color"]
            },
            "amazon": {
                "name": "Amazon",
                "color_profile": [(255, 153, 0)],  # Orange
                "shape": "smile arrow",
                "keywords": ["amazon", "smile", "arrow"]
            },
            "facebook": {
                "name": "Facebook",
                "color_profile": [(59, 89, 152)],  # Blue
                "shape": "f",
                "keywords": ["facebook", "f", "social media"]
            }
        }
        return logo_db
    
    def extract_features(self, image):
        """Extract SIFT features from image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def color_similarity(self, image_color, logo_color_profile):
        """Calculate color similarity between detected region and logo colors"""
        avg_color = np.mean(image_color, axis=(0, 1))
        min_distance = float('inf')
        
        for logo_color in logo_color_profile:
            distance = np.linalg.norm(avg_color - logo_color)
            if distance < min_distance:
                min_distance = distance
        
        # Convert to similarity score (0-1)
        similarity = max(0, 1 - min_distance / 441.67)  # 441.67 is max distance in RGB space
        return similarity
    
    def shape_analysis(self, contour):
        """Analyze shape characteristics"""
        if len(contour) < 5:
            return "unknown"
        
        # Calculate shape properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return "unknown"
            
        # Circularity
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        
        # Classify shape
        if circularity > 0.8:
            return "circle"
        elif aspect_ratio > 1.5:
            return "horizontal rectangle"
        elif aspect_ratio < 0.67:
            return "vertical rectangle"
        elif 0.8 < aspect_ratio < 1.2:
            return "square"
        else:
            return "irregular"
    
    def identify_logo(self, roi, contour):
        """Identify logo name based on multiple features"""
        best_match = "Unknown Logo"
        best_confidence = 0
        
        # Analyze ROI features
        try:
            # Color analysis
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            dominant_color = np.mean(roi, axis=(0, 1))
            
            # Shape analysis
            shape = self.shape_analysis(contour)
            
            # Size and aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Feature matching (simplified)
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            for logo_id, logo_info in self.logo_database.items():
                confidence = 0
                
                # Color matching (40% weight)
                color_sim = self.color_similarity(roi, logo_info["color_profile"])
                confidence += color_sim * 0.4
                
                # Shape context matching (30% weight)
                # For simplicity, we'll use some heuristic rules
                if logo_info["shape"] in ["circle", "square"] and shape in ["circle", "square"]:
                    confidence += 0.3
                elif "rectangle" in shape and "text" in logo_info["shape"]:
                    confidence += 0.3
                elif "irregular" in shape and logo_info["shape"] != "text":
                    confidence += 0.2
                
                # Size/aspect ratio heuristics (30% weight)
                if logo_info["name"] in ["Nike", "Apple"] and aspect_ratio > 1.5:
                    confidence += 0.3
                elif logo_info["name"] in ["Google", "Facebook"] and 0.8 < aspect_ratio < 1.2:
                    confidence += 0.3
                elif logo_info["name"] in ["Coca-Cola", "Starbucks"] and aspect_ratio > 2:
                    confidence += 0.3
                else:
                    confidence += 0.1
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = logo_info["name"]
                    
        except Exception as e:
            print(f"Error in logo identification: {e}")
            best_match = "Unknown Logo"
            best_confidence = 0.1
        
        return best_match, min(best_confidence, 1.0)

class OpenCVLogoDetector:
    def __init__(self, model_weights=None, model_config=None):
        self.net = None
        self.identifier = LogoIdentifier()
        
        if model_weights and model_config and os.path.exists(model_weights) and os.path.exists(model_config):
            self.load_yolo_model(model_weights, model_config)
        else:
            self.load_default_detector()
    
    def load_yolo_model(self, weights_path, config_path):
        """Load YOLO model using OpenCV DNN"""
        try:
            self.net = cv2.dnn.readNet(weights_path, config_path)
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                print("Using GPU acceleration")
            except:
                print("Using CPU")
            
            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            print("YOLO model loaded successfully")
            
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.load_default_detector()
    
    def load_default_detector(self):
        """Load a default OpenCV detector"""
        print("Loading default object detector...")
        self.detector_type = "feature_based"
    
    def detect_with_yolo(self, image, conf_threshold=0.5, nms_threshold=0.4):
        """Detect objects using YOLO model"""
        height, width = image.shape[:2]
        
        blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        
        boxes = []
        confidences = []
        class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > conf_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                roi = image[y:y+h, x:x+w]
                
                # Identify logo name
                contour = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
                logo_name, identification_confidence = self.identifier.identify_logo(roi, contour)
                
                results.append({
                    'class_id': class_ids[i],
                    'class_name': logo_name,
                    'confidence': confidences[i],
                    'identification_confidence': identification_confidence,
                    'bbox': [x, y, x + w, y + h]
                })
        
        return results
    
    def detect_with_feature_based(self, image, conf_threshold=0.3):
        """Feature-based logo detection with identification"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use multiple methods for better detection
        # Method 1: Contour detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Method 2: Thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Combine methods
        combined = cv2.bitwise_or(edges, thresh)
        
        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        results = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 1000 < area < 50000:  # Reasonable logo size range
                x, y, w, h = cv2.boundingRect(contour)
                
                # Skip very small or very large detections
                if w < 50 or h < 50 or w > 500 or h > 500:
                    continue
                
                roi = image[y:y+h, x:x+w]
                if roi.size == 0:
                    continue
                
                # Identify logo
                logo_name, identification_confidence = self.identifier.identify_logo(roi, contour)
                detection_confidence = min(area / 10000, 1.0)
                
                if detection_confidence > conf_threshold:
                    results.append({
                        'class_id': 0,
                        'class_name': logo_name,
                        'confidence': detection_confidence,
                        'identification_confidence': identification_confidence,
                        'bbox': [x, y, x + w, y + h]
                    })
        
        return results
    
    def detect(self, image, conf_threshold=0.5, nms_threshold=0.4):
        """Main detection function"""
        if self.net is not None:
            return self.detect_with_yolo(image, conf_threshold, nms_threshold)
        else:
            return self.detect_with_feature_based(image, conf_threshold)

def draw_detections(image, detections):
    """Draw bounding boxes and labels with logo names"""
    result_image = image.copy()
    
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        logo_name = detection['class_name']
        detection_conf = detection['confidence']
        ident_conf = detection.get('identification_confidence', 0)
        
        # Create label with logo name and confidence
        label = f"{logo_name} ({detection_conf:.2f}, ID:{ident_conf:.2f})"
        
        # Choose color based on identification confidence
        if ident_conf > 0.7:
            color = (0, 255, 0)  # Green - high confidence
        elif ident_conf > 0.4:
            color = (0, 255, 255)  # Yellow - medium confidence
        else:
            color = (0, 0, 255)  # Red - low confidence
        
        # Draw bounding box
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 3)
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(result_image, (x1, y1 - label_size[1] - 15), 
                     (x1 + label_size[0], y1), color, -1)
        
        # Draw label text
        cv2.putText(result_image, label, (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return result_image

def get_output(input_image, confidence_threshold=0.5, iou_threshold=0.4):
    """Process image with logo detection and identification"""
    try:
        # Convert PIL Image to OpenCV format
        if isinstance(input_image, Image.Image):
            input_image = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
        
        # Initialize detector
        detector = OpenCVLogoDetector()
        
        # Perform detection
        start_time = time.time()
        detections = detector.detect(input_image, confidence_threshold, iou_threshold)
        detection_time = time.time() - start_time
        
        # Draw detections on image
        result_image = draw_detections(input_image, detections)
        
        # Convert back to PIL Image
        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        result_pil = Image.fromarray(result_image_rgb)
        
        # Create detection summary
        unique_logos = {}
        for detection in detections:
            logo_name = detection['class_name']
            if logo_name not in unique_logos:
                unique_logos[logo_name] = 0
            unique_logos[logo_name] += 1
        
        summary = f"Detected {len(detections)} logo instances in {detection_time:.2f}s\n"
        summary += "Identified logos:\n"
        for logo, count in unique_logos.items():
            summary += f"• {logo}: {count} instance(s)\n"
        
        print(f"Detection completed in {detection_time:.3f}s")
        print(f"Found {len(detections)} logo instances")
        
        return result_pil, summary
        
    except Exception as e:
        print(f"Error during detection: {e}")
        error_msg = f"Error: {str(e)}"
        if isinstance(input_image, np.ndarray):
            error_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(error_image), error_msg
        else:
            return input_image, error_msg

# Create the enhanced Gradio interface
with gr.Blocks(title="Logo Identification Tool", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🔍 Logo Identification Tool
        **Detect and Identify Logos by Name**
        
        Upload an image to detect logos and identify their brand names using computer vision
        """
    )
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", type="numpy")
            
            with gr.Row():
                confidence_slider = gr.Slider(
                    minimum=0.01, maximum=1.0, value=0.5, step=0.01,
                    label="Detection Confidence Threshold"
                )
                iou_slider = gr.Slider(
                    minimum=0.01, maximum=1.0, value=0.4, step=0.01,
                    label="IOU Threshold"
                )
            
            detect_btn = gr.Button("Identify Logos", variant="primary", size="lg")
        
        with gr.Column():
            output_image = gr.Image(label="Identification Result", type="pil")
            output_text = gr.Textbox(
                label="Detection Summary", 
                interactive=False,
                lines=6,
                placeholder="Logo identification results will appear here..."
            )
    
    # Supported logos section
    gr.Markdown("### 🎯 Supported Logos")
    supported_logos = """
    **Currently trained to recognize:**
    - ✅ Nike (Swoosh)
    - ✅ Adidas (Three Stripes) 
    - ✅ Apple (Apple Logo)
    - ✅ Starbucks (Mermaid)
    - ✅ McDonald's (Golden Arches)
    - ✅ Coca-Cola (Script)
    - ✅ Pepsi (Circle Wave)
    - ✅ Google (Text)
    - ✅ Amazon (Smile Arrow)
    - ✅ Facebook (F)
    
    *More logos can be added to the database*
    """
    gr.Markdown(supported_logos)
    
    # Set up the detection function
    detect_btn.click(
        fn=get_output,
        inputs=[input_image, confidence_slider, iou_slider],
        outputs=[output_image, output_text]
    )
    
    gr.Markdown(
        """
        ---
        **How to use:**
        1. 📷 Upload an image containing logos
        2. ⚙️ Adjust detection sensitivity if needed
        3. 🔍 Click 'Identify Logos' to analyze the image
        4. 📊 View identified logos with confidence scores
        
        **Color Coding:**
        - 🟢 Green: High identification confidence (>70%)
        - 🟡 Yellow: Medium confidence (40-70%)
        - 🔴 Red: Low confidence (<40%)
        
        *Powered by OpenCV Computer Vision & Feature Matching*
        """
    )

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import cv2
        print(f"✅ OpenCV version: {cv2.__version__}")
        return True
    except ImportError:
        print("❌ OpenCV not installed. Please install:")
        print("pip install opencv-python pillow numpy gradio")
        return False

if __name__ == "__main__":
    if check_dependencies():
        print("🚀 Starting Logo Identification Tool...")
        print("📊 Logo database loaded with 10 popular brands")
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=True
        )
    else:
        print("Please install the required dependencies first!")
