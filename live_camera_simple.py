"""
Live Camera Inference Script (Simplified Version - No YOLO Required)
- Processes entire frame or uses color-based leaf detection
- Disease classification on detected regions
"""
import cv2
import numpy as np
import onnxruntime as ort
import json
import time
import argparse
from PIL import Image


class SimplifiedLeafDetector:
    def __init__(self, classifier_model_path, config_path, 
                 detection_mode='whole_frame', min_area=5000):
        """
        Initialize simplified leaf detector
        
        Args:
            classifier_model_path: Path to disease classification ONNX model
            config_path: Path to model config JSON
            detection_mode: 'whole_frame', 'color_based', or 'grid'
            min_area: Minimum area for color-based detection (pixels)
        """
        print("Initializing Simplified Leaf Disease Detector...")
        print(f"Detection Mode: {detection_mode}")
        
        self.detection_mode = detection_mode
        self.min_area = min_area
        
        # Load classification model (ONNX) with CUDA support if available
        print(f"Loading classification model: {classifier_model_path}")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.classifier_session = ort.InferenceSession(classifier_model_path, providers=providers)
        
        # Check which provider is being used
        active_provider = self.classifier_session.get_providers()[0]
        if 'CUDA' in active_provider:
            print(f"✓ Using GPU acceleration (CUDA)")
        else:
            print(f"⚠ Using CPU (inference will be slower)")
        
        # Load class labels
        with open(config_path, 'r') as f:
            config = json.load(f)
            self.id2label = config['id2label']
        
        print("Initialization complete!")
    
    def detect_leaves_color_based(self, frame):
        """
        Detect leaf regions using color-based segmentation (green color detection)
        Returns list of bounding boxes for detected leaf regions
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define range for green color (leaves)
        # These values may need tuning based on your lighting conditions
        lower_green1 = np.array([25, 40, 40])
        upper_green1 = np.array([85, 255, 255])
        
        # Create mask for green regions
        mask = cv2.inRange(hsv, lower_green1, upper_green1)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get bounding boxes for significant contours
        bboxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_area:
                x, y, w, h = cv2.boundingRect(contour)
                # Add some padding
                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(frame.shape[1] - x, w + 2*padding)
                h = min(frame.shape[0] - y, h + 2*padding)
                bboxes.append((x, y, x+w, y+h))
        
        return bboxes
    
    def detect_leaves_grid(self, frame, grid_size=2):
        """
        Divide frame into grid and classify each section
        Returns list of bounding boxes for grid cells
        """
        height, width = frame.shape[:2]
        cell_height = height // grid_size
        cell_width = width // grid_size
        
        bboxes = []
        for i in range(grid_size):
            for j in range(grid_size):
                x1 = j * cell_width
                y1 = i * cell_height
                x2 = x1 + cell_width
                y2 = y1 + cell_height
                bboxes.append((x1, y1, x2, y2))
        
        return bboxes
    
    def preprocess_for_classifier(self, image_crop, image_size=256):
        """Preprocess cropped region for disease classification"""
        # Convert from BGR (OpenCV) to RGB
        image_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        pil_image = pil_image.resize((image_size, image_size))
        
        # Convert to numpy array and normalize
        img_array = np.array(pil_image).astype(np.float32) / 255.0
        
        # Normalize using ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_array = (img_array - mean) / std
        
        # Transpose to CHW format and add batch dimension
        img_array = img_array.transpose(2, 0, 1)
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
        
        return img_array
    
    def classify_region(self, image_crop):
        """Classify disease in the cropped region"""
        # Preprocess image
        input_data = self.preprocess_for_classifier(image_crop)
        
        # Run inference
        input_name = self.classifier_session.get_inputs()[0].name
        outputs = self.classifier_session.run(None, {input_name: input_data})
        logits = outputs[0][0]
        
        # Calculate softmax probabilities
        probs = np.exp(logits) / np.sum(np.exp(logits))
        
        # Get prediction
        predicted_class_idx = np.argmax(logits)
        confidence = float(probs[predicted_class_idx])
        predicted_label = self.id2label[str(predicted_class_idx)]
        
        return predicted_label, confidence
    
    def detect_and_classify(self, frame):
        """Detect regions and classify diseases"""
        results_list = []
        
        if self.detection_mode == 'whole_frame':
            # Classify the entire frame
            disease_label, disease_conf = self.classify_region(frame)
            results_list.append({
                'bbox': (0, 0, frame.shape[1], frame.shape[0]),
                'disease': disease_label,
                'disease_conf': disease_conf
            })
        
        elif self.detection_mode == 'color_based':
            # Use color-based leaf detection
            bboxes = self.detect_leaves_color_based(frame)
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    disease_label, disease_conf = self.classify_region(crop)
                    results_list.append({
                        'bbox': bbox,
                        'disease': disease_label,
                        'disease_conf': disease_conf
                    })
        
        elif self.detection_mode == 'grid':
            # Use grid-based detection
            bboxes = self.detect_leaves_grid(frame, grid_size=2)
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    disease_label, disease_conf = self.classify_region(crop)
                    results_list.append({
                        'bbox': bbox,
                        'disease': disease_label,
                        'disease_conf': disease_conf
                    })
        
        return results_list
    
    def draw_results(self, frame, results):
        """Draw bounding boxes and labels on frame"""
        for result in results:
            x1, y1, x2, y2 = result['bbox']
            disease = result['disease']
            disease_conf = result['disease_conf']
            
            # Choose color based on health status
            if 'healthy' in disease.lower():
                color = (0, 255, 0)  # Green for healthy
            else:
                color = (0, 0, 255)  # Red for diseased
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label text
            disease_short = disease.replace('Tomato___', '').replace('_', ' ')
            label = f"{disease_short}"
            conf_text = f"{disease_conf:.1%}"
            
            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            (conf_w, conf_h), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Draw filled rectangle for text background
            cv2.rectangle(frame, (x1, y1 - label_h - conf_h - 10), 
                         (x1 + max(label_w, conf_w) + 10, y1), color, -1)
            
            # Draw text
            cv2.putText(frame, label, (x1 + 5, y1 - conf_h - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, conf_text, (x1 + 5, y1 - 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame


def main():
    parser = argparse.ArgumentParser(description="Simplified live camera inference for leaf disease detection")
    parser.add_argument("--classifier-model", type=str, default="final_tomato_model/model.onnx",
                       help="Path to disease classifier ONNX model")
    parser.add_argument("--config", type=str, default="final_tomato_model/config.json",
                       help="Path to model config.json")
    parser.add_argument("--camera", type=int, default=0,
                       help="Camera device index (default: 0)")
    parser.add_argument("--mode", type=str, default="color_based",
                       choices=['whole_frame', 'color_based', 'grid'],
                       help="Detection mode: whole_frame, color_based, or grid")
    parser.add_argument("--min-area", type=int, default=5000,
                       help="Minimum area for color-based detection (pixels)")
    parser.add_argument("--width", type=int, default=1280,
                       help="Camera frame width (default: 1280)")
    parser.add_argument("--height", type=int, default=720,
                       help="Camera frame height (default: 720)")
    parser.add_argument("--fps", type=int, default=30,
                       help="Camera FPS (default: 30)")
    parser.add_argument("--display", action="store_true",
                       help="Display video window")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("SIMPLIFIED LEAF DISEASE DETECTION")
    print("="*60)
    print(f"Detection Mode: {args.mode}")
    print("  - whole_frame: Classify entire camera view")
    print("  - color_based: Detect green regions as leaves")
    print("  - grid: Divide frame into grid and classify each cell")
    print("="*60 + "\n")
    
    # Initialize detector
    detector = SimplifiedLeafDetector(
        classifier_model_path=args.classifier_model,
        config_path=args.config,
        detection_mode=args.mode,
        min_area=args.min_area
    )
    
    # Open camera
    print(f"Opening camera {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Camera opened successfully!")
    print("\nControls:")
    print("  'q' - Quit")
    print("  's' - Save current frame")
    print("  'p' - Pause/Resume")
    print("  'm' - Switch detection mode")
    print("\nStarting live inference...\n")
    
    # FPS calculation
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0
    paused = False
    frame_count = 0
    
    detection_modes = ['whole_frame', 'color_based', 'grid']
    current_mode_idx = detection_modes.index(args.mode)
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                
                if not ret:
                    print("Error: Failed to read frame")
                    break
                
                # Record frame for pause functionality
                current_frame = frame.copy()
                
                # Detect and classify
                start_time = time.time()
                results = detector.detect_and_classify(frame)
                inference_time = time.time() - start_time
                
                # Draw results
                frame = detector.draw_results(frame, results)
                
                # Calculate FPS
                fps_counter += 1
                if fps_counter >= 10:
                    current_fps = fps_counter / (time.time() - fps_start_time)
                    fps_counter = 0
                    fps_start_time = time.time()
                
                # Draw FPS and info
                mode_text = f"Mode: {detector.detection_mode}"
                info_text = f"FPS: {current_fps:.1f} | Inference: {inference_time*1000:.1f}ms | Regions: {len(results)}"
                cv2.putText(frame, mode_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame, info_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Print detection results
                if results and frame_count % 30 == 0:  # Print every 30 frames
                    print(f"\nFrame {frame_count}: Detected {len(results)} region(s)")
                    for i, result in enumerate(results, 1):
                        print(f"  Region {i}: {result['disease']} ({result['disease_conf']:.1%})")
                
                frame_count += 1
            else:
                frame = current_frame.copy()
                cv2.putText(frame, "PAUSED - Press 'p' to resume", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Display frame
            if args.display:
                cv2.imshow('Leaf Disease Detection (Simplified)', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                filename = f"capture_{int(time.time())}.jpg"
                cv2.imwrite(filename, current_frame)
                print(f"Saved frame: {filename}")
            elif key == ord('p'):
                paused = not paused
                print("Paused" if paused else "Resumed")
            elif key == ord('m'):
                # Switch detection mode
                current_mode_idx = (current_mode_idx + 1) % len(detection_modes)
                detector.detection_mode = detection_modes[current_mode_idx]
                print(f"Switched to mode: {detector.detection_mode}")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if args.display:
            cv2.destroyAllWindows()
        print("\nCamera released. Goodbye!")


if __name__ == "__main__":
    main()
