"""
Live Camera Inference Script for Jetson Nano
- YOLO for leaf detection
- Swin Transformer for disease classification
"""
import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
import json
import time
import argparse
from PIL import Image

class LeafDiseaseDetector:
    def __init__(self, yolo_model_path, classifier_model_path, config_path, confidence_threshold=0.5):
        """Initialize YOLO detector and disease classifier"""
        print("Initializing Leaf Disease Detector...")
        
        # Load YOLO model for leaf detection
        print(f"Loading YOLO model: {yolo_model_path}")
        self.yolo_model = YOLO(yolo_model_path)
        self.confidence_threshold = confidence_threshold
        
        # Load classification model (ONNX) with CUDA for Jetson Nano
        print(f"Loading classification model: {classifier_model_path}")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.classifier_session = ort.InferenceSession(classifier_model_path, providers=providers)
        
        # Check which provider is being used
        active_provider = self.classifier_session.get_providers()[0]
        if 'CUDA' in active_provider:
            print(f"✓ Using GPU acceleration (CUDA)")
        else:
            print(f"⚠ Using CPU (CUDA not available - inference will be slower)")
        
        # Load class labels
        with open(config_path, 'r') as f:
            config = json.load(f)
            self.id2label = config['id2label']
        
        print("Initialization complete!")
    
    def preprocess_for_classifier(self, image_crop, image_size=256):
        """Preprocess cropped leaf image for disease classification"""
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
    
    def classify_leaf(self, image_crop):
        """Classify disease in the cropped leaf image"""
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
        """Detect leaves using YOLO and classify diseases"""
        results_list = []
        
        # Run YOLO detection
        results = self.yolo_model(frame, conf=self.confidence_threshold, verbose=False)
        
        # Process each detection
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = self.yolo_model.names[cls]
                
                # Check if detected object is a leaf/plant
                # YOLO class 'potted plant' is typically class 58 in COCO
                # You can also train a custom YOLO model specifically for leaves
                if 'plant' in class_name.lower() or 'leaf' in class_name.lower() or cls == 58:
                    # Crop the detected leaf region
                    leaf_crop = frame[y1:y2, x1:x2]
                    
                    if leaf_crop.size > 0:  # Ensure valid crop
                        # Classify the disease
                        disease_label, disease_conf = self.classify_leaf(leaf_crop)
                        
                        results_list.append({
                            'bbox': (x1, y1, x2, y2),
                            'yolo_conf': conf,
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
            yolo_conf = result['yolo_conf']
            
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
            conf_text = f"Conf: {disease_conf:.2%}"
            
            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            (conf_w, conf_h), _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Draw filled rectangles for text background
            cv2.rectangle(frame, (x1, y1 - label_h - conf_h - 10), 
                         (x1 + max(label_w, conf_w) + 10, y1), color, -1)
            
            # Draw text
            cv2.putText(frame, label, (x1 + 5, y1 - conf_h - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, conf_text, (x1 + 5, y1 - 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame


def main():
    parser = argparse.ArgumentParser(description="Live camera inference for leaf disease detection")
    parser.add_argument("--yolo-model", type=str, default="yolov8n.pt", 
                       help="Path to YOLO model (default: yolov8n.pt)")
    parser.add_argument("--classifier-model", type=str, default="final_tomato_model/model.onnx",
                       help="Path to disease classifier ONNX model")
    parser.add_argument("--config", type=str, default="final_tomato_model/config.json",
                       help="Path to model config.json")
    parser.add_argument("--camera", type=int, default=0,
                       help="Camera device index (default: 0)")
    parser.add_argument("--confidence", type=float, default=0.5,
                       help="YOLO detection confidence threshold (default: 0.5)")
    parser.add_argument("--width", type=int, default=1280,
                       help="Camera frame width (default: 1280)")
    parser.add_argument("--height", type=int, default=720,
                       help="Camera frame height (default: 720)")
    parser.add_argument("--fps", type=int, default=30,
                       help="Camera FPS (default: 30)")
    parser.add_argument("--display", action="store_true",
                       help="Display video window (disable for headless operation)")
    args = parser.parse_args()
    
    # Initialize detector
    detector = LeafDiseaseDetector(
        yolo_model_path=args.yolo_model,
        classifier_model_path=args.classifier_model,
        config_path=args.config,
        confidence_threshold=args.confidence
    )
    
    # Open camera
    print(f"\nOpening camera {args.camera}...")
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
    print("\nPress any key to start...")
    
    if args.display:
        cv2.waitKey(0)
    
    # FPS calculation
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0
    paused = False
    frame_count = 0
    
    print("\nStarting live inference...\n")
    
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
                info_text = f"FPS: {current_fps:.1f} | Inference: {inference_time*1000:.1f}ms | Leaves: {len(results)}"
                cv2.putText(frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Print detection results
                if results:
                    print(f"Frame {frame_count}: Detected {len(results)} leaf(s)")
                    for i, result in enumerate(results, 1):
                        print(f"  Leaf {i}: {result['disease']} ({result['disease_conf']:.2%})")
                
                frame_count += 1
            else:
                frame = current_frame.copy()
                cv2.putText(frame, "PAUSED - Press 'p' to resume", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Display frame
            if args.display:
                cv2.imshow('Leaf Disease Detection', frame)
            
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
