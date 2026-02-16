"""
Test Script for Live Camera Inference Setup
This script verifies that all components are working correctly before running the full system.
"""
import sys
import os


def test_imports():
    """Test if all required packages can be imported"""
    print("\n" + "="*60)
    print("Testing Package Imports")
    print("="*60)
    
    results = {}
    
    # Test OpenCV
    try:
        import cv2
        print(f"✓ OpenCV {cv2.__version__}")
        results['opencv'] = True
    except ImportError as e:
        print(f"✗ OpenCV - NOT INSTALLED")
        print(f"  Install: pip3 install opencv-python")
        results['opencv'] = False
    
    # Test NumPy
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
        results['numpy'] = True
    except ImportError:
        print(f"✗ NumPy - NOT INSTALLED")
        print(f"  Install: pip3 install numpy")
        results['numpy'] = False
    
    # Test PIL
    try:
        from PIL import Image
        import PIL
        print(f"✓ Pillow {PIL.__version__}")
        results['pillow'] = True
    except ImportError:
        print(f"✗ Pillow - NOT INSTALLED")
        print(f"  Install: pip3 install Pillow")
        results['pillow'] = False
    
    # Test ONNX Runtime
    try:
        import onnxruntime as ort
        print(f"✓ ONNX Runtime {ort.__version__}")
        providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in providers:
            print(f"  ✓ GPU Support Available (CUDA)")
        else:
            print(f"  ⚠ CPU Only (No CUDA support)")
        results['onnxruntime'] = True
    except ImportError:
        print(f"✗ ONNX Runtime - NOT INSTALLED")
        print(f"  Install: pip3 install onnxruntime-gpu")
        results['onnxruntime'] = False
    
    # Test Ultralytics (YOLO)
    try:
        from ultralytics import YOLO
        import ultralytics
        print(f"✓ Ultralytics {ultralytics.__version__}")
        results['ultralytics'] = True
    except ImportError:
        print(f"✗ Ultralytics - NOT INSTALLED")
        print(f"  Install: pip3 install ultralytics")
        results['ultralytics'] = False
    
    return results


def test_camera(camera_id=0):
    """Test camera access"""
    print("\n" + "="*60)
    print(f"Testing Camera (Device {camera_id})")
    print("="*60)
    
    try:
        import cv2
    except ImportError:
        print("✗ OpenCV not installed, skipping camera test")
        return False
    
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"✗ Could not open camera {camera_id}")
        print("\nTroubleshooting:")
        print("  1. Check if camera is connected")
        print("  2. Try different camera index (--camera 1, 2, etc.)")
        print("  3. Check camera permissions")
        print("  4. For CSI camera on Jetson, may need GStreamer pipeline")
        return False
    
    # Try to read a frame
    ret, frame = cap.read()
    if not ret:
        print(f"✗ Could not read frame from camera {camera_id}")
        cap.release()
        return False
    
    height, width = frame.shape[:2]
    print(f"✓ Camera opened successfully")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    
    # Save a test frame
    test_frame_path = "test_camera_frame.jpg"
    cv2.imwrite(test_frame_path, frame)
    print(f"  Test frame saved: {test_frame_path}")
    
    cap.release()
    return True


def test_classifier_model(model_path='final_tomato_model/model.onnx', 
                          config_path='final_tomato_model/config.json'):
    """Test disease classifier model"""
    print("\n" + "="*60)
    print("Testing Disease Classifier Model")
    print("="*60)
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"✗ Model file not found: {model_path}")
        return False
    
    if not os.path.exists(config_path):
        print(f"✗ Config file not found: {config_path}")
        return False
    
    print(f"✓ Model file found: {model_path}")
    print(f"✓ Config file found: {config_path}")
    
    try:
        import onnxruntime as ort
        import json
        import numpy as np
        
        # Load model
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(model_path, providers=providers)
        
        active_provider = session.get_providers()[0]
        print(f"✓ Model loaded successfully")
        print(f"  Using: {active_provider}")
        
        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        num_classes = len(config['id2label'])
        print(f"✓ Config loaded successfully")
        print(f"  Number of classes: {num_classes}")
        
        # Test inference with dummy input
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        print(f"  Input name: {input_name}")
        print(f"  Input shape: {input_shape}")
        
        # Create dummy input
        dummy_input = np.random.randn(1, 3, 256, 256).astype(np.float32)
        outputs = session.run(None, {input_name: dummy_input})
        
        print(f"✓ Test inference successful")
        print(f"  Output shape: {outputs[0].shape}")
        
        # Print classes
        print(f"\n  Disease classes:")
        for idx, label in config['id2label'].items():
            print(f"    {idx}: {label}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing model: {e}")
        return False


def test_yolo_model(model_path='yolov8n.pt'):
    """Test YOLO model"""
    print("\n" + "="*60)
    print("Testing YOLO Model")
    print("="*60)
    
    try:
        from ultralytics import YOLO
        import numpy as np
        
        # Load YOLO model (will download if not present)
        print(f"Loading YOLO model: {model_path}")
        model = YOLO(model_path)
        print(f"✓ YOLO model loaded successfully")
        
        # Test with dummy image
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        results = model(dummy_image, verbose=False)
        
        print(f"✓ Test inference successful")
        print(f"  Model can detect {len(model.names)} classes")
        
        # Check if plant-related classes exist
        plant_classes = [name for name in model.names.values() if 'plant' in name.lower() or 'leaf' in name.lower()]
        if plant_classes:
            print(f"  ✓ Found plant-related classes: {plant_classes}")
        else:
            print(f"  ⚠ No plant-specific classes found")
            print(f"    Consider training a custom YOLO model for better leaf detection")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing YOLO: {e}")
        return False


def test_cuda():
    """Test CUDA availability"""
    print("\n" + "="*60)
    print("Testing CUDA/GPU Support")
    print("="*60)
    
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        
        if 'CUDAExecutionProvider' in providers:
            print("✓ CUDA is available")
            return True
        else:
            print("⚠ CUDA is NOT available")
            print("  The system will use CPU, which will be slower")
            print("  For Jetson Nano, ensure:")
            print("    1. CUDA is properly installed")
            print("    2. onnxruntime-gpu is installed")
            return False
    except ImportError:
        print("✗ Cannot test CUDA (onnxruntime not installed)")
        return False


def print_summary(results):
    """Print test summary"""
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = all(results.values())
    
    if all_passed:
        print("✓ All tests passed! You're ready to run the live camera system.")
        print("\nRun the system with:")
        print("  python3 live_camera_inference.py --display")
    else:
        print("⚠ Some tests failed. Please fix the issues above before proceeding.")
        print("\nFailed components:")
        for component, passed in results.items():
            if not passed:
                print(f"  - {component}")
    
    return all_passed


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test live camera inference setup")
    parser.add_argument("--camera", type=int, default=0,
                       help="Camera device index to test")
    parser.add_argument("--skip-camera", action="store_true",
                       help="Skip camera test")
    parser.add_argument("--classifier-model", type=str, default="final_tomato_model/model.onnx",
                       help="Path to classifier model")
    parser.add_argument("--config", type=str, default="final_tomato_model/config.json",
                       help="Path to config file")
    parser.add_argument("--yolo-model", type=str, default="yolov8n.pt",
                       help="Path to YOLO model")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("LIVE CAMERA INFERENCE - SETUP TEST")
    print("="*60)
    print("This script will verify your setup is ready for live inference")
    
    # Run tests
    results = {}
    
    # Test imports
    import_results = test_imports()
    results.update(import_results)
    
    # Test CUDA
    cuda_available = test_cuda()
    results['cuda'] = cuda_available
    
    # Test camera
    if not args.skip_camera and results.get('opencv', False):
        camera_ok = test_camera(args.camera)
        results['camera'] = camera_ok
    
    # Test classifier model
    if results.get('onnxruntime', False):
        classifier_ok = test_classifier_model(args.classifier_model, args.config)
        results['classifier'] = classifier_ok
    
    # Test YOLO
    if results.get('ultralytics', False):
        yolo_ok = test_yolo_model(args.yolo_model)
        results['yolo'] = yolo_ok
    
    # Print summary
    all_passed = print_summary(results)
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
