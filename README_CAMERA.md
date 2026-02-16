# Live Camera Inference for Tomato Leaf Disease Detection

This script enables real-time tomato leaf disease detection using:
1. **YOLO** for leaf/plant detection
2. **Swin Transformer** for disease classification on Jetson Nano

## Features

- 🎥 Real-time camera feed processing
- 🌱 YOLO-based leaf detection
- 🔬 Disease classification with confidence scores
- 📊 Live FPS monitoring
- 💾 Frame capture capability
- ⏸️ Pause/Resume functionality
- 🎨 Color-coded visualization (Green=Healthy, Red=Diseased)

## Requirements

### Hardware
- **Jetson Nano** (or compatible NVIDIA Jetson device)
- **USB Camera** or **CSI Camera**
- Minimum 4GB RAM recommended

### Software
```bash
# Install dependencies
pip3 install -r requirements_camera.txt

# For Jetson Nano, install ONNX Runtime with GPU support:
pip3 install onnxruntime-gpu
```

## YOLO Model Setup

You have two options for YOLO:

### Option 1: Use Pre-trained YOLOv8 (Quick Start)
```bash
# Download YOLOv8 nano model (automatic on first run)
# The script will automatically download yolov8n.pt
python3 live_camera_inference.py
```

**Note:** Standard YOLO models detect "potted plant" (class 58) from COCO dataset. This works for basic plant detection but may not be optimal for leaves.

### Option 2: Train Custom YOLO for Leaf Detection (Recommended)
For better accuracy, train YOLOv8 on a leaf detection dataset:

1. **Collect/Download Leaf Dataset**
   - Use datasets like PlantDoc, LeafSnap, or create your own
   - Annotate images with bounding boxes around leaves

2. **Train Custom YOLO Model**
   ```bash
   # Install Ultralytics
   pip3 install ultralytics
   
   # Train YOLOv8
   yolo task=detect mode=train model=yolov8n.pt data=leaf_dataset.yaml epochs=100 imgsz=640
   ```

3. **Use Custom Model**
   ```bash
   python3 live_camera_inference.py --yolo-model path/to/custom_yolo.pt
   ```

## Usage

### Basic Usage
```bash
# Using default camera (camera 0)
python3 live_camera_inference.py

# Specify camera device
python3 live_camera_inference.py --camera 0

# Using CSI camera on Jetson Nano
python3 live_camera_inference.py --camera 0
```

### Advanced Options
```bash
python3 live_camera_inference.py \
    --yolo-model yolov8n.pt \
    --classifier-model final_tomato_model/model.onnx \
    --config final_tomato_model/config.json \
    --camera 0 \
    --confidence 0.5 \
    --width 1280 \
    --height 720 \
    --fps 30 \
    --display
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--yolo-model` | `yolov8n.pt` | Path to YOLO model for leaf detection |
| `--classifier-model` | `final_tomato_model/model.onnx` | Path to disease classifier |
| `--config` | `final_tomato_model/config.json` | Path to config file |
| `--camera` | `0` | Camera device index |
| `--confidence` | `0.5` | YOLO detection confidence threshold |
| `--width` | `1280` | Camera frame width |
| `--height` | `720` | Camera frame height |
| `--fps` | `30` | Camera FPS |
| `--display` | `False` | Enable video window display |

### Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit the application |
| `s` | Save current frame as JPG |
| `p` | Pause/Resume detection |

## CSI Camera Setup (Jetson Nano)

If using a CSI camera on Jetson Nano, you may need to use GStreamer pipeline:

```python
# Modify the camera initialization in the script:
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

# Replace cap = cv2.VideoCapture(args.camera) with:
cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
```

## Output

The script will:
1. Display live video feed with bounding boxes
2. Show disease classification for each detected leaf
3. Print detection results to console:
   ```
   Frame 42: Detected 2 leaf(s)
     Leaf 1: Tomato___healthy (95.32%)
     Leaf 2: Tomato___Early_blight (87.45%)
   ```

### Color Coding
- 🟢 **Green Box**: Healthy leaf
- 🔴 **Red Box**: Diseased leaf

## Disease Classes

The model can detect 10 tomato leaf conditions:
1. Bacterial spot
2. Early blight
3. Late blight
4. Leaf Mold
5. Septoria leaf spot
6. Spider mites (Two-spotted spider mite)
7. Target Spot
8. Tomato Yellow Leaf Curl Virus
9. Tomato mosaic virus
10. Healthy

## Performance Tips

### Jetson Nano Optimization

1. **Enable MAX Performance Mode**
   ```bash
   sudo nvpmodel -m 0
   sudo jetson_clocks
   ```

2. **Reduce Resolution for Better FPS**
   ```bash
   python3 live_camera_inference.py --width 640 --height 480
   ```

3. **Use Lighter YOLO Model**
   ```bash
   python3 live_camera_inference.py --yolo-model yolov8n.pt  # Nano (fastest)
   ```

4. **Adjust Confidence Threshold**
   ```bash
   python3 live_camera_inference.py --confidence 0.6  # Higher = fewer false positives
   ```

### Expected Performance
- **Resolution**: 640x480 → ~15-20 FPS
- **Resolution**: 1280x720 → ~8-12 FPS
- **YOLO Model**: Nano (yolov8n) is fastest
- **GPU Acceleration**: Ensure CUDA is properly configured

## Troubleshooting

### Camera Not Opening
```bash
# List available cameras
v4l2-ctl --list-devices

# Test camera
sudo apt-get install v4l-utils
v4l2-ctl --list-formats-ext
```

### ONNX Runtime GPU Issues
```bash
# Verify CUDA installation
nvcc --version

# Reinstall onnxruntime-gpu
pip3 uninstall onnxruntime onnxruntime-gpu
pip3 install onnxruntime-gpu
```

### Low FPS
- Reduce camera resolution
- Use yolov8n (nano) instead of larger models
- Lower confidence threshold to reduce processing
- Close other applications

### YOLO Not Detecting Leaves
- Pre-trained YOLO may not detect leaves well
- Train a custom YOLO model on leaf datasets
- Adjust `--confidence` threshold
- Ensure good lighting conditions

## Headless Operation (No Display)

For running without a display (e.g., remote Jetson):
```bash
# Remove --display flag
python3 live_camera_inference.py

# The script will still process and print results to console
```

## Integration with Other Systems

You can modify the script to:
- Send results to a database
- Trigger alerts for diseased plants
- Log detections to files
- Stream results over network
- Control actuators (e.g., spraying systems)

## License

This project is part of the Tomato Disease Classification system.

## Support

For issues or questions:
1. Check Jetson Nano documentation
2. Verify camera compatibility
3. Ensure all dependencies are installed
4. Test with standard YOLO detection first
