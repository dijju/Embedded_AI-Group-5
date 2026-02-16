# Quick Start Guide - Live Camera Inference on Jetson Nano

## 🚀 Quick Setup (5 Minutes)

### Step 1: Install Dependencies
```bash
# On your Jetson Nano
cd Final_Project/FP

# Install requirements
pip3 install -r requirements_camera.txt

# For Jetson Nano, ensure ONNX Runtime GPU support:
pip3 install onnxruntime-gpu
```

### Step 2: Test Your Setup
```bash
# Run the setup test script
python3 test_setup.py

# This will verify:
# - All packages are installed
# - Camera is working
# - Models are loaded correctly
# - CUDA/GPU is available
```

### Step 3: Run Live Inference

**Option A: Simplified Version (Recommended - Works Immediately)**
```bash
# Use color-based leaf detection (no YOLO training needed)
python3 live_camera_simple.py --mode color_based --display

# Or try whole frame mode (simplest)
python3 live_camera_simple.py --mode whole_frame --display

# Switch modes during runtime by pressing 'm'
```

**Option B: With YOLO (Requires Training or Pre-trained Model)**
```bash
# Basic run (using default pre-trained YOLO)
python3 live_camera_inference.py --display

# Note: Standard YOLO may not detect leaves well
# See LEAF_DETECTION_GUIDE.md for training custom YOLO
```

## 🎯 What Each Script Does

### 1a. `live_camera_simple.py` - Simplified Detection ⭐ **RECOMMENDED**
**Purpose:** Real-time leaf disease detection without needing YOLO training

**Usage:**
```bash
# Color-based detection (finds green regions)
python3 live_camera_simple.py --mode color_based --display

# Whole frame (simplest)
python3 live_camera_simple.py --mode whole_frame --display

# Grid-based (divides frame into sections)
python3 live_camera_simple.py --mode grid --display
```

**Controls:**
- Press `m` to switch detection modes
- Press `q` to quit
- Press `s` to save frame
- Press `p` to pause/resume

**Why use this?**
- ✅ Works immediately, no training needed
- ✅ No YOLO required
- ✅ Good for most use cases
- ⚠️ May need lighting adjustments for color detection

### 1b. `live_camera_inference.py` - YOLO-Based Detection
**Purpose:** Real-time leaf detection using YOLO + disease classification

**Usage:**
```bash
# Simple run
python3 live_camera_inference.py --display

# Advanced options
python3 live_camera_inference.py \
    --camera 0 \
    --confidence 0.5 \
    --width 1280 \
    --height 720
```

**Why use this?**
- ✅ More accurate with custom trained YOLO
- ⚠️ Requires proper leaf detection dataset
- ⚠️ Standard YOLO doesn't detect leaves well

**See [LEAF_DETECTION_GUIDE.md](LEAF_DETECTION_GUIDE.md) for training custom YOLO**

### 2. `test_setup.py` - Setup Testing
**Purpose:** Verify everything is installed correctly

**Usage:**
```bash
# Test everything
python3 test_setup.py

# Test without camera
python3 test_setup.py --skip-camera

# Test specific camera
python3 test_setup.py --camera 1
```

### 3. `train_yolo_leaf.py` - Custom YOLO Training
**Purpose:** Train YOLOv8 specifically for leaf detection (optional, for better accuracy)

**Usage:**
```bash
# Create dataset configuration
python3 train_yolo_leaf.py --mode create-yaml --dataset-path ./leaf_dataset

# Train model
python3 train_yolo_leaf.py --mode train --epochs 100

# Use custom model
python3 live_camera_inference.py --yolo-model runs/leaf_detection/train/weights/best.pt
```

## 📊 Expected Results

### With Pre-trained YOLO (Quick Start)
- **Pros:** No training needed, works immediately
- **Cons:** May miss some leaves, detects general "plants"
- **Use Case:** Quick testing, proof of concept

### With Custom YOLO (Recommended for Production)
- **Pros:** Better leaf detection, fewer false positives
- **Cons:** Requires dataset and training time
- **Use Case:** Deployment, production systems

## 🔧 Troubleshooting Guide

### Problem: "Camera not found"
```bash
# List available cameras
ls /dev/video*

# Test camera
v4l2-ctl --list-devices

# Try different camera index
python3 live_camera_inference.py --camera 1
```

### Problem: "Low FPS / Slow performance"
```bash
# Enable Jetson Nano max performance
sudo nvpmodel -m 0
sudo jetson_clocks

# Reduce resolution
python3 live_camera_inference.py --width 640 --height 480

# Use nano model (fastest)
python3 live_camera_inference.py --yolo-model yolov8n.pt
```

### Problem: "CUDA not available"
```bash
# Check CUDA installation
nvcc --version

# Verify onnxruntime-gpu
python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"

# Should show: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

### Problem: "YOLO not detecting leaves"
```bash
# Lower confidence threshold
python3 live_camera_inference.py --confidence 0.3

# Or train custom YOLO model (see train_yolo_leaf.py)
```

## 📁 Project Structure After Setup

```
FP/
├── live_camera_inference.py      ← Main live camera script
├── test_setup.py                  ← Setup verification script
├── train_yolo_leaf.py             ← YOLO training helper
├── requirements_camera.txt        ← Dependencies list
├── README_CAMERA.md               ← Detailed documentation
├── QUICKSTART.md                  ← This file
│
├── final_tomato_model/
│   ├── model.onnx                 ← Disease classifier
│   └── config.json                ← Model configuration
│
└── runs/                          ← Created after training custom YOLO
    └── leaf_detection/
        └── train/
            └── weights/
                └── best.pt        ← Your custom YOLO model
```

## 🎬 Complete Workflow

### For Quick Demo (5 minutes):
```bash
# 1. Install
pip3 install -r requirements_camera.txt

# 2. Test
python3 test_setup.py

# 3. Run (simplified - works immediately)
python3 live_camera_simple.py --mode color_based --display
```

### For Production Deployment (Depends on approach):

**Approach A: Using Simplified Detection (Fast Setup)**
```bash
# 1. Install dependencies
pip3 install -r requirements_camera.txt

# 2. Test on Jetson Nano
python3 live_camera_simple.py --mode color_based --display

# 3. Tune for your environment
# - Adjust --min-area for leaf size
# - Try different modes (whole_frame, color_based, grid)
# - Optimize lighting

# 4. Deploy
# - Remove --display for headless operation
# - Add logging/database integration
# - Set up as system service
```

**Approach B: Using Custom YOLO (Best Accuracy)**
```bash
# 1. Install dependencies
pip3 install -r requirements_camera.txt

# 2. Prepare leaf detection dataset
# - Collect leaf images
# - Annotate with bounding boxes
# - Organize in YOLO format

# 3. Train custom YOLO
python3 train_yolo_leaf.py --mode train --epochs 100

# 4. Test custom model
python3 live_camera_inference.py \
    --yolo-model runs/leaf_detection/train/weights/best.pt \
    --display

# 5. Deploy
# - Remove --display for headless operation
# - Add logging/database integration
# - Set up as system service
```

## 💡 Tips for Best Performance

### Lighting
- Use good, even lighting
- Avoid direct sunlight causing glare
- LED grow lights work well

### Camera Position
- Mount camera 20-40cm above plants
- Angle slightly downward
- Ensure leaves fill frame

### Model Optimization
- Start with default settings
- Tune confidence threshold based on your environment
- Consider training custom YOLO for your specific setup

### Jetson Nano Settings
```bash
# Max performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Check temperature
sudo apt-get install jetson-stats
sudo jtop

# Keep under 80°C for reliability
```

## 📸 Example Output

Console output:
```
Frame 42: Detected 2 leaf(s)
  Leaf 1: healthy (95.32%)
  Leaf 2: Early_blight (87.45%)

FPS: 12.4 | Inference: 85.2ms | Leaves: 2
```

Video display:
- Green boxes around healthy leaves
- Red boxes around diseased leaves
- Labels showing disease type and confidence

## 🔗 Next Steps

1. **Test with real plants** - Point camera at tomato plants
2. **Tune parameters** - Adjust confidence threshold for your needs
3. **Train custom YOLO** - For better leaf detection in your environment
4. **Add features** - Database logging, alerts, etc.
5. **Deploy** - Set up as system service for continuous monitoring

## 📚 Additional Resources

- YOLOv8 Documentation: https://docs.ultralytics.com/
- ONNX Runtime: https://onnxruntime.ai/
- Jetson Nano Guide: https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit

## 🆘 Getting Help

If you encounter issues:

1. Run `python3 test_setup.py` to diagnose
2. Check [README_CAMERA.md](README_CAMERA.md) for detailed documentation
3. Ensure all dependencies are installed
4. Verify camera and CUDA are working

---

**Ready to go? Start with the simplified version:**
```bash
python3 live_camera_simple.py --mode color_based --display
```

**Or test the full setup first:**
```bash
python3 test_setup.py && python3 live_camera_simple.py --mode color_based --display
```

**Want to train custom YOLO?** See [LEAF_DETECTION_GUIDE.md](LEAF_DETECTION_GUIDE.md)

Good luck! 🌱
