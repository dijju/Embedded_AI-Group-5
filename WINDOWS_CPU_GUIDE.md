# Running on Windows (Without CUDA)

## 🖥️ Current Situation

Your Windows machine doesn't have CUDA/GPU support available. This is normal for most Windows PCs that don't have NVIDIA GPUs with CUDA installed.

## ✅ What's Been Fixed

The scripts have been updated to automatically:
- Detect if CUDA is available
- Fall back to CPU if not
- Show clear messages about which device is being used

## 🚀 How to Use Now

### 1. Training YOLO (Optional)

```bash
# This will now automatically use CPU
python train_yolo_leaf.py --mode train

# Or explicitly specify CPU
python train_yolo_leaf.py --mode train --device cpu

# For faster testing, reduce epochs and batch size
python train_yolo_leaf.py --mode train --device cpu --epochs 10 --batch-size 4
```

**Note:** Training on CPU is MUCH slower than GPU. For a full 100 epochs, it could take hours. Consider:
- Using a smaller model: `--model-size n` (nano - fastest)
- Reducing epochs: `--epochs 10` for testing
- Using Google Colab with free GPU for training
- Training on a machine with GPU

### 2. Live Camera Inference

```bash
# Run the live camera system (will auto-detect and use CPU)
python live_camera_inference.py --display

# Or use pre-downloaded YOLO model
python live_camera_inference.py --yolo-model yolov8n.pt --display
```

**Note:** Real-time performance may be slower on CPU. Expect:
- 5-10 FPS on decent CPUs
- 15-20+ FPS with GPU (Jetson Nano, etc.)

### 3. Test Setup

```bash
# This will tell you what's available
python test_setup.py
```

## 🎯 Recommendations

### For Development/Testing on Windows:

1. **Skip custom YOLO training** - Use pre-trained YOLOv8 directly:
   ```bash
   python live_camera_inference.py --display
   ```

2. **Test the disease classifier** - This works fine on CPU:
   ```bash
   python test_setup.py
   ```

3. **Use the Web Interface** - Simpler for testing individual images:
   ```bash
   python app.py
   # Then open http://localhost:5000
   ```

### For Production Deployment:

Deploy on Jetson Nano or similar device with GPU for best performance:
- Copy the entire FP folder to your Jetson Nano
- Install dependencies: `pip3 install -r requirements_camera.txt`
- Run: `python3 live_camera_inference.py --display`

## 🔧 Getting CUDA on Windows (Optional)

If you want GPU acceleration on Windows:

1. **Check your GPU**:
   ```bash
   nvidia-smi
   ```
   You need an NVIDIA GPU (not AMD or Intel integrated graphics)

2. **Install CUDA Toolkit**:
   - Download from: https://developer.nvidia.com/cuda-downloads
   - Choose Windows version matching your system

3. **Install PyTorch with CUDA**:
   ```bash
   pip3 uninstall torch
   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Verify**:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   # Should print: True
   ```

## 📊 Performance Comparison

| Device | YOLO Inference | Classification | Total FPS |
|--------|----------------|----------------|-----------|
| CPU (Windows) | ~50-100ms | ~20-40ms | 8-12 FPS |
| Jetson Nano GPU | ~15-30ms | ~10-20ms | 15-25 FPS |
| High-end GPU | ~5-10ms | ~5-10ms | 50+ FPS |

## ⚡ Quick Commands Summary

```bash
# Just want to test the web interface? (Easiest)
python app.py

# Want to test live camera (will be slower on CPU)
python live_camera_inference.py --display

# Want to train YOLO on CPU (will be very slow)
python train_yolo_leaf.py --mode train --device cpu --epochs 10 --batch-size 4

# Test your setup
python test_setup.py
```

## 💡 Best Approach for Your Setup

Since you're on Windows without CUDA:

1. **For Testing**: Use the web interface
   ```bash
   python app.py
   ```

2. **For Development**: Test scripts work correctly
   ```bash
   python test_setup.py
   ```

3. **For Deployment**: Transfer to Jetson Nano when ready
   - All the scripts are ready to run on Jetson Nano with GPU
   - Just copy the FP folder and install dependencies
   - Will automatically use GPU on Jetson Nano

## 🎬 Next Steps

1. ✅ Scripts are now fixed and will work on CPU
2. Choose your next action:
   - 🌐 Test web interface: `python app.py`  
   - 🎥 Test camera (slower): `python live_camera_inference.py --display`
   - 🔍 Check setup: `python test_setup.py`
   - 🚀 Deploy to Jetson Nano: Transfer and run there

---

**TL;DR:** Everything works now! It will just use CPU instead of GPU. For development on Windows, use the web interface. For production, deploy to Jetson Nano for better performance.
