# Leaf Detection Solutions Guide

## 🎯 The Problem

Standard YOLO models (trained on COCO dataset) don't detect individual leaves well because:
- COCO only has "potted plant" class, not "leaf"
- It detects whole plants, not individual leaves
- Not optimized for close-up leaf images

## ✅ Solutions

### Option 1: Use Simplified Script (Works Now - No Training Needed) **⭐ RECOMMENDED**

Use the new `live_camera_simple.py` which doesn't require YOLO:

```bash
# Color-based detection (detects green regions)
python live_camera_simple.py --mode color_based --display

# Whole frame classification (simplest)
python live_camera_simple.py --mode whole_frame --display

# Grid-based (divides frame into sections)
python live_camera_simple.py --mode grid --display
```

**Detection Modes:**

1. **whole_frame** - Classifies entire camera view
   - ✅ Fastest
   - ✅ No detection needed
   - ⚠️ Only one prediction per frame
   - 👍 Best for: Single leaf fills camera view

2. **color_based** - Detects green regions automatically
   - ✅ No training required
   - ✅ Finds multiple leaves
   - ⚠️ Lighting dependent
   - 👍 Best for: Multiple leaves, good lighting

3. **grid** - Divides frame into grid cells
   - ✅ Checks multiple regions
   - ✅ Simple and reliable
   - ⚠️ May include non-leaf areas
   - 👍 Best for: Scanning large areas

**Controls:**
- `m` - Switch between modes on the fly
- `q` - Quit
- `s` - Save frame
- `p` - Pause

### Option 2: Train Custom YOLO with Proper Dataset **⭐ BEST ACCURACY**

For production use, train YOLO specifically for tomato leaves.

#### Step 1: Get a Leaf Detection Dataset

**Option A: Use Existing Datasets**

1. **PlantDoc Dataset** (Recommended)
   ```bash
   # Download from: https://github.com/pratikkayal/PlantDoc-Dataset
   # Contains leaf images with bounding boxes
   ```

2. **PlantVillage + Annotations**
   ```bash
   # Use PlantVillage images and create annotations
   # Tool: LabelImg or Roboflow
   ```

3. **Roboflow Universe**
   ```bash
   # Search for "tomato leaf detection" datasets
   # URL: https://universe.roboflow.com/
   # Many pre-annotated leaf datasets available
   ```

**Option B: Create Your Own Dataset**

1. **Collect Images** (100-500 images minimum)
   - Take photos of tomato leaves from different angles
   - Various lighting conditions
   - Different disease stages
   - Mix of healthy and diseased leaves

2. **Annotate with LabelImg**
   ```bash
   # Install LabelImg
   pip install labelImg
   
   # Run annotation tool
   labelImg
   
   # Draw bounding boxes around each leaf
   # Label them all as "leaf"
   # Save in YOLO format
   ```

3. **Organize Dataset**
   ```
   leaf_dataset/
   ├── images/
   │   ├── train/
   │   │   ├── img001.jpg
   │   │   ├── img002.jpg
   │   │   └── ...
   │   └── val/
   │       ├── img101.jpg
   │       └── ...
   └── labels/
       ├── train/
       │   ├── img001.txt
       │   ├── img002.txt
       │   └── ...
       └── val/
           ├── img101.txt
           └── ...
   ```

#### Step 2: Train Custom YOLO

```bash
# Create dataset config
python train_yolo_leaf.py --mode create-yaml --dataset-path ./leaf_dataset

# Train on Jetson Nano or GPU machine
python train_yolo_leaf.py --mode train --epochs 100 --device 0

# Or on CPU (much slower)
python train_yolo_leaf.py --mode train --epochs 100 --device cpu
```

#### Step 3: Use Custom Model

```bash
# Use your trained model
python live_camera_inference.py \
    --yolo-model runs/leaf_detection/train/weights/best.pt \
    --display
```

### Option 3: Use Pre-trained Leaf Detection Model

Some communities have shared leaf detection models:

```bash
# Example: Download from Roboflow or Ultralytics Hub
# Then use with your script
python live_camera_inference.py --yolo-model path/to/leaf_model.pt --display
```

## 📊 Comparison

| Method | Accuracy | Speed | Setup Time | Training Required |
|--------|----------|-------|------------|-------------------|
| Simplified (color) | Medium | Fast | 0 min | No ✅ |
| Simplified (whole) | Medium | Very Fast | 0 min | No ✅ |
| Custom YOLO | High | Fast | Hours-Days | Yes ⚠️ |
| Pre-trained YOLO | High | Fast | 5 min | No ✅ |

## 🚀 Recommended Workflow

### For Immediate Testing:
```bash
# Use simplified script with color detection
python live_camera_simple.py --mode color_based --display
```

### For Production:

1. **Get Dataset**
   - Download PlantDoc or similar: 30 minutes
   - Or create own: 2-4 hours for 200 images

2. **Train YOLO**
   ```bash
   python train_yolo_leaf.py --mode train --epochs 100
   ```
   - On GPU: 1-3 hours
   - On CPU: 8-24 hours (not recommended)

3. **Deploy**
   ```bash
   python live_camera_inference.py \
       --yolo-model runs/leaf_detection/train/weights/best.pt \
       --display
   ```

## 💡 Practical Tips

### For Color-Based Detection (Immediate Use)

Adjust color ranges if detection is poor:

Edit `live_camera_simple.py` lines around color detection:
```python
# Line ~72-74 - Adjust these values based on your lighting
lower_green1 = np.array([25, 40, 40])   # [Hue, Sat, Val]
upper_green1 = np.array([85, 255, 255])

# For darker leaves:
lower_green1 = np.array([25, 30, 20])

# For bright lighting:
lower_green1 = np.array([30, 50, 60])
```

Adjust minimum area:
```bash
# Detect smaller leaves
python live_camera_simple.py --mode color_based --min-area 2000 --display

# Only detect large leaves
python live_camera_simple.py --mode color_based --min-area 10000 --display
```

### Camera Setup

For best results:
- **Distance**: 20-40cm from leaves
- **Lighting**: Even, diffused light (avoid harsh shadows)
- **Background**: Plain background helps color detection
- **Angle**: Slightly above, looking down at leaves

## 🎬 Quick Commands

```bash
# Try simplified version NOW (no training):
python live_camera_simple.py --mode color_based --display

# Switch modes during runtime by pressing 'm'

# If you have a custom YOLO model:
python live_camera_inference.py --yolo-model your_model.pt --display

# Test with static image first:
python final_tomato_model/inference_jetson.py --image test_leaf.jpg
```

## 📚 Resources for Leaf Detection Datasets

1. **PlantDoc**: https://github.com/pratikkayal/PlantDoc-Dataset
2. **Roboflow Universe**: https://universe.roboflow.com/ (search "leaf detection")
3. **LabelImg Tool**: https://github.com/heartexlabs/labelImg
4. **YOLO Training Guide**: https://docs.ultralytics.com/modes/train/

## 🆘 Which One Should I Use?

**Right now for testing?**
→ `python live_camera_simple.py --mode color_based --display`

**For production deployment on Jetson Nano?**
→ Train custom YOLO or use simplified version

**Don't have time to train?**
→ `live_camera_simple.py` works well for most cases

**Need highest accuracy?**
→ Train custom YOLO with proper leaf dataset

---

**Start here:**
```bash
# Test all three modes and see which works best for you
python live_camera_simple.py --mode whole_frame --display
python live_camera_simple.py --mode color_based --display  
python live_camera_simple.py --mode grid --display
```

The simplified script is ready to use NOW and works reasonably well without any training! 🌱

