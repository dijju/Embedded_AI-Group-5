# Quick Reference: Which Script Should I Use?

## 🚦 Decision Tree

```
Do you have leaves to detect right now?
│
├─ YES, I want to test immediately
│  └─> Use: live_camera_simple.py --mode color_based
│      ✅ Works now, no training needed
│      ✅ Detects green regions automatically
│
├─ YES, but camera shows one leaf at a time
│  └─> Use: live_camera_simple.py --mode whole_frame
│      ✅ Fastest option
│      ✅ Processes entire frame
│
├─ YES, but I have a trained YOLO leaf model
│  └─> Use: live_camera_inference.py --yolo-model your_model.pt
│      ✅ Most accurate
│      ✅ Best for production
│
└─ NO, I want to train custom YOLO first
   └─> Follow: LEAF_DETECTION_GUIDE.md
       ⚠️ Takes time to prepare dataset
       ✅ Best long-term solution
```

## 📝 Command Quick Reference

### Immediate Testing (No Setup)
```bash
# Simplified - Color detection
python live_camera_simple.py --mode color_based --display

# Simplified - Whole frame
python live_camera_simple.py --mode whole_frame --display

# Simplified - Grid
python live_camera_simple.py --mode grid --display
```

### With Custom YOLO
```bash
# After training (see LEAF_DETECTION_GUIDE.md)
python live_camera_inference.py \
    --yolo-model runs/leaf_detection/train/weights/best.pt \
    --display
```

### Testing
```bash
# Check if everything works
python test_setup.py

# Test web interface (single images)
python app.py
```

## 🎮 Runtime Controls

All live camera scripts support:
- **Q** - Quit
- **S** - Save current frame
- **P** - Pause/Resume

The simplified script also has:
- **M** - Switch detection mode on the fly

## 📊 Feature Comparison

| Feature | live_camera_simple.py | live_camera_inference.py |
|---------|----------------------|-------------------------|
| **Ready to use** | ✅ Yes | ⚠️ Needs YOLO model |
| **Training required** | ❌ No | ✅ Yes (for accuracy) |
| **Multi-leaf detection** | ✅ Yes | ✅ Yes |
| **Speed (CPU)** | ⚡ Fast | ⚡ Fast |
| **Speed (GPU)** | ⚡⚡ Very Fast | ⚡⚡ Very Fast |
| **Accuracy** | 🟡 Good | 🟢 Excellent (with training) |
| **Lighting sensitive** | ⚠️ Yes (color mode) | ✅ More robust |
| **Setup time** | 0 minutes | Hours (dataset + training) |
| **Best for** | Testing, demos | Production |

## 🎯 Recommended Path

### Today (Testing Phase):
```bash
# Use simplified version
python live_camera_simple.py --mode color_based --display

# Try different modes to see what works best
# Press 'm' to switch modes during runtime
```

### This Week (If Deploying):
1. Test simplified version thoroughly
2. Decide if accuracy is sufficient
3. If YES → Deploy simplified version
4. If NO → Collect leaf dataset and train YOLO

### For Production:
```bash
# Option 1: Simplified (if accuracy is good enough)
python live_camera_simple.py --mode color_based

# Option 2: Custom YOLO (if need highest accuracy)
python live_camera_inference.py --yolo-model trained_model.pt
```

## 🔧 Troubleshooting

### "Not detecting leaves" with color_based mode
```bash
# Adjust minimum area
python live_camera_simple.py --mode color_based --min-area 2000 --display

# Or edit the HSV color ranges in the script
# See LEAF_DETECTION_GUIDE.md for details
```

### "Too slow"
```bash
# Use whole_frame mode (fastest)
python live_camera_simple.py --mode whole_frame --display

# Or reduce resolution
python live_camera_simple.py --mode color_based --width 640 --height 480 --display
```

### "YOLO not detecting anything"
```bash
# Switch to simplified version instead
python live_camera_simple.py --mode color_based --display

# Or train custom YOLO (see LEAF_DETECTION_GUIDE.md)
```

## 📚 Documentation Map

- **[QUICKSTART.md](QUICKSTART.md)** - Get started quickly
- **[LEAF_DETECTION_GUIDE.md](LEAF_DETECTION_GUIDE.md)** - Detailed leaf detection solutions
- **[README_CAMERA.md](README_CAMERA.md)** - Full documentation
- **[WINDOWS_CPU_GUIDE.md](WINDOWS_CPU_GUIDE.md)** - Running without GPU
- **This file** - Quick decisions and commands

## 💡 Pro Tips

1. **Start Simple**: Begin with `whole_frame` mode to verify classification works
2. **Test Modes**: Try all three modes (m key) to see what works best
3. **Tune Colors**: If using color_based, tune HSV values for your lighting
4. **Consider Grid**: Grid mode works well when leaves are spread across frame
5. **Production Ready**: Simplified version is production-ready for many use cases

---

## TL;DR

**Start here:**
```bash
python live_camera_simple.py --mode color_based --display
```

Press **m** to try different modes. Use what works best. 🌱

**Need highest accuracy?** 
See [LEAF_DETECTION_GUIDE.md](LEAF_DETECTION_GUIDE.md) to train custom YOLO.
