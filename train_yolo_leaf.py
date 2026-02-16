"""
Train Custom YOLOv8 Model for Leaf Detection

This script helps you train a YOLOv8 model specifically for detecting leaves.
You'll need a dataset with annotated leaf images.
"""
import os
from ultralytics import YOLO
import yaml
import torch


def create_dataset_yaml(dataset_path, output_path='leaf_dataset.yaml'):
    """
    Create YAML configuration file for YOLO training
    
    Expected dataset structure:
    dataset_path/
        images/
            train/
                image1.jpg
                image2.jpg
            val/
                image3.jpg
        labels/
            train/
                image1.txt
                image2.txt
            val/
                image3.txt
    """
    
    dataset_config = {
        'path': os.path.abspath(dataset_path),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,  # Number of classes (1 for leaf)
        'names': ['leaf']  # Class names
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"Created dataset configuration: {output_path}")
    return output_path


def train_yolo_leaf_detector(
    dataset_yaml='leaf_dataset.yaml',
    model_size='n',  # n, s, m, l, x
    epochs=100,
    img_size=640,
    batch_size=16,
    device='auto'  # 'auto', 'cpu', or GPU device number
):
    """
    Train YOLOv8 model for leaf detection
    
    Args:
        dataset_yaml: Path to dataset YAML configuration
        model_size: Model size (n=nano, s=small, m=medium, l=large, x=xlarge)
        epochs: Number of training epochs
        img_size: Input image size
        batch_size: Batch size (reduce if GPU memory is limited)
        device: Device to use ('auto', 'cpu', or GPU device number)
    """
    
    # Auto-detect CUDA availability
    if device == 'auto':
        if torch.cuda.is_available():
            device = '0'
            print(f"✓ CUDA detected: Using GPU (device 0)")
        else:
            device = 'cpu'
            print(f"⚠ CUDA not available: Using CPU (training will be slower)")
    
    print("\n" + "="*60)
    print("Training YOLOv8 Leaf Detection Model")
    print("="*60)
    print(f"Model Size: YOLOv8{model_size}")
    print(f"Epochs: {epochs}")
    print(f"Image Size: {img_size}")
    print(f"Batch Size: {batch_size}")
    print(f"Device: {device}")
    print("="*60 + "\n")
    
    # Load pre-trained YOLOv8 model
    model = YOLO(f'yolov8{model_size}.pt')
    
    # Train the model
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        project='runs/leaf_detection',
        name='train',
        patience=20,  # Early stopping patience
        save=True,
        plots=True,
        cache=True,  # Cache images for faster training
        workers=4,
        augment=True,
        hsv_h=0.015,  # Image HSV-Hue augmentation
        hsv_s=0.7,    # Image HSV-Saturation augmentation
        hsv_v=0.4,    # Image HSV-Value augmentation
        degrees=10,   # Image rotation (+/- deg)
        translate=0.1, # Image translation (+/- fraction)
        scale=0.5,    # Image scale (+/- gain)
        flipud=0.2,   # Image flip up-down (probability)
        fliplr=0.5,   # Image flip left-right (probability)
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best model saved at: runs/leaf_detection/train/weights/best.pt")
    print(f"Last model saved at: runs/leaf_detection/train/weights/last.pt")
    
    return results


def validate_model(model_path, dataset_yaml='leaf_dataset.yaml'):
    """Validate trained model"""
    print("\n" + "="*60)
    print("Validating Model")
    print("="*60)
    
    model = YOLO(model_path)
    results = model.val(data=dataset_yaml)
    
    print(f"\nValidation Results:")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    
    return results


def test_inference(model_path, test_image_path):
    """Test model inference on a single image"""
    print("\n" + "="*60)
    print("Testing Model Inference")
    print("="*60)
    
    model = YOLO(model_path)
    results = model(test_image_path)
    
    # Display results
    for result in results:
        result.show()
        result.save(filename='test_result.jpg')
    
    print(f"Test result saved as: test_result.jpg")
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train YOLOv8 for leaf detection")
    parser.add_argument("--mode", type=str, default="train", 
                       choices=['train', 'validate', 'test', 'create-yaml'],
                       help="Operation mode")
    parser.add_argument("--dataset-path", type=str, default="./leaf_dataset",
                       help="Path to dataset directory")
    parser.add_argument("--dataset-yaml", type=str, default="leaf_dataset.yaml",
                       help="Path to dataset YAML file")
    parser.add_argument("--model-size", type=str, default="n",
                       choices=['n', 's', 'm', 'l', 'x'],
                       help="YOLOv8 model size (n=nano, s=small, m=medium, l=large, x=xlarge)")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--img-size", type=int, default=640,
                       help="Input image size")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use ('auto' for auto-detect, 'cpu' for CPU, or GPU device number like '0')")
    parser.add_argument("--model-path", type=str, default="runs/leaf_detection/train/weights/best.pt",
                       help="Path to trained model (for validation/testing)")
    parser.add_argument("--test-image", type=str, default="test.jpg",
                       help="Path to test image")
    
    args = parser.parse_args()
    
    if args.mode == 'create-yaml':
        create_dataset_yaml(args.dataset_path, args.dataset_yaml)
        
    elif args.mode == 'train':
        # Create YAML if it doesn't exist
        if not os.path.exists(args.dataset_yaml):
            print(f"Dataset YAML not found. Creating: {args.dataset_yaml}")
            create_dataset_yaml(args.dataset_path, args.dataset_yaml)
        
        # Train model
        train_yolo_leaf_detector(
            dataset_yaml=args.dataset_yaml,
            model_size=args.model_size,
            epochs=args.epochs,
            img_size=args.img_size,
            batch_size=args.batch_size,
            device=args.device
        )
        
    elif args.mode == 'validate':
        validate_model(args.model_path, args.dataset_yaml)
        
    elif args.mode == 'test':
        test_inference(args.model_path, args.test_image)
    
    print("\nDone!")
