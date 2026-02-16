import torch
import onnx
import onnxruntime as ort
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import os

def convert_model_to_onnx(model_path="./final_tomato_model", output_path="./final_tomato_model/model.onnx"):
    """
    Convert the fine-tuned Swinv2 model to ONNX format for Jetson Nano deployment.
    
    Args:
        model_path: Path to the saved model directory
        output_path: Path where the ONNX model will be saved
    """
    print("="*60)
    print("Converting Model to ONNX for Jetson Nano")
    print("="*60)
    
    # 1. Load the model and processor
    print(f"\n[1/5] Loading model from {model_path}...")
    model = AutoModelForImageClassification.from_pretrained(model_path)
    image_processor = AutoImageProcessor.from_pretrained(model_path)
    model.eval()
    
    print(f"✓ Model loaded: {model.config.architectures[0]}")
    print(f"✓ Number of classes: {model.config.num_labels}")
    print(f"✓ Input size: {image_processor.size['height']}x{image_processor.size['height']}")
    
    # 2. Create dummy input
    print("\n[2/5] Creating dummy input for export...")
    batch_size = 1
    height = image_processor.size["height"]
    width = image_processor.size["height"]
    dummy_input = torch.randn(batch_size, 3, height, width)
    
    # 3. Export to ONNX
    print(f"\n[3/5] Exporting to ONNX format...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,  # Opset 14 is well-supported on Jetson Nano
        do_constant_folding=True,  # Optimization for inference
        input_names=['pixel_values'],
        output_names=['logits'],
        dynamic_axes={
            'pixel_values': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        },
        verbose=False
    )
    print(f"✓ ONNX model exported to: {output_path}")
    
    # 4. Verify ONNX model
    print("\n[4/5] Verifying ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model is valid")
    
    # 5. Test inference with ONNX Runtime
    print("\n[5/5] Testing ONNX Runtime inference...")
    ort_session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
    
    # Test with dummy input
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)
    
    # Compare with PyTorch output
    with torch.no_grad():
        pytorch_output = model(dummy_input).logits
    
    # Check if outputs match
    np.testing.assert_allclose(
        pytorch_output.numpy(), 
        ort_outputs[0], 
        rtol=1e-03, 
        atol=1e-05
    )
    print("✓ ONNX Runtime outputs match PyTorch outputs")
    
    # Print model info
    print("\n" + "="*60)
    print("Model Information for Jetson Nano Deployment")
    print("="*60)
    print(f"Input name: {ort_session.get_inputs()[0].name}")
    print(f"Input shape: {ort_session.get_inputs()[0].shape}")
    print(f"Input type: {ort_session.get_inputs()[0].type}")
    print(f"Output name: {ort_session.get_outputs()[0].name}")
    print(f"Output shape: {ort_session.get_outputs()[0].shape}")
    print(f"Output type: {ort_session.get_outputs()[0].type}")
    
    # Get file size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nONNX Model Size: {file_size_mb:.2f} MB")
    
    print("\n" + "="*60)
    print("Conversion Complete!")
    print("="*60)
    print(f"\n📦 ONNX model ready for Jetson Nano: {output_path}")
    print("\n💡 Next Steps for Jetson Nano Deployment:")
    print("   1. Copy the ONNX model and preprocessor_config.json to your Jetson Nano")
    print("   2. Install ONNX Runtime: pip install onnxruntime")
    print("   3. Use the inference example below to run predictions")
    
    return output_path, image_processor

def create_inference_example(model_path="./final_tomato_model"):
    """Create a sample inference script for Jetson Nano"""
    
    inference_code = '''"""
Inference script for ONNX model on Jetson Nano
Usage: python inference_onnx.py --image path/to/image.jpg
"""
import onnxruntime as ort
import numpy as np
from PIL import Image
import json
import argparse

def preprocess_image(image_path, image_size=256):
    """Preprocess image for model input"""
    # Load and resize image
    image = Image.open(image_path).convert('RGB')
    image = image.resize((image_size, image_size))
    
    # Convert to numpy array and normalize
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Normalize using ImageNet stats (same as training)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    
    # Transpose to CHW format and add batch dimension
    img_array = img_array.transpose(2, 0, 1)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict(model_path, image_path, config_path):
    """Run inference on Jetson Nano"""
    # Load ONNX model with CUDA execution provider for GPU acceleration
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(model_path, providers=providers)
    
    # Load label mapping
    with open(config_path, 'r') as f:
        config = json.load(f)
    id2label = config['id2label']
    
    # Preprocess image
    input_data = preprocess_image(image_path)
    
    # Run inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_data})
    logits = outputs[0][0]
    
    # Get prediction
    predicted_class_idx = np.argmax(logits)
    confidence = np.exp(logits[predicted_class_idx]) / np.sum(np.exp(logits))  # Softmax
    predicted_label = id2label[str(predicted_class_idx)]
    
    return predicted_label, confidence, logits

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ONNX inference on Jetson Nano")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default="model.onnx", help="Path to ONNX model")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json")
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    print(f"Processing image: {args.image}")
    
    label, confidence, logits = predict(args.model, args.image, args.config)
    
    print(f"\\nPrediction: {label}")
    print(f"Confidence: {confidence:.2%}")
    print(f"\\nAll class probabilities:")
    
    # Load config for all labels
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Softmax for all classes
    probs = np.exp(logits) / np.sum(np.exp(logits))
    for idx, prob in enumerate(probs):
        print(f"  {config['id2label'][str(idx)]}: {prob:.2%}")
'''
    
    inference_path = os.path.join(model_path, "inference_jetson.py")
    with open(inference_path, 'w') as f:
        f.write(inference_code)
    
    print(f"\n📝 Inference example created: {inference_path}")

if __name__ == "__main__":
    # Convert the model
    onnx_path, processor = convert_model_to_onnx()
    
    # Create inference example
    create_inference_example()
    
    print("\n✅ All done! Your model is ready for Jetson Nano deployment.")
