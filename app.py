"""
Flask Web Application for Tomato Disease Classification
"""
from flask import Flask, render_template, request, jsonify
import onnxruntime as ort
import numpy as np
from PIL import Image
import json
import io
import os

app = Flask(__name__)

# Configuration
MODEL_PATH = os.path.join('final_tomato_model', 'model.onnx')
CONFIG_PATH = os.path.join('final_tomato_model', 'config.json')

# Load config at startup
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)
    id2label = config['id2label']

# Initialize ONNX session
providers = ['CPUExecutionProvider']  # Use CPU for web deployment
session = ort.InferenceSession(MODEL_PATH, providers=providers)

def preprocess_image(image, image_size=256):
    """Preprocess image for model input"""
    # Resize image
    image = image.convert('RGB')
    image = image.resize((image_size, image_size))
    
    # Convert to numpy array and normalize
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Normalize using ImageNet stats (ensure float32)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    
    # Transpose to CHW format and add batch dimension
    img_array = img_array.transpose(2, 0, 1)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Ensure final array is float32
    return img_array.astype(np.float32)

def predict(image):
    """Run inference on the image"""
    # Preprocess image
    input_data = preprocess_image(image)
    
    # Run inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_data})
    logits = outputs[0][0]
    
    # Calculate softmax probabilities
    probs = np.exp(logits) / np.sum(np.exp(logits))
    
    # Get top prediction
    predicted_class_idx = np.argmax(logits)
    confidence = float(probs[predicted_class_idx])
    predicted_label = id2label[str(predicted_class_idx)]
    
    # Get all class probabilities sorted by confidence
    all_predictions = []
    for idx, prob in enumerate(probs):
        all_predictions.append({
            'label': id2label[str(idx)],
            'confidence': float(prob)
        })
    
    # Sort by confidence
    all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
    
    return {
        'prediction': predicted_label,
        'confidence': confidence,
        'all_predictions': all_predictions
    }

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_image():
    """Handle image upload and prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        try:
            # Read and process the image
            image = Image.open(io.BytesIO(file.read()))
            
            # Get prediction
            result = predict(image)
            
            return jsonify(result)
        
        except Exception as e:
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500

if __name__ == '__main__':
    print("="*60)
    print("Tomato Disease Classification Web App")
    print("="*60)
    print(f"Model loaded: {MODEL_PATH}")
    print(f"Number of classes: {len(id2label)}")
    print("\nStarting server on http://localhost:5000")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
