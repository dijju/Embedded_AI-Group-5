# Tomato Disease Classification Web Application

A simple web interface for classifying tomato leaf diseases using a fine-tuned Swinv2 transformer model.

## Features

- 🖼️ Drag & drop or click to upload images
- 🔍 Real-time classification with confidence scores
- 📊 Display top 5 predictions
- 🎨 Modern, responsive UI
- ⚡ Fast inference using ONNX Runtime

## Disease Categories

The model can detect the following tomato leaf conditions:
1. Bacterial Spot
2. Early Blight
3. Late Blight
4. Leaf Mold
5. Septoria Leaf Spot
6. Spider Mites (Two-spotted spider mite)
7. Target Spot
8. Tomato Yellow Leaf Curl Virus
9. Tomato Mosaic Virus
10. Healthy

## Installation

1. Install the required packages:
```bash
pip install -r requirements_web.txt
```

## Running the Application

1. Make sure you're in the Final_Project directory
2. Run the Flask application:
```bash
python app.py
```

3. Open your web browser and navigate to:
```
http://localhost:5000
```

## Usage

1. Click on the upload area or drag & drop an image of a tomato leaf
2. Click the "Classify Image" button
3. View the classification results with confidence scores
4. Upload another image to classify more samples

## Technical Details

- **Backend**: Flask (Python)
- **Model**: Swinv2 Transformer (ONNX format)
- **Input Size**: 256x256 pixels
- **Inference**: ONNX Runtime (CPU)
- **Frontend**: HTML5, CSS3, Vanilla JavaScript

## File Structure

```
Final_Project/
├── app.py                      # Flask web application
├── templates/
│   └── index.html             # Frontend HTML
├── static/
│   └── style.css              # Styling
├── final_tomato_model/
│   ├── model.onnx             # ONNX model
│   └── config.json            # Model configuration
└── requirements_web.txt        # Python dependencies
```

## Notes

- The application uses CPU for inference by default
- Supported image formats: JPG, PNG, JPEG
- The model expects RGB images
- Images are automatically resized to 256x256 pixels

## Troubleshooting

If you encounter any issues:

1. Make sure all dependencies are installed: `pip install -r requirements_web.txt`
2. Verify that `model.onnx` exists in the `final_tomato_model` folder
3. Check that you're running the application from the correct directory
4. Ensure port 5000 is not being used by another application
