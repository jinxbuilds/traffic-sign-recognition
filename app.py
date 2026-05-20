from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from PIL import Image
from preprocess import preprocess

import numpy as np
import os
import cv2


app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

# Load model
try:
    model = load_model(os.path.join("model", "stage2.h5"))
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Traffic sign classes (GTSRB dataset - 43 classes)
classes = {
    0: "Speed limit (20 km/h)",
    1: "Speed limit (30 km/h)",
    2: "Speed limit (50 km/h)",
    3: "Speed limit (60 km/h)",
    4: "Speed limit (70 km/h)",
    5: "Speed limit (80 km/h)",
    6: "End of speed limit (80 km/h)",
    7: "Speed limit (100 km/h)",
    8: "Speed limit (120 km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No entry",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry for vehicles over 3.5 metric tons",
    18: "Dangerous curve to the left",
    19: "Dangerous curve to the right",
    20: "Dangerous curve ahead",
    21: "Children crossing",
    22: "Bicycles crossing",
    23: "Slippery road",
    24: "Steep descent",
    25: "Road works",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Animals crossing",
    29: "Speed limit ends (80 km/h)",
    30: "Speed limit (100 km/h) ends",
    31: "Speed limit (120 km/h) ends",
    32: "Keep right",
    33: "Keep left",
    34: "Roundabout mandatory",
    35: "End of no passing",
    36: "End of no passing for vehicles over 3.5 metric tons",
    37: "Pedestrian crossing",
    38: "Level crossing",
    39: "Level crossing with gates",
    40: "Railway crossing",
    41: "Level crossing without gates",
    42: "Bumpy road"
}

from skimage import exposure

def preprocess_image(image):

    # Convert PIL image to numpy array
    image = np.array(image)

    # Resize exactly like training
    image = cv2.resize(image, (32, 32))

    # Use SAME preprocessing as training
    X, _ = preprocess(np.array([image]))

    return X

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        # Check if image is provided
        if "file" not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files["file"]
        
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400
        
        # Open and preprocess image
        image = Image.open(file)
        processed = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(processed, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = float(np.max(prediction))
        
        result = classes.get(predicted_class, "Unknown class")
        
        return jsonify({
            "prediction": result,
            "class_id": int(predicted_class),
            "confidence": round(confidence * 100, 2)
        })
    
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port
    )