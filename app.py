from flask import Flask, request, jsonify
import joblib
import numpy as np
from PIL import Image
import json
import io
from flask_cors import CORS
import cv2 

app = Flask(__name__)
CORS(app)
# Load model components
scaler = joblib.load("models/scaler.pkl")
pca = joblib.load("models/pca.pkl")
model = joblib.load("models/digit_model.pkl")

# Load facts
with open("facts.json", "r") as f:
    facts = json.load(f)

@app.route("/predict", methods=["POST"])
def predict_digit():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    # Read & preprocess
    # 1️⃣ Invert image if needed (white digit on black background)
    if np.mean(img) > 127:  # means background is light
        img = cv2.bitwise_not(img)

    # 2️⃣ Threshold to binary (remove gray background)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3️⃣ Find bounding box around the digit (crop)
    coords = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(coords)
    img = img[y:y+h, x:x+w]

    # 4️⃣ Resize to 20x20 (keeping aspect ratio)
    img = cv2.resize(img, (20, 20), interpolation=cv2.INTER_AREA)

    # 5️⃣ Pad to 28x28 (MNIST style)
    img_padded = np.pad(img, ((4, 4), (4, 4)), mode='constant', constant_values=0)

    # 6️⃣ Flatten & scale
    img_flat = img_padded.flatten().astype('float32').reshape(1, -1)
    img_scaled = scaler.transform(img_flat)
    img_pca = pca.transform(img_scaled)
    pred = int(model.predict(img_pca)[0])

    return jsonify({
        "digit": pred,
        "fact": facts[str(pred)]
    })

if __name__ == "__main__":
    app.run(debug=True)
