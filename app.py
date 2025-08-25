import os
import io
import json
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# Load model and class indices
model = load_model("models/plant_disease_model.h5")
with open("models/class_indices.json") as f:
    class_indices = json.load(f)
idx_to_class = {v: k for k, v in class_indices.items()}

def predict_image(image_path):
    """For predictions using a file path (disk-based)."""
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    class_id = np.argmax(preds)
    class_name = idx_to_class[class_id]
    confidence = float(preds[class_id])
    return {"label": class_name, "prob": confidence}

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")  # Ensure index.html is in templates/

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # ✅ Read image directly from memory (no saving to disk)
    img_bytes = file.read()
    img = load_img(io.BytesIO(img_bytes), target_size=(224, 224))

    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    class_id = np.argmax(preds)
    class_name = idx_to_class[class_id]
    confidence = float(preds[class_id])

    return jsonify({"top": {"label": class_name, "prob": confidence}})

if __name__ == "__main__":
    app.run(debug=True)
