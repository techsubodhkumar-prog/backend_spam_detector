import os
import gdown
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODEL_PATH = "spam_model.h5"
# Replace with your actual Google Drive File ID
GDRIVE_FILE_ID = "YOUR_GOOGLE_DRIVE_FILE_ID_HERE"

def ensure_model_downloaded():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model weights from Google Drive...")
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    email = data.get("email", "")

    if not email.strip():
        return jsonify({"error": "Please enter an email."}), 400

    # Ensure model file is downloaded
    ensure_model_downloaded()

    # Import prediction logic internally to conserve RAM at startup
    from model import predict_email
    prediction, confidence = predict_email(email)

    return jsonify({
        "prediction": prediction,
        "confidence": round(confidence * 100, 2)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)