import os
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return jsonify({
        "status": "online",
        "message": "Spam Detector API is running!"
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    email = data.get("email", "")

    if not email.strip():
        return jsonify({"error": "Please enter an email."}), 400

    from model import predict_email
    prediction, confidence = predict_email(email)

    return jsonify({
        "prediction": prediction,
        "confidence": round(confidence * 100, 2)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)