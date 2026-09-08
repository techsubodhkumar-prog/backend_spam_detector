from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from model import predict_email

app = Flask(__name__)
CORS(app)  # Allows your portfolio frontend to make POST requests to this API

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    email = data.get("email", "")

    if not email.strip():
        return jsonify({
            "error": "Please enter an email."
        }), 400

    prediction, confidence = predict_email(email)

    return jsonify({
        "prediction": prediction,
        "confidence": round(confidence * 100, 2)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)