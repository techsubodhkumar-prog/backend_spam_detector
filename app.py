from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
# Enable CORS for all origins
CORS(app)

@app.route("/", methods=["GET"])
def home():
    return "Backend is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "email" not in data:
            return jsonify({"error": "No email text provided."}), 400
        
        email_text = data["email"]
        
        # --- YOUR ML MODEL PREDICTION LOGIC HERE ---
        # Example output structure:
        # return jsonify({"prediction": "Spam", "confidence": 95.4})
        
        return jsonify({"prediction": "Ham", "confidence": 98.2})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)