import os
import pickle
import gdown
from utils import preprocess_text

MAX_LEN = 100
MODEL_PATH = "email_spam_detection.keras"
FILE_ID = "1klLTDj9ghspnmsYs5QU090b81iaA_Aut"
URL = f"https://drive.google.com/uc?id={FILE_ID}"

# Global placeholders for lazy loading
_model = None
_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        with open("tokenizer.pkl", "rb") as f:
            _tokenizer = pickle.load(f)
    return _tokenizer


def get_model():
    global _model
    if _model is None:
        # Download model if it does not exist locally
        if not os.path.exists(MODEL_PATH):
            print("Downloading model from Google Drive...")
            gdown.download(URL, MODEL_PATH, quiet=False)

        # Import TensorFlow inside function to optimize RAM
        import tensorflow as tf
        from tensorflow.keras.models import load_model

        # Disable GPU allocation for CPU-only execution
        tf.config.set_visible_devices([], "GPU")

        print("Loading Keras model into memory...")
        _model = load_model(MODEL_PATH, compile=False)
        print("Model loaded successfully.")

    return _model


def predict_email(text):
    # 1. Preprocess raw text
    cleaned = preprocess_text(text)

    # 2. Tokenize text input
    tokenizer = get_tokenizer()
    sequence = tokenizer.texts_to_sequences([cleaned])

    # 3. Pad sequence to fixed length
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    padded = pad_sequences(
        sequence, 
        maxlen=MAX_LEN, 
        padding="post", 
        truncating="post"
    )

    # 4. Generate prediction using cached model
    model = get_model()
    output = model(padded, training=False)
    probability = float(output.numpy()[0][0])

    # 5. Output classification and confidence
    if probability >= 0.5:
        return "Spam", probability
    else:
        return "Ham", 1 - probability