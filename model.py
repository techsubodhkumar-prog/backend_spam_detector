import os
import pickle
import gdown

# ============================================================
# Memory Optimization Flags for Low-RAM Hosting (512MB RAM)
# ============================================================
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

# Disable GPU allocation entirely
tf.config.set_visible_devices([], "GPU")

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import preprocess_text

MAX_LEN = 100
MODEL_PATH = "email_spam_detection.keras"
FILE_ID = "1klLTDj9ghspnmsYs5QU090b81iaA_Aut"
URL = f"https://drive.google.com/uc?id={FILE_ID}"

# Download model if not present
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(URL, MODEL_PATH, quiet=False)

print("Loading TensorFlow model...")
model = load_model(MODEL_PATH, compile=False)
print("Model loaded successfully.")

# Load Tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

def predict_email(text):
    cleaned = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    output = model(padded, training=False)
    probability = float(output.numpy()[0][0])

    if probability >= 0.5:
        return "Spam", probability
    else:
        return "Ham", 1 - probability