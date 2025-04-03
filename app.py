from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import tensorflow as tf
import numpy as np
import os
import tempfile
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import logging
from waitress import serve  # Production server
from dotenv import load_dotenv  # For environment variables

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure CORS properly
CORS(app, resources={
    r"/transcribe": {"origins": os.getenv("ALLOWED_ORIGINS", "*")},
    r"/predict": {"origins": os.getenv("ALLOWED_ORIGINS", "*")}
})

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Model Loading ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

def load_models():
    """Load models with error handling"""
    try:
        # Whisper model
        if not hasattr(app, 'whisper_model'):
            logger.info("Loading Whisper model...")
            app.whisper_model = whisper.load_model("base")
            logger.info("Whisper model loaded!")

        # Text model
        if not hasattr(app, 'text_model'):
            logger.info("Loading Text Classifier model...")
            model_path = r'C:\Users\Soham\Desktop\Soham\atp_idc\flask backend\models\ai_text_classifier.keras'
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")
            app.text_model = tf.keras.models.load_model(model_path)
            logger.info("Text Classifier model loaded!")

    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        raise

# Initialize models
load_models()

# --- Routes --- (Keep your existing /transcribe, /predict, /health routes unchanged)
# ... [Your existing route code remains exactly the same] ...

# --- Main ---
if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    environment = os.getenv("FLASK_ENV", "development")
    
    if environment == "production":
        # Production server (Waitress)
        logger.info(f"Starting production server on port {port}")
        serve(app, host="0.0.0.0", port=port)
    else:
        # Development server
        logger.info(f"Starting development server on port {port}")
        app.run(host="0.0.0.0", port=port, debug=True)