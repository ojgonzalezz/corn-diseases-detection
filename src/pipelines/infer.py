#####################################################################################
# -------------------------------- Inference Pipeline -------------------------------
#####################################################################################

##########################
# ---- Depedendencies ----
##########################

import io 
import ast
import pathlib
import numpy as np
from PIL import Image
import tensorflow as tf

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from src.core.load_env import EnvLoader

# Custom Exceptions for Inference Pipeline
class NoModelToLoadError(Exception):
    """Raised when model file cannot be found."""
    pass

class NoLabelsError(Exception):
    """Raised when class labels cannot be loaded from configuration."""
    pass

#####################
# ---- Inference ----
#####################

# ---- Config ----
env_vars = EnvLoader().get_all()
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

try:
    IMG_SIZE = ast.literal_eval(env_vars.get("IMAGE_SIZE", None))
    if not isinstance(IMG_SIZE, tuple) or len(IMG_SIZE) != 2:
        raise ValueError("IMAGE_SIZE must be a tuple of two integers.")
except (ValueError, SyntaxError) as e:
    IMG_SIZE = (224, 224)

try:
    MODEL_PATH = PROJECT_ROOT / "models" / "exported" / "best_VGG16.keras"
    if not MODEL_PATH.exists():
        raise NoModelToLoadError(f"Model file not found at {MODEL_PATH}")
except NoModelToLoadError as e:
    print(f"⚠️ Warning: {e}")
    MODEL_PATH = None

try:
    LABELS = env_vars.get("CLASS_NAMES")
    if not LABELS:
        raise NoLabelsError("CLASS_NAMES not found in environment variables")
    _labels = ast.literal_eval(LABELS)
except (NoLabelsError, KeyError, ValueError, SyntaxError) as e:
    print(f"⚠️ Warning: Could not load labels - {e}")
    _labels = None

# Get NUM_CLASSES from config
try:
    NUM_CLASSES = int(env_vars.get("NUM_CLASSES", 4))
except (ValueError, TypeError):
    NUM_CLASSES = 4
    print(f"⚠️ Warning: Could not parse NUM_CLASSES. Using default: {NUM_CLASSES}")

# --- Carga de modelo ---
_model = None
if MODEL_PATH and MODEL_PATH.exists():
    try:
        with tf.device('/CPU:0'):
            # Dentro de este bloque, todas las operaciones se ejecutarán en la CPU
            _model = tf.keras.models.load_model(MODEL_PATH)
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        _model = None
else:
    print("⚠️ Warning: Model path not set or file doesn't exist. Inference will not work until model is trained.")


# ---- Image preprocessing ----
def preprocess_image(file_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
    return arr


#---- Inference function ----
def predict(file_bytes: bytes):
    """
    Predict the class of a corn leaf image.

    Args:
        file_bytes: Raw image bytes

    Returns:
        dict: Prediction results with label, confidence, and probabilities
    """
    if _model is None:
        return {
            "error": "Model not loaded",
            "message": "Model file not found. Please train the model first."
        }

    try:
        x = preprocess_image(file_bytes)
        preds = _model.predict(x, verbose=0)[0]
        probs = preds.astype(float).tolist()
        idx = int(np.argmax(preds))
        
        # Validación de consistencia
        num_classes = len(preds)
        expected_classes = NUM_CLASSES  # Get from config instead of hardcoded

        if num_classes != expected_classes:
            print(f"⚠️  Advertencia: Modelo devuelve {num_classes} clases, esperaba {expected_classes}")
        
        if _labels:
            if len(_labels) != expected_classes:
                print(f"⚠️  Advertencia: {len(_labels)} labels para {expected_classes} clases")
            label = _labels[idx] if idx < len(_labels) else f"class_{idx}"
        else:
            label = str(idx)
        
        # Crear diccionario con probabilidades
        class_probs = {}
        if _labels and len(_labels) == expected_classes:
            for i, class_label in enumerate(_labels):
                class_probs[class_label] = probs[i] if i < len(probs) else 0.0
        else:
            for i in range(min(num_classes, expected_classes)):
                class_probs[f"class_{i}"] = probs[i]
        
        return {
            "predicted_label": label,
            "predicted_index": idx,
            "confidence": float(probs[idx]),
            "all_probabilities": class_probs,
            "raw_probabilities": probs
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "message": "Error durante la predicción"
        }

