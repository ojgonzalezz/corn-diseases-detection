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
from src.utils.errors import *

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
    MODEL_PATH = PROJECT_ROOT / "models" / "Exported" / "best_VGG16.keras" 
except NoModelToLoadError:
    pass

try:
    LABELS = env_vars["CLASS_NAMES"]
    _labels = ast.literal_eval(LABELS)
except NoLabelsError:
    pass

with tf.device('/CPU:0'):
    # Dentro de este bloque, todas las operaciones se ejecutarán en la CPU
    _model = tf.keras.models.load_model(MODEL_PATH)

# --- Carga de modelo ---
_model = tf.keras.models.load_model(MODEL_PATH)
#_labels = LABELS


# ---- Image preprocessing ----
def preprocess_image(file_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
    return arr


#---- Inference function ----
def predict(file_bytes: bytes):
    try:
        x = preprocess_image(file_bytes)
        preds = _model.predict(x, verbose=0)[0]
        probs = preds.astype(float).tolist()
        idx = int(np.argmax(preds))
        
        # Validación de consistencia
        num_classes = len(preds)
        expected_classes = 4  # Tu modelo tiene 4 clases
        
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

