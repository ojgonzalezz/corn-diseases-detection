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

from src.core.config import config
from src.utils.paths import paths

# Excepciones personalizadas para el pipeline de inferencia
class NoModelToLoadError(Exception):
    """Se lanza cuando no se encuentra el archivo del modelo."""
    pass

class NoLabelsError(Exception):
    """Se lanza cuando no se pueden cargar las etiquetas de clases desde la configuración."""
    pass

#####################
# ---- Inference ----
#####################

# ---- Config ----
try:
    IMG_SIZE = config.data.image_size
    if not isinstance(IMG_SIZE, tuple) or len(IMG_SIZE) != 2:
        raise ValueError("IMAGE_SIZE must be a tuple of two integers.")
except (ValueError, AttributeError) as e:
    IMG_SIZE = (224, 224)

try:
    # Usar sistema centralizado de rutas
    MODEL_PATH = paths.get_model_path("best_VGG16.keras")
    if not MODEL_PATH.exists():
        raise NoModelToLoadError(f"Archivo del modelo no encontrado en {MODEL_PATH}")
except NoModelToLoadError as e:
    print(f"[ADVERTENCIA] {e}")
    MODEL_PATH = None

try:
    _labels = config.data.class_names
    if not _labels:
        raise NoLabelsError("CLASS_NAMES no encontrado en configuración")
except (NoLabelsError, AttributeError) as e:
    print(f"[ADVERTENCIA] No se pudieron cargar las etiquetas - {e}")
    _labels = None

# Obtener NUM_CLASSES de la configuración
try:
    NUM_CLASSES = config.data.num_classes
except (ValueError, TypeError, AttributeError):
    NUM_CLASSES = 4
    print(f"[ADVERTENCIA] No se pudo parsear NUM_CLASSES. Usando valor por defecto: {NUM_CLASSES}")

# --- Carga de modelo ---
_model = None
if MODEL_PATH and MODEL_PATH.exists():
    try:
        with tf.device('/CPU:0'):
            # Dentro de este bloque, todas las operaciones se ejecutarán en la CPU
            _model = tf.keras.models.load_model(MODEL_PATH)
        print(f"[OK] Modelo cargado exitosamente desde {MODEL_PATH}")
    except Exception as e:
        print(f"[ERROR] Error al cargar modelo: {e}")
        _model = None
else:
    print("[ADVERTENCIA] Ruta del modelo no configurada o archivo no existe. La inferencia no funcionará hasta que se entrene el modelo.")


# ---- Image preprocessing ----
def preprocess_image(file_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
    return arr


#---- Función de inferencia ----
def predict(file_bytes: bytes):
    """
    Predice la clase de una imagen de hoja de maíz.

    Args:
        file_bytes: Bytes crudos de la imagen

    Returns:
        dict: Resultados de predicción con etiqueta, confianza y probabilidades
    """
    if _model is None:
        return {
            "error": "Modelo no cargado",
            "message": "Archivo del modelo no encontrado. Entrene el modelo primero."
        }

    try:
        x = preprocess_image(file_bytes)
        preds = _model.predict(x, verbose=0)[0]
        probs = preds.astype(float).tolist()
        idx = int(np.argmax(preds))
        
        # Validación de consistencia
        num_classes = len(preds)
        expected_classes = NUM_CLASSES  # Obtener de configuración en lugar de hardcodeado

        if num_classes != expected_classes:
            print(f"[ADVERTENCIA] Modelo devuelve {num_classes} clases, se esperaban {expected_classes}")

        if _labels:
            if len(_labels) != expected_classes:
                print(f"[ADVERTENCIA] {len(_labels)} etiquetas para {expected_classes} clases")
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

