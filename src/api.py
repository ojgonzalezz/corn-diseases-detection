# Archivo: src/api.py

import tensorflow as tf
import numpy as np
import pathlib
from PIL import Image
import io
import requests # <-- Importación nueva
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# --- 1. CONFIGURACIÓN E INICIALIZACIÓN ---
app = FastAPI(title="API de Detección de Enfermedades en Maíz", version="1.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- DESCARGA Y CARGA DEL MODELO ---
MODEL_DIR = pathlib.Path("downloaded_model")
MODEL_PATH = MODEL_DIR / "fine_tuned_best_model.keras"
# ▼▼▼ PEGA AQUÍ TU ENLACE DE HUGGING FACE ▼▼▼
MODEL_URL = "https://huggingface.co/felipepflorezo/corn-disease-classifier/resolve/main/fine_tuned_best_model.keras" 

def download_model(url, path):
    """Descarga el modelo si no existe localmente."""
    if not path.exists():
        print(f"🧠 Modelo no encontrado. Descargando desde {url}...")
        path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(url)
        response.raise_for_status() # Lanza un error si la descarga falla
        with open(path, "wb") as f:
            f.write(response.content)
        print("✅ Modelo descargado exitosamente.")
    else:
        print("🧠 Modelo ya existe localmente.")

# Ejecutar la descarga al iniciar la API
download_model(MODEL_URL, MODEL_PATH)

print(f"🧠 Cargando modelo desde: {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Modelo cargado exitosamente.")

CLASS_NAMES = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

# (El resto del código de preprocess_image y predict se mantiene exactamente igual...)
def preprocess_image(image_bytes: bytes, target_size=(224, 224)) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != 'RGB': img = img.convert('RGB')
    img = img.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    processed_image = preprocess_image(image_bytes)
    predictions = model.predict(processed_image)
    predicted_class_index = int(np.argmax(predictions, axis=1)[0])
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence = float(np.max(predictions))
    return {
        "predicted_class": predicted_class_name,
        "confidence": f"{confidence:.2%}",
        "all_probabilities": {CLASS_NAMES[i]: f"{float(predictions[0][i]):.2%}" for i in range(len(CLASS_NAMES))}
    }

if __name__ == '__main__':
    print("🚀 Iniciando servidor de API con Uvicorn...")
    uvicorn.run(app, host="127.0.0.1", port=8000)