# Archivo: src/api.py

import tensorflow as tf
import numpy as np
import pathlib
from PIL import Image
import io
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# --- 1. CONFIGURACIÓN E INICIALIZACIÓN ---
app = FastAPI(title="API de Detección de Enfermedades en Maíz", version="1.0")

# Permitir solicitudes CORS para que el frontend pueda comunicarse con la API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite cualquier origen
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el modelo de IA al iniciar la API
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / 'models' / 'fine_tuned_best_model.keras'

print(f"🧠 Cargando modelo desde: {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Modelo cargado exitosamente.")

# Definir los nombres de las clases en el orden correcto
CLASS_NAMES = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']


def preprocess_image(image_bytes: bytes, target_size=(224, 224)) -> np.ndarray:
    """
    Toma los bytes de una imagen, la preprocesa y la prepara para el modelo.
    """
    # Abrir la imagen desde los bytes en memoria
    img = Image.open(io.BytesIO(image_bytes))
    
    # Asegurarse de que la imagen tenga 3 canales (RGB)
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    # Redimensionar la imagen
    img = img.resize(target_size)
    
    # Convertir a un array de numpy y normalizar
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    
    # Añadir una dimensión de lote (batch) para que el modelo la acepte
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# --- 2. DEFINIR EL ENDPOINT DE PREDICCIÓN ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Recibe un archivo de imagen, realiza una predicción y devuelve los resultados.
    """
    # Leer el contenido del archivo en memoria
    image_bytes = await file.read()
    
    # Preprocesar la imagen
    processed_image = preprocess_image(image_bytes)
    
    # Realizar la predicción
    predictions = model.predict(processed_image)
    
    # Obtener el índice de la clase con la mayor probabilidad
    predicted_class_index = int(np.argmax(predictions, axis=1)[0])
    
    # Obtener el nombre de la clase
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    
    # Obtener la confianza de la predicción
    confidence = float(np.max(predictions))
    
    return {
        "predicted_class": predicted_class_name,
        "confidence": f"{confidence:.2%}",
        "all_probabilities": {CLASS_NAMES[i]: f"{float(predictions[0][i]):.2%}" for i in range(len(CLASS_NAMES))}
    }

# --- 3. EJECUTAR LA API ---
if __name__ == '__main__':
    print("🚀 Iniciando servidor de API con Uvicorn...")
    uvicorn.run(app, host="127.0.0.1", port=8000)