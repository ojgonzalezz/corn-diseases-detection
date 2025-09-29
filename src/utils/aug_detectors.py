#####################################################################################
# ---------------------- Data augmentation detection utilities ----------------------
#####################################################################################


#########################
# ---- Depdendencies ----
#########################

import os
from PIL import Image
import imagehash
import tensorflow as tf
import numpy as np
import ast
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
from src.core.load_env import EnvLoader

#####################################
# ---- Image embedding distances ----
#####################################


model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
env_vars = EnvLoader().get_all()
sizes = tuple(ast.literal_eval(env_vars.get("IMAGE_SIZE", "[]")))

def get_embedding(img_path, target_size=sizes):
    """
    Convierte una imagen en un embedding usando ResNet50.
    """
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return model.predict(x, verbose=0)[0]

def detect_similar_images_embeddings(image_dir, threshold=0.95):
    """
    Detecta imágenes similares en un directorio usando embeddings + similitud coseno.
    
    Params:
    - image_dir: carpeta con imágenes
    - threshold: similitud coseno mínima para considerar dos imágenes como similares
    
    Returns:
    - Lista de pares (img1, img2, similitud)
    """
    embeddings = {}
    similar_pairs = []

    # Extraer embeddings
    for fname in os.listdir(image_dir):
        path = os.path.join(image_dir, fname)
        try:
            emb = get_embedding(path)
            embeddings[fname] = emb
        except Exception as e:
            print(f"Error con {fname}: {e}")
            continue

    # Comparar embeddings
    files = list(embeddings.keys())
    for i in range(len(files)):
        for j in range(i+1, len(files)):
            sim = cosine_similarity(
                embeddings[files[i]].reshape(1, -1),
                embeddings[files[j]].reshape(1, -1)
            )[0][0]
            if sim >= threshold:
                similar_pairs.append((files[i], files[j], sim))

    return similar_pairs


# Ejemplo de uso
if __name__ == "__main__":
    similares = detect_similar_images_embeddings("ruta/a/imagenes", threshold=0.95)
    for img1, img2, sim in similares:
        print(f"Posibles aumentaciones: {img1} ↔ {img2} (sim={sim:.3f})")