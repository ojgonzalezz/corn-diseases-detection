#####################################################################################
# ---------------------- Data augmentation detection utilities ----------------------
#####################################################################################


#########################
# ---- Depdendencies ----
#########################

import os
import sys
from PIL import Image
import pathlib
from typing import Dict, Any, Set
from collections import defaultdict
import ast
import tensorflow as tf
import numpy as np
import ast
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
# Environment variables loading
sys.path.append(os.path.abspath(os.path.join("..", "src")))
from src.core.load_env import EnvLoader

#################################################
# ---- Carga de modelo embedding prentrenado ----
#################################################

model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
env_vars = EnvLoader().get_all()
sizes = tuple(ast.literal_eval(env_vars.get("IMAGE_SIZE", "[]")))

def preprocess_pil(img, target_size=(224, 224)):
    """
    Convierte una PIL.Image a tensor listo para ResNet50.
    """
    img_resized = img.resize(target_size)
    x = image.img_to_array(img_resized)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def get_embedding(img):
    """
    Genera el embedding de una imagen PIL usando ResNet50.
    """
    tensor = preprocess_pil(img)
    emb = model.predict(tensor, verbose=0)
    return emb[0]


#######################################
# ---- Detector de similares ----------
#######################################

def detect_similar_images_embeddings(datasets, threshold=0.95):
    """
    Detecta pares de imágenes similares usando embeddings + coseno.
    
    Params:
    - datasets: dict -> salida de load_raw_data()
    - threshold: float -> similitud mínima para considerar imágenes similares
    
    Returns:
    - Dict[str, Dict[str, List[Tuple]]]: Diccionario con pares similares por dataset/categoría
    """
    
    # Cargar env_vars dentro de la función o pasar como argumento
    env_vars = EnvLoader().get_all()
    
    try:
        categories = set(ast.literal_eval(env_vars.get("CLASS_NAMES", "[]")))
    except Exception:
        categories = set()
        for dataset_key in datasets.keys():
            categories.update(datasets[dataset_key]["images"].keys())

    # 1. Inicialización de las estructuras de embeddings y pares
    # Usaremos una lista simple para los pares y una para los embeddings a comparar
    embeddings_store = {} # Almacena {uid: embedding} para la comparación
    pairs = {} # Almacena el resultado final {dataset_key: {category: List}}

    print("🔎 Fase 1: Generando Embeddings para datasets 'augmented'...")

    # 2. Generar embeddings SOLAMENTE para datasets 'augmented'
    for dataset_key, dataset in datasets.items():
        pairs[dataset_key] = {} # Inicializar la estructura de salida para el dataset
        
        if dataset.get("dataset_consideration") != "augmented":
            # Ignorar data_1 (no-augmentation) o datasets no etiquetados para esta fase
            for category in categories:
                 pairs[dataset_key][category] = []
            continue 

        # Procesar datasets AUMENTADOS
        for category, img_list in dataset["images"].items():
            
            embeddings_store[category] = [] # Lista temporal para embeddings de esta categoría
            uid_map = [] # Mapeo de índices a IDs únicos para esta categoría

            for i, img in enumerate(img_list):
                uid = f"{dataset_key}/{category}/{i}"
                
                try:
                    emb = get_embedding(img)
                    embeddings_store[category].append(emb)
                    uid_map.append(uid)
                except Exception as e:
                    print(f"❌ Error al generar embedding para {uid}: {e}")
                    # No añadimos este embedding ni su UID si falla

            # 3. Comparar Embeddings y detectar pares
            print(f"  ✨ Comparando {len(embeddings_store[category])} embeddings en '{dataset_key}/{category}'...")
            
            if len(embeddings_store[category]) < 2:
                pairs[dataset_key][category] = []
                continue
                
            matrix = np.array(embeddings_store[category])
            sims = cosine_similarity(matrix)

            # 4. Buscar pares con similitud >= threshold
            detected_pairs = []
            keys_len = len(uid_map)
            for i in range(keys_len):
                for j in range(i + 1, keys_len):
                    if sims[i, j] >= threshold:
                        detected_pairs.append((uid_map[i], uid_map[j], sims[i, j]))

            pairs[dataset_key][category] = detected_pairs
            print(f"  ✅ Pares detectados: {len(detected_pairs)} pares encontrados.")
            
    return pairs


##################################################
# ---- Filtro de imagenes altamente similares ----
##################################################

def filter_similar_images(datasets: Dict[str, Any], threshold=0.95) -> Dict[str, Any]:
    """
    Filtra los datasets para remover imágenes altamente similares (posibles aumentaciones duplicadas).

    Utiliza los resultados de detect_similar_images_embeddings. Prioriza la eliminación 
    de la segunda imagen de cada par (la más nueva o secundaria).

    Args:
        datasets (Dict): Diccionario con los datos brutos cargados (data_1, data_2, etc.).

    Returns:
        Dict: Una nueva estructura de datasets con imágenes duplicadas eliminadas.
    """
    print("🔬 Fase de Filtrado: Detectando y eliminando imágenes similares (duplicados).")

    # 1. Ejecutar la detección de pares similares
    # Se usa el umbral por defecto (0.95) o se ajusta si es necesario
    similar_pairs_results = detect_similar_images_embeddings(datasets, threshold)

    # 2. Recolectar UIDs para eliminación
    # Guardaremos los UIDs de las imágenes que queremos descartar.
    uids_to_remove: Set[str] = set()
    
    for dataset_key, categories_data in similar_pairs_results.items():
        if dataset_key.startswith("data_") and datasets[dataset_key].get("dataset_consideration") == "augmented":
            
            for category, pairs_list in categories_data.items():
                if not pairs_list:
                    continue
                    
                for uid1, uid2, similarity in pairs_list:
                    # En la tupla (uid1, uid2, similitud), descartamos uid2 para conservar uid1.
                    # Esto garantiza que al menos una versión se mantenga.
                    uids_to_remove.add(uid2)

    print(f"🗑️ Total de imágenes a remover (duplicados detectados): {len(uids_to_remove)}")
    
    # 3. Construir el nuevo dataset filtrado
    filtered_datasets = {}

    for dataset_key, dataset in datasets.items():
        filtered_datasets[dataset_key] = {
            "dataset_consideration": dataset.get("dataset_consideration"),
            "images": defaultdict(list)
        }
        
        if not dataset_key.startswith("data_"):
            # Si hay otras claves que no son data_1, data_2, etc., simplemente copiarlas
            filtered_datasets[dataset_key] = dataset
            continue

        for category, img_list in dataset["images"].items():
            
            # Reiniciar el contador de imágenes para esta categoría
            kept_count = 0
            
            for i, img in enumerate(img_list):
                # Generar el UID que se utilizó para la detección
                uid = f"{dataset_key}/{category}/{i}"
                
                # 4. Decidir si mantener o descartar
                if uid not in uids_to_remove:
                    filtered_datasets[dataset_key]["images"][category].append(img)
                    kept_count += 1
            
            removed_count = len(img_list) - kept_count
            print(f"  [{dataset_key}/{category}] Mantenidas: {kept_count}, Eliminadas: {removed_count}")

    print("✅ Filtrado de duplicados completado. Los datasets filtrados han sido retornados.")

    print("\n📦 Iniciando exportación a 'data/processed'...")
    try:
        # Se asume que el script está en src/utils/
        PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

    except NameError:
        # Fallback si el script se ejecuta directamente
        PROJECT_ROOT = pathlib.Path(os.getcwd())

    PROCESSED_ROOT = PROJECT_ROOT / 'data' / 'processed'
    
    for dataset_key, data in filtered_datasets.items():
        if not dataset_key.startswith("data_"):
            continue

        # Crear subdirectorio: data/processed/data_1 o data/processed/data_2
        dataset_output_dir = PROCESSED_ROOT / dataset_key
        
        for category, img_list in data["images"].items():
            # Crear subdirectorio de clase: data/processed/data_1/Blight
            class_output_dir = dataset_output_dir / category
            class_output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"  Guardando {len(img_list)} imágenes en: {class_output_dir.relative_to(PROJECT_ROOT)}")

            # Guardar cada imagen
            for i, img in enumerate(img_list):
                # Usar un nombre de archivo que incluya el índice original
                file_name = f"{category}_{i:04d}.png" 
                file_path = class_output_dir / file_name
                try:
                    # Guardar la imagen (asumimos que es un objeto PIL.Image)
                    img.save(file_path)
                except Exception as e:
                    print(f"    ⚠️ Fallo al guardar {file_name}: {e}")

    print("✅ Exportación a 'data/processed' completada.")
    return filtered_datasets