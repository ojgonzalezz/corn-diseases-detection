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
from typing import Dict, Any, Set, Optional
from collections import defaultdict
import ast
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg_preprocess
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_preprocess
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
# Environment variables loading
sys.path.append(os.path.abspath(os.path.join("..", "src")))
from src.core.config import config
from src.utils.paths import paths

#################################################
# ---- Carga de modelo embedding prentrenado ----
#################################################

def _get_embedding_model():
    """
    Carga el modelo de embedding basado en la configuración.

    Returns:
        Modelo de TensorFlow/Keras configurado para generar embeddings.
    """
    model_name = config.data.embedding_model
    weights = config.data.embedding_weights
    include_top = config.data.embedding_include_top
    pooling = config.data.embedding_pooling

    if model_name == "ResNet50":
        return ResNet50(weights=weights, include_top=include_top, pooling=pooling)
    elif model_name == "VGG16":
        return VGG16(weights=weights, include_top=include_top, pooling=pooling)
    elif model_name == "MobileNetV2":
        return MobileNetV2(weights=weights, include_top=include_top, pooling=pooling)
    else:
        raise ValueError(f"Modelo de embedding no soportado: {model_name}")

def _get_preprocess_function():
    """
    Retorna la función de preprocesamiento correspondiente al modelo configurado.

    Returns:
        Función de preprocesamiento de TensorFlow/Keras.
    """
    model_name = config.data.embedding_model

    if model_name == "ResNet50":
        return resnet_preprocess
    elif model_name == "VGG16":
        return vgg_preprocess
    elif model_name == "MobileNetV2":
        return mobilenet_preprocess
    else:
        raise ValueError(f"Modelo de embedding no soportado: {model_name}")

# Cargar modelo de embedding según configuración
model = _get_embedding_model()
preprocess_input = _get_preprocess_function()
sizes = config.data.image_size

def preprocess_pil(img, target_size: Optional[tuple] = None):
    """
    Convierte una PIL.Image a tensor listo para el modelo de embedding.

    Args:
        img: Imagen PIL a procesar.
        target_size: Tamaño objetivo (ancho, alto). Si es None, usa IMAGE_SIZE de la configuración.

    Returns:
        Tensor preprocesado listo para el modelo de embedding.
    """
    if target_size is None:
        target_size = config.data.image_size

    img_resized = img.resize(target_size)
    x = image.img_to_array(img_resized)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def get_embedding(img):
    """
    Genera el embedding de una imagen PIL usando el modelo configurado.

    Args:
        img: Imagen PIL a procesar.

    Returns:
        Vector de embedding generado por el modelo.
    """
    tensor = preprocess_pil(img)
    emb = model.predict(tensor, verbose=0)
    return emb[0]


#######################################
# ---- Detector de similares ----------
#######################################

def detect_similar_images_embeddings(datasets, threshold: Optional[float] = None):
    """
    Detecta pares de imágenes similares usando embeddings + coseno.

    Args:
        datasets: dict -> salida de load_raw_data()
        threshold: float -> similitud mínima para considerar imágenes similares.
                   Si es None, usa IM_SIM_THRESHOLD de la configuración.

    Returns:
        Dict[str, Dict[str, List[Tuple]]]: Diccionario con pares similares por dataset/categoría
    """
    # Usar threshold de configuración si no se proporciona
    if threshold is None:
        threshold = config.data.similarity_threshold

    # Obtener categorías de la configuración
    try:
        categories = set(config.data.class_names)
    except Exception:
        categories = set()
        for dataset_key in datasets.keys():
            categories.update(datasets[dataset_key]["images"].keys())

    # 1. Inicialización de las estructuras de embeddings y pares
    # Usaremos una lista simple para los pares y una para los embeddings a comparar
    embeddings_store = {} # Almacena {uid: embedding} para la comparación
    pairs = {} # Almacena el resultado final {dataset_key: {category: List}}

    print("[BUSQUEDA] Fase 1: Generando Embeddings para datasets 'augmented'...")

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
                    print(f"[ERROR] Error al generar embedding para {uid}: {e}")
                    # No añadimos este embedding ni su UID si falla

            # 3. Comparar Embeddings y detectar pares
            print(f"  [INFO] Comparando {len(embeddings_store[category])} embeddings en '{dataset_key}/{category}'...")
            
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
            print(f"  [OK] Pares detectados: {len(detected_pairs)} pares encontrados.")
            
    return pairs


##################################################
# ---- Filtro de imagenes altamente similares ----
##################################################

def filter_similar_images(datasets: Dict[str, Any], threshold: Optional[float] = None) -> Dict[str, Any]:
    """
    Filtra los datasets para remover imágenes altamente similares (posibles aumentaciones duplicadas).

    Utiliza los resultados de detect_similar_images_embeddings. Prioriza la eliminación
    de la segunda imagen de cada par (la más nueva o secundaria).

    Args:
        datasets (Dict): Diccionario con los datos brutos cargados (data_1, data_2, etc.).
        threshold (float, optional): Umbral de similitud. Si es None, usa IM_SIM_THRESHOLD de la configuración.

    Returns:
        Dict: Una nueva estructura de datasets con imágenes duplicadas eliminadas.
    """
    # Usar threshold de configuración si no se proporciona
    if threshold is None:
        threshold = config.data.similarity_threshold

    print(f"[CONFIG] Fase de Filtrado: Detectando y eliminando imágenes similares (umbral={threshold:.2f}).")

    # 1. Ejecutar la detección de pares similares
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

    print(f"[ELIMINAR] Total de imágenes a remover (duplicados detectados): {len(uids_to_remove)}")
    
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

    print("[OK] Filtrado de duplicados completado. Los datasets filtrados han sido retornados.")

    print("\n[CARGA] Iniciando exportación a 'data/processed'...")
    # Usar sistema centralizado de rutas
    PROCESSED_ROOT = paths.data_processed
    paths.ensure_dir(PROCESSED_ROOT)

    for dataset_key, data in filtered_datasets.items():
        if not dataset_key.startswith("data_"):
            continue

        # Crear subdirectorio: data/processed/data_1 o data/processed/data_2
        dataset_output_dir = PROCESSED_ROOT / dataset_key

        for category, img_list in data["images"].items():
            # Crear subdirectorio de clase: data/processed/data_1/Blight
            class_output_dir = dataset_output_dir / category
            class_output_dir.mkdir(parents=True, exist_ok=True)

            print(f"  Guardando {len(img_list)} imágenes en: {paths.relative_to_root(class_output_dir)}")

            # Guardar cada imagen
            for i, img in enumerate(img_list):
                # Usar un nombre de archivo que incluya el índice original
                file_name = f"{category}_{i:04d}.png"
                file_path = class_output_dir / file_name
                try:
                    # Guardar la imagen (asumimos que es un objeto PIL.Image)
                    img.save(file_path)
                except Exception as e:
                    print(f"    [ADVERTENCIA] Fallo al guardar {file_name}: {e}")

    print("[OK] Exportación a 'data/processed' completada.")
    return filtered_datasets