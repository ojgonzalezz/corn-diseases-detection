#####################################################################################
# -------------------------- Data Preprocessing Utilities ---------------------------
#####################################################################################


#########################
# ---- Depdendencies ----
#########################

import pathlib
import shutil
import ast
import math
import random
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import numpy as np
from PIL import Image
from src.adapters.data_loader import load_raw_data, load_split_data, load_split_data_paths
from src.utils.image_modifier import ImageAugmentor
from src.utils.data_augmentator import DataAugmenter
from src.core.config import config
from src.utils.aug_detectors import filter_similar_images
from src.utils.utils import stratified_split_dataset
from src.utils.paths import paths


##################################
# ---- Data Spliting Function ----
##################################

def data_split(datasets: Dict[str, Any], environment_variables: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orquesta el pipeline de procesamiento de datos: filtra imágenes duplicadas 
    por embedding (de-augmentación), unifica los datasets y realiza una división 
    estratificada para generar los conjuntos de Train, Val y Test.
    """

    # 1. Preparación de Metadatos y Variables de Entorno
    env_varss = environment_variables
    
    # 1a. Carga y parseo de CATEGORIES
    try:
        categories = set(ast.literal_eval(env_varss.get("CLASS_NAMES", "[]")))
        if not categories:
            raise ValueError("CLASS_NAMES no contiene categorías válidas.")
    except (ValueError, SyntaxError, KeyError) as e:
        # Fallback robusto: Derivar categorías directamente de los datos
        print(f"[ADVERTENCIA] No se pudo parsear CLASS_NAMES del entorno ({e}). Derivando de los datos...")
        categories = set()
        for dataset_key in datasets.keys():
            categories.update(datasets[dataset_key]["images"].keys())

    # 1b. Carga y parseo de SPLIT RATIOS
    try:
        split_ratios_str = env_varss.get("SPLIT_RATIOS", "(0.7, 0.15, 0.15)")
        split_ratios = tuple(ast.literal_eval(split_ratios_str))
        if not (isinstance(split_ratios, tuple) and len(split_ratios) == 3 and math.isclose(sum(split_ratios), 1.0)):
            raise ValueError("Los ratios deben ser una tupla de 3 elementos que sumen 1.0.")
    except (ValueError, SyntaxError, TypeError) as e:
        split_ratios = (0.7, 0.15, 0.15)
        print(f"[ADVERTENCIA] Split ratios: Error al cargar ({e}). Usando el tuple de emergencia: {split_ratios}")
        
    # 1c. Carga de umbral de similaridad (threshold)
    try:
        # Intentar primero con el nombre correcto, luego con typo legacy
        sim_threshold = env_varss.get("IM_SIM_THRESHOLD")
        if sim_threshold is None:
            # Fallback a typo legacy (deprecado)
            sim_threshold = env_varss.get("IM_SIM_TRESHOLD", "0.95")
            if env_varss.get("IM_SIM_TRESHOLD"):
                print("[ADVERTENCIA] IM_SIM_TRESHOLD está deprecado. Usa IM_SIM_THRESHOLD en su lugar.")
        sim_threshold = float(ast.literal_eval(sim_threshold))
    except (ValueError, SyntaxError, TypeError) as e:
        sim_threshold = 0.95
        print(f"[ADVERTENCIA] Umbral de similaridad: Error al cargar ({e}). Usando el valor por defecto: {sim_threshold}")


    print(f"[EVAL] Configuración de Split: Ratios={split_ratios}, Umbral de Filtrado={sim_threshold}")
    raw_keys_to_iterate = [key for key in datasets.keys() if key.startswith("data_")]

    # 2. De-aumentar (filtrar) imágenes duplicadas
    print("\n[BUSQUEDA] Fase 1: Filtrando imágenes similares (De-augmentación)...")
    datasets_aug_filtered = filter_similar_images(datasets, sim_threshold) # Asume que filter_similar_images acepta el threshold

    # 3. Inicialización y Unificación del Dataset
    print("[PROCESO] Fase 2: Unificando datasets por categoría...")
    all_images_dataset = {
        cat: [] for cat in categories
    }
    
    for dataset_key in raw_keys_to_iterate:
        #consideration = datasets_aug_filtered[dataset_key]["dataset_consideration"]
        
        for cat in categories:
            # Asegurar que la categoría exista antes de iterar
            image_list = datasets_aug_filtered[dataset_key]["images"].get(cat)
            
            if image_list is None:
                # La categoría no existe en este dataset
                continue

            for img in image_list:
                all_images_dataset[cat].append(img)


    # 4. Data split estratificado
    print("\n[SPLIT] Fase 3: Realizando el Split Estratificado...")
    splited_dataset = stratified_split_dataset(all_images_dataset, split_ratios)

    print("\n[OK] Pipeline de Data Split y Filtrado completado.")
    return splited_dataset

######################################
# ---- Data Downsampling Function ----
######################################

def downsample_dataset(split_datasets: Dict[str, Dict[str, List[Any]]], environment_variables: Dict[str, Any]) -> Dict[str, Dict[str, List[Any]]]:
    """
    Realiza submuestreo exclusivamente en el conjunto 'train' del data split para balancear 
    las clases a la misma cantidad que la clase minoritaria.

    Args:
        split_datasets (Dict): Diccionario con los conjuntos divididos ('train', 'val', 'test').
                               Cada conjunto interno es un diccionario {'categoria': [lista de imágenes]}.
        environment_variables (Dict): Diccionario con variables de entorno (usado para obtener categorías si es necesario).

    Returns:
        Dict: El diccionario de splits actualizado con el conjunto 'train' balanceado.
    """
    
    TRAIN_SET_KEY = 'train'
    if TRAIN_SET_KEY not in split_datasets:
        raise ValueError("El diccionario de entrada debe contener la clave 'train'.")
        
    # Extraer el conjunto de entrenamiento para la modificación
    train_data = split_datasets[TRAIN_SET_KEY]
    
    # 1. Determinar el tamaño mínimo objetivo (el tamaño de la clase minoritaria)
    category_lengths = {
        cat: len(train_data.get(cat, []))
        for cat in train_data.keys()
    }
    
    # Filtrar categorías vacías antes de buscar el mínimo para evitar mínimo=0 si hay una clase sin datos
    non_empty_lengths = [length for length in category_lengths.values() if length > 0]
    if not non_empty_lengths:
        print("[ADVERTENCIA] Advertencia: El conjunto de entrenamiento está vacío. No se aplicó downsampling.")
        return split_datasets
        
    minimum_lenght = min(non_empty_lengths)
    
    print(f"\n[BALANCE]  Iniciando Downsampling en el conjunto '{TRAIN_SET_KEY}'.")
    print(f"[CALCULO] Tamaño objetivo por clase: {minimum_lenght} imágenes.")
    
    # 2. Inicializar el diccionario de salida downsampled (solo para el conjunto 'train')
    downsampled_train_set = defaultdict(list)

    # 3. Proceso de Submuestreo por Categoría (SOLO en 'train')
    for category, image_list in train_data.items():
        current_total = len(image_list)
        
        if current_total <= minimum_lenght:
            # Si ya está balanceada o es la clase mínima, se mantiene intacta.
            downsampled_train_set[category] = image_list.copy()
            print(f"  [{category}] Mantenida: {current_total} imágenes.")
            continue

        # Aplicar submuestreo
        random.shuffle(image_list)
        
        # Mantenemos solo el número mínimo de imágenes.
        sampled_images = image_list[:minimum_lenght]
        downsampled_train_set[category] = sampled_images
        
        print(f"  [BALANCE]  Categoría '{category}' submuestreada: {current_total} -> {minimum_lenght} imágenes.")


    # 4. Reconstruir el diccionario de splits final
    final_split_datasets = {
        TRAIN_SET_KEY: downsampled_train_set, # Usar el conjunto downsampled
        'val': split_datasets.get('val', {}),  # Copiar 'val' (sin downsampling)
        'test': split_datasets.get('test', {}) # Copiar 'test' (sin downsampling)
    }

    # 5. Generar resumen final
    final_train_count = sum(len(v) for v in final_split_datasets[TRAIN_SET_KEY].values())
    print(f"\n[OK] Downsampling de 'train' completado. Nuevo Total en 'train': {final_train_count}")
            
    return final_split_datasets


##################################################
# ---- Data Downsampling Function importancia ----
##################################################

def downsample_dataset_importancia(datasets: Dict[str, Any], environment_variables: Dict[str, Any]) -> Dict[str, Any]:
    """
    Realiza submuestreo en los datasets combinados para balancear las clases,
    preservando la estructura original de datasets separados.
    Da prioridad a mantener los datos con consideración 'no-augmentation'.

    Args:
        datasets (Dict): Diccionario con los datos brutos cargados (data_1, data_2, etc.).
        split_ratios (tuple): Ratios de división para train, val, y test.

    Returns:
        Dict: Un nuevo diccionario con la misma estructura, pero con menos imágenes.
    """
    
    # 1. Preparación de Metadatos
    if environment_variables is None:
        env_varss = EnvLoader().get_all()
    else:
        env_varss = environment_variables
    # Usar ast.literal_eval para obtener la lista de clases de forma segura
    try:
        categories = set(ast.literal_eval(env_varss.get("CLASS_NAMES", "[]")))
    except (ValueError, SyntaxError, KeyError) as e:
        # Si falla la carga, derivar categorías de los datos
        print(f"[ADVERTENCIA] Warning: Could not parse CLASS_NAMES ({e}). Deriving from data...")
        categories = set()
        for dataset_key in datasets.keys():
            categories.update(datasets[dataset_key]["images"].keys())
            
    if not categories:
        raise ValueError("No se pudieron determinar los nombres de las clases (categories).")

    raw_keys_to_iterate = [key for key in datasets.keys() if key.startswith("data_")]

    datasets_annotations = {
        dataset_key: {
            "consideration": datasets[dataset_key].get("dataset_consideration", "undefined"),
            "images": datasets[dataset_key]["images"], # Referencia a las imágenes originales
            "lenghts_per_category":{cat: len(datasets[dataset_key]["images"].get(cat, [])) for cat in categories}
            } 
        for dataset_key in raw_keys_to_iterate
    }
    
    # 2. Determinar el tamaño mínimo objetivo
    total_lenghts = {
        cat : sum(
            datasets_annotations[dataset_key]["lenghts_per_category"].get(cat, 0) for dataset_key in datasets_annotations.keys()
        )
        for cat in categories
    }

    # El mínimo total de imágenes en todas las categorías
    minimum_lenght = min(list(total_lenghts.values()))
    
    print(f"[CALCULO] Tamaño objetivo total por clase (Total): {minimum_lenght} imágenes.")
    
    # 3. Inicializar el diccionario de salida
    downsampled_datasets = {
        dataset_key: {
            "dataset_consideration": datasets[dataset_key].get("dataset_consideration", "undefined"),
            "images": defaultdict(list)
        }
        for dataset_key in raw_keys_to_iterate
    }

    # 4. Proceso de Submuestreo por Categoría
    for category in categories:
        
        # 4a. Calcular la necesidad de muestreo
        current_total = total_lenghts[category]
        if current_total <= minimum_lenght:
            # Si ya está balanceada o es la clase mínima, no hacemos submuestreo.
            for dataset_key in raw_keys_to_iterate:
                downsampled_datasets[dataset_key]["images"][category].extend(
                    datasets_annotations[dataset_key]["images"].get(category, [])
                )
            continue
            
        needed_to_remove = current_total - minimum_lenght
        
        # 4b. Identificar datasets por prioridad (Preservar 'no-augmentation')
        # Separamos los datasets en alta prioridad (no-augmentation) y baja prioridad.
        priority_datasets = {k: v for k, v in datasets_annotations.items() 
                             if v['consideration'] == 'no-augmentation'}
        low_priority_datasets = {k: v for k, v in datasets_annotations.items() 
                                 if v['consideration'] != 'no-augmentation'}

        # 4c. Recolectar todas las imágenes en listas separadas (con su clave original)
        all_images = []
        for dataset_key in raw_keys_to_iterate:
            images = datasets_annotations[dataset_key]["images"].get(category, [])
            # Guardamos cada imagen con una tupla (imagen, clave_del_dataset, prioridad)
            priority = 1 if datasets_annotations[dataset_key]['consideration'] == 'no-augmentation' else 0
            all_images.extend([(img, dataset_key, priority) for img in images])
            
        random.shuffle(all_images)
        
        # Ordenar: priorizar 'no-augmentation' (prioridad 1) al final para que sean las últimas en ser eliminadas
        # Usamos -priority para ordenar de mayor prioridad a menor, manteniendo los 'no-augmentation'
        all_images.sort(key=lambda x: -x[2]) 
        
        # 4d. Aplicar submuestreo
        # Mantenemos las 'minimum_lenght' imágenes y descartamos el resto.
        sampled_images_with_keys = all_images[:minimum_lenght]
        
        # 4e. Reasignar imágenes muestreadas a su estructura original
        for img, dataset_key, _ in sampled_images_with_keys:
            downsampled_datasets[dataset_key]["images"][category].append(img)
        
        print(f"  [BALANCE]  Categoría '{category}' submuestreada a {minimum_lenght} imágenes.")


    # 5. Generar un resumen de la nueva distribución (opcional, pero útil)
    print("\n[OK] Resumen de la Nueva Distribución Downsampled:")
    for dataset_key, data in downsampled_datasets.items():
        total = sum(len(imgs) for imgs in data["images"].values())
        print(f"  {dataset_key} ({data['dataset_consideration']}): Total {total} imágenes")
        for cat, imgs in data["images"].items():
            print(f"    - {cat}: {len(imgs)} imágenes")
            
    return downsampled_datasets


######################################
# ---- Data Oversampling Function ----
######################################

def oversample_dataset(split_datasets: Dict[str, Dict[str, List[Any]]], environment_variables: Dict[str, Any]) -> Dict[str, Dict[str, List[Any]]]:
    """
    Realiza sobremuestreo (oversampling) en el conjunto 'train' para balancear 
    las clases minoritarias, utilizando transformaciones de aumento de datos.

    El tamaño objetivo es la clase mayoritaria + un máximo definido por el entorno.

    Args:
        split_datasets (Dict): Diccionario con los conjuntos divididos ('train', 'val', 'test').
        environment_variables (Dict): Diccionario con variables de entorno (ej., 'CLASS_NAMES').

    Returns:
        Dict: El diccionario de splits actualizado con el conjunto 'train' balanceado.
    """
    
    # 1. Preparación de Metadatos
    TRAIN_SET_KEY = 'train'
    if TRAIN_SET_KEY not in split_datasets:
        raise ValueError("El diccionario de entrada debe contener la clave 'train'.")

    env_vars = environment_variables

    try:
        categories = list(ast.literal_eval(env_vars.get("CLASS_NAMES", "[]")))
    except (ValueError, SyntaxError, KeyError) as e:
        print(f"[ADVERTENCIA] Warning: Could not parse CLASS_NAMES ({e}). Using keys from train set...")
        categories = list(split_datasets[TRAIN_SET_KEY].keys())
            
    if not categories:
        raise ValueError("No se pudieron determinar los nombres de las clases (categories).")

    # Inicializar aumentadores de datos (asumiendo que están disponibles globalmente o se importan)
    augmenter = DataAugmenter() 
    image_modifier = ImageAugmentor()
    
    # Extraer el conjunto de entrenamiento para la modificación
    train_data = split_datasets[TRAIN_SET_KEY]
    
    # 2. Determinar el tamaño objetivo
    category_lengths = {
        cat: len(train_data.get(cat, []))
        for cat in train_data.keys()
    }
    
    non_empty_lengths = [length for length in category_lengths.values() if length > 0]
    if not non_empty_lengths:
        print("[ADVERTENCIA] Advertencia: El conjunto de entrenamiento está vacío. No se aplicó oversampling.")
        return split_datasets
        
    # Calcular el tamaño de la clase mayoritaria (el máximo)
    max_train_size = max(non_empty_lengths)
    
    # Obtener el límite de balanceo (MAX_ADDED_BALANCE)
    try:
        MAX_ADDED_BALANCE = int(env_vars.get("MAX_ADDED_BALANCE", 50))
    except (ValueError, TypeError) as e:
        print(f"[ADVERTENCIA] Warning: Could not parse MAX_ADDED_BALANCE ({e}). Using default: 50")
        MAX_ADDED_BALANCE = 50
        
    # [OBJETIVO] Tamaño objetivo: Clase mayoritaria + límite extra
    target_size = max_train_size + MAX_ADDED_BALANCE
    
    print(f"\n[BALANCE]  Iniciando Oversampling en el conjunto '{TRAIN_SET_KEY}'.")
    print(f"[CALCULO] Tamaño objetivo por clase: {target_size} imágenes.")
    
    # 3. Inicializar el diccionario de salida para el conjunto 'train'
    oversampled_train_set = {cat: image_list.copy() for cat, image_list in train_data.items()}

    # 4. Definir las transformaciones de calidad y espaciales
    quality_transforms = [
        image_modifier.downsample,
        image_modifier.distort,
        image_modifier.add_noise,
        image_modifier.adjust_contrast,
        image_modifier.adjust_brightness,
        image_modifier.adjust_sharpness,
    ]
    
    spatial_transforms = [
        lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
        lambda img: img.transpose(Image.FLIP_TOP_BOTTOM),
        lambda img: img.rotate(random.randint(-30, 30)),
    ]

    # 5. Proceso de Sobremuestreo por Categoría (SOLO en 'train')
    for category in categories:
        # Usamos el conjunto *copiado* para añadir las imágenes
        images_list = oversampled_train_set[category]
        current_total = len(images_list)
        
        if current_total >= target_size:
            # 5a. Límite: Si es la clase mayoritaria o ya superó el target, limitamos.
            random.shuffle(images_list)
            oversampled_train_set[category] = images_list[:target_size]
            print(f"  [{category}] Limitada: {current_total} -> {target_size} imágenes.")
            continue

        # 5b. Sobremuestreo (Clase Minoritaria)
        needed = target_size - current_total
        
        if needed > 0:
            original_images = images_list.copy() # Copia de las originales para muestrear
            
            for _ in tqdm(range(needed), desc=f"Aumentando '{category}'", unit="img"):
                img = random.choice(original_images)
                
                # Paso 1: Aplicar dos transformaciones de calidad consecutivamente
                chosen_quality_transforms = random.sample(quality_transforms, 2)
                transformed_img = img
                
                for transform_func in chosen_quality_transforms:
                    img_np = np.array(transformed_img)
                    try:
                        # Manejo de argumentos específicos
                        if transform_func == image_modifier.distort:
                            transformed_img_np = image_modifier.distort(
                                img_np, axis=random.choice(['horizontal', 'vertical'])
                            )
                        elif transform_func == image_modifier.adjust_color_intensity:
                            transformed_img_np = image_modifier.adjust_color_intensity(
                                img_np, channel=random.randint(0, 2)
                            )
                        else:
                            transformed_img_np = transform_func(img_np)
                    except TypeError:
                         transformed_img_np = transform_func(img_np)
                         
                    transformed_img = Image.fromarray(transformed_img_np)

                # Paso 2: Aplicar una transformación espacial aleatoria
                chosen_spatial_transform = random.choice(spatial_transforms)
                final_augmented_img = chosen_spatial_transform(transformed_img)

                # Añadir la imagen aumentada
                oversampled_train_set[category].append(final_augmented_img)
        
            print(f"  [GRAFICO] Categoría '{category}' aumentada: {current_total} -> {target_size} imágenes.")

    # 6. Reconstruir el diccionario de splits final
    final_split_datasets = {
        TRAIN_SET_KEY: oversampled_train_set, # Usar el conjunto sobremuestreado
        'val': split_datasets.get('val', {}),  # Copiar 'val' sin cambios
        'test': split_datasets.get('test', {}) # Copiar 'test' sin cambios
    }

    # 7. Generar resumen final
    final_train_count = sum(len(v) for v in final_split_datasets[TRAIN_SET_KEY].values())
    print(f"\n[OK] Oversampling de 'train' completado. Nuevo Total en 'train': {final_train_count}")
            
    return final_split_datasets


##############################################
# ---- Split and balance dataset Function ----
##############################################

def split_and_balance_dataset(balanced: str = "downsample", split_ratios: tuple = None, use_existing_split: bool = True) -> Dict[str, Any]:
    """
    Realiza una división estratificada del dataset, aplica la estrategia de balanceo
    (downsample, oversample, o ninguno) solo al conjunto de entrenamiento, y genera un resumen.

    Args:
        balanced (str): Estrategia de balanceo a aplicar ("downsample", "oversample", o cualquier otra
                        cadena para 'desbalanceado').
        split_ratios (tuple, optional): Ratios de división para train, val, test.
                                        Si None, usa valores del .env. Debe sumar 1.0.
        use_existing_split (bool): Si True, carga datos ya divididos de data/train, data/val, data/test.
                                   Si False, busca datos raw en data/raw/data_1, data/raw/data_2.

    Returns:
        dict: Un diccionario con los conjuntos de datos divididos ('train', 'val', 'test'),
              donde cada conjunto es un diccionario de la forma {'clase': [lista de imágenes]}.
    """

    print("\n[CARGA] Llamando a la función de carga de datos...")

    # Verificar qué datos están disponibles
    data_train_exists = paths.data_train.exists()
    data_raw_exists = paths.data_raw.exists()

    if use_existing_split and data_train_exists:
        # Usar datos ya divididos (caso común)
        print("[INFO] Usando datos ya divididos (data/train, data/val, data/test)...")
        split_datasets = load_split_data()

        if not split_datasets or all(len(split) == 0 for split in split_datasets.values()):
            raise ValueError("No se cargaron imágenes de los datos divididos. Verifica data/train, data/val, data/test.")

        print("[OK] Carga de datos divididos completada.")

    elif data_raw_exists:
        # Usar datos raw y hacer el split (caso legacy)
        print("[INFO] Usando datos raw (data/raw/data_1, data/raw/data_2) y realizando split...")
        datasets = load_raw_data()

        if not datasets:
            raise ValueError("No se cargó ninguna imagen de data/raw. Verifica las rutas y los tipos de archivo.")

        print("[OK] Carga de datos raw completada.")

        # Realizar split y filtrado de duplicados (De-augmentación)
        # Crear diccionario de environment variables para compatibilidad
        env_vars = {
            'CLASS_NAMES': str(config.data.class_names),
            'SPLIT_RATIOS': str(split_ratios) if split_ratios else str(config.data.split_ratios),
            'IM_SIM_THRESHOLD': str(config.data.similarity_threshold),
            'MAX_ADDED_BALANCE': str(config.data.max_added_balance)
        }
        split_datasets = data_split(datasets, env_vars)

    else:
        # No hay datos disponibles
        raise FileNotFoundError(
            f"No se encontraron datos. Verifica que exista:\n"
            f"  - data/train, data/val, data/test (datos divididos), o\n"
            f"  - data/raw/data_1, data/raw/data_2 (datos raw)\n"
            f"Rutas verificadas:\n"
            f"  - {paths.data_train} (existe: {data_train_exists})\n"
            f"  - {paths.data_raw} (existe: {data_raw_exists})"
        )

    # 2. Balanceo de datos
    # Crear diccionario de environment variables para compatibilidad con funciones legacy
    env_vars = {
        'CLASS_NAMES': str(config.data.class_names),
        'MAX_ADDED_BALANCE': str(config.data.max_added_balance)
    }
    
    if balanced == "downsample":
        datasets_for_model = downsample_dataset(split_datasets, env_vars)
        print(f"\n[BALANCE]  Modo balanceado: Submuestreo de datos exitoso.")

    elif balanced == "oversample":
        datasets_for_model = oversample_dataset(split_datasets, env_vars)
        print("\n[GRAFICO] Modo balanceado: Sobremuestreo de datos exitoso.")

    else: # balanced = 'none' o cualquier otro valor
        datasets_for_model = split_datasets 
        print("\n[GRAFICO] Modo desbalanceado: Usando todas las imágenes disponibles.")
    
    # El diccionario final debe ser {set_type: {category: count}}
    final_counts = {}
    
    # La estructura de datasets_for_model es {set_type: {category: image_list}}
    for set_type, set_data in datasets_for_model.items():
        # Contar el número de imágenes en cada categoría
        final_counts[set_type] = {
            cat: len(image_list)
            for cat, image_list in set_data.items()
        }

    # Asumiendo que todas las clases están en 'train' (o se usa el set más completo)
    all_categories = sorted(list(datasets_for_model['train'].keys()))
    
    # --- Resumen y retorno ---
    print("\n" + "="*60)
    print("[OK] Proceso de división y balanceo completado exitosamente.")
    print("="*60)
    print("[EVAL] Resumen de la Distribución Final:")
    
    # Cabecera dinámica
    header = f"{'Clase':<20} | {'Train':>7} | {'Val':>7} | {'Test':>7} | {'Total':>7}"
    print(header)
    print("-" * len(header))
    
    totals = defaultdict(int)
    
    for class_name in all_categories:
        # Extraer el conteo para cada set, usando 0 si la clase no existe en un set
        count_train = final_counts.get('train', {}).get(class_name, 0)
        count_val = final_counts.get('val', {}).get(class_name, 0)
        count_test = final_counts.get('test', {}).get(class_name, 0)
        
        total_class = count_train + count_val + count_test
        
        # Acumular totales
        totals['train'] += count_train
        totals['val'] += count_val
        totals['test'] += count_test
        
        print(f"{class_name:<20} | {count_train:>7} | {count_val:>7} | {count_test:>7} | {total_class:>7}")
    
    # Imprimir la fila de totales
    print("-" * len(header))
    total_all = sum(totals.values())
    print(f"{'TOTAL':<20} | {totals['train']:>7} | {totals['val']:>7} | {totals['test']:>7} | {total_all:>7}")
    print("="*60)

    return datasets_for_model


###########################################
# ---- Save Splitted-augmented dataset ----
###########################################

def project_dataset(data_aug_split: Dict[str, Dict[str, List[Any]]]) -> None:
    """
    Exporta el dataset dividido y balanceado a carpetas por tipo de set y por clase.

    Args:
        data_aug_split: Diccionario con estructura {'train': {...}, 'val': {...}, 'test': {...}},
                       donde cada subset contiene {'categoria': [lista_imagenes]}.

    Returns:
        None

    Example:
        >>> split_data = {'train': {'Blight': [img1, img2]}, 'val': {...}, 'test': {...}}
        >>> project_dataset(split_data)
    """
    # Usar sistema centralizado de rutas
    SPLIT_DIR = paths.data_processed / "split"
    paths.ensure_dir(SPLIT_DIR)

    total_images_saved = 0

    # Bucle 1: Iterar sobre el tipo de conjunto (train, val, test)
    for set_type, categories_dict in data_aug_split.items():
        if set_type not in ['train', 'val', 'test']:
            continue

        set_images_count = 0

        # Bucle 2: Iterar sobre las categorías (clases de la enfermedad)
        for category, img_list in categories_dict.items():
            dataset_type_dir = SPLIT_DIR / set_type / category  # Ruta completa: .../split/train/Blight
            dataset_type_dir.mkdir(parents=True, exist_ok=True)

            print(f"  Guardando {len(img_list)} imágenes en: {paths.relative_to_root(dataset_type_dir)}")

            # Bucle 3: Guardar cada imagen individualmente
            for i, img in enumerate(img_list):
                file_name = f"{category}_{i:04d}.png"
                file_path = dataset_type_dir / file_name
                try:
                    # Guardar la imagen (asumimos que es un objeto PIL.Image)
                    img.save(file_path)
                    set_images_count += 1
                except Exception as e:
                    print(f"    [ADVERTENCIA] Fallo al guardar {file_name} en {category}: {e}")

        total_images_saved += set_images_count
        print(f"  [INFO] Total imágenes exportadas en '{set_type}': {set_images_count}")

    print(f"\n[OK] Exportación a '{paths.relative_to_root(SPLIT_DIR)}' completada. Total guardado: {total_images_saved} imágenes.")


def split_and_balance_dataset_efficient(balanced: str = "oversample",
                                       split_ratios: tuple = None,
                                       use_existing_split: bool = True) -> Tuple[Dict[str, Any], Dict]:
    """
    Realiza una división estratificada del dataset, aplica la estrategia de balanceo
    (oversample, downsample, o ninguno) solo al conjunto de entrenamiento, y genera un resumen.
    Versión EFICIENTE que devuelve rutas de archivos en lugar de imágenes en memoria.

    Args:
        balanced (str): Estrategia de balanceo a aplicar ("downsample", "oversample", o cualquier otra
                        cadena para 'desbalanceado').
        split_ratios (tuple, optional): Ratios de división para train, val, test.
                                        Si None, usa valores del .env. Debe sumar 1.0.
        use_existing_split (bool): Si True, carga datos ya divididos de data/train, data/val, data/test.
                                   Si False, busca datos raw en data/raw/data_1, data/raw/data_2.

    Returns:
        Tuple[Dict, Dict]: (datasets_dict, label_to_int_mapping)
            - datasets_dict: Diccionario con los conjuntos de datos divididos ('train', 'val', 'test'),
              donde cada conjunto es un diccionario de la forma {'clase': [lista de rutas de archivos]}.
            - label_to_int_mapping: Diccionario que mapea nombres de clases a integers.
    """
    print("\n[CARGA] Llamando a la función de carga eficiente de rutas de archivos...")

    # Verificar qué datos están disponibles
    data_train_exists = paths.data_train.exists()
    data_raw_exists = paths.data_raw.exists()

    if use_existing_split and data_train_exists:
        # Usar datos ya divididos (caso común)
        print("[INFO] Usando rutas de archivos ya divididos (data/train, data/val, data/test)...")
        split_datasets = load_split_data_paths()

        if not split_datasets or all(len(split) == 0 for split in split_datasets.values()):
            raise ValueError("No se cargaron rutas de archivos de los datos divididos. Verifica data/train, data/val, data/test.")

        print("[OK] Carga de rutas de archivos divididos completada.")

    elif data_raw_exists:
        # Usar datos raw y hacer el split (caso legacy)
        print("[INFO] Usando datos raw (data/raw/data_1, data/raw/data_2) y realizando split...")
        raise NotImplementedError("La versión eficiente aún no soporta datos raw. Usa datos ya divididos.")

    else:
        raise ValueError("No se encontraron datos ni en data/train ni en data/raw. Verifica las rutas.")

    # Aplicar balanceo SOLO al conjunto de entrenamiento
    if balanced.lower() == "oversample":
        print("\n[BALANCE]  Iniciando Oversampling eficiente en el conjunto 'train'.")
        split_datasets['train'] = apply_oversampling_to_paths(split_datasets['train'])
        print("[OK] Oversampling eficiente de 'train' completado.")

    elif balanced.lower() == "downsample":
        print("\n[BALANCE]  Iniciando Downsampling eficiente en el conjunto 'train'.")
        split_datasets['train'] = apply_downsampling_to_paths(split_datasets['train'])
        print("[OK] Downsampling eficiente de 'train' completado.")

    else:
        print(f"\n[BALANCE]  Usando datos desbalanceados para 'train' (balanced='{balanced}').")

    # Crear mapeo de labels a integers
    all_classes = set()
    for split_name, split_data in split_datasets.items():
        all_classes.update(split_data.keys())

    label_to_int = {label: i for i, label in enumerate(sorted(all_classes))}

    # Resumen final
    print("\n============================================================\n[EVAL] Resumen de la Distribución Final:")
    print("Clase                |   Train |     Val |    Test |   Total")
    print("------------------------------------------------------------")

    train_counts = {cls: len(paths) for cls, paths in split_datasets['train'].items()}
    val_counts = {cls: len(paths) for cls, paths in split_datasets['val'].items()}
    test_counts = {cls: len(paths) for cls, paths in split_datasets['test'].items()}

    all_classes_sorted = sorted(all_classes)
    totals = {'train': 0, 'val': 0, 'test': 0, 'total': 0}

    for cls in all_classes_sorted:
        train_cnt = train_counts.get(cls, 0)
        val_cnt = val_counts.get(cls, 0)
        test_cnt = test_counts.get(cls, 0)
        total_cnt = train_cnt + val_cnt + test_cnt

        print(f"{cls:<20} | {train_cnt:>6} | {val_cnt:>6} | {test_cnt:>6} | {total_cnt:>6}")

        totals['train'] += train_cnt
        totals['val'] += val_cnt
        totals['test'] += test_cnt
        totals['total'] += total_cnt

    print("------------------------------------------------------------")
    print(f"{'TOTAL':<20} | {totals['train']:>6} | {totals['val']:>6} | {totals['test']:>6} | {totals['total']:>>6}")
    print("============================================================")

    print("\n[OK] Proceso de división y balanceo eficiente completado exitosamente.")

    return split_datasets, label_to_int


def apply_oversampling_to_paths(train_data: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Aplica oversampling eficiente duplicando rutas de archivos en lugar de imágenes en memoria.

    Args:
        train_data: Diccionario clase -> lista de rutas de archivos

    Returns:
        Diccionario con oversampling aplicado
    """
    # Calcular el tamaño objetivo (máximo de todas las clases)
    max_samples = max(len(paths) for paths in train_data.values())
    print(f"[CALCULO] Tamaño objetivo por clase: {max_samples} rutas.")

    balanced_data = {}

    for class_name, file_paths in train_data.items():
        current_count = len(file_paths)
        balanced_data[class_name] = list(file_paths)  # Copiar rutas originales

        if current_count < max_samples:
            # Duplicar rutas hasta alcanzar el objetivo
            while len(balanced_data[class_name]) < max_samples:
                remaining = max_samples - len(balanced_data[class_name])
                extend_count = min(remaining, current_count)
                balanced_data[class_name].extend(file_paths[:extend_count])

            print(f"  [AUMENTO] Categoría '{class_name}' aumentada: {current_count} -> {len(balanced_data[class_name])} rutas.")

    return balanced_data


def apply_downsampling_to_paths(train_data: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Aplica downsampling eficiente reduciendo rutas de archivos en lugar de imágenes en memoria.

    Args:
        train_data: Diccionario clase -> lista de rutas de archivos

    Returns:
        Diccionario con downsampling aplicado
    """
    # Calcular el tamaño objetivo (mínimo de todas las clases)
    min_samples = min(len(paths) for paths in train_data.values())
    print(f"[CALCULO] Tamaño objetivo por clase: {min_samples} rutas.")

    balanced_data = {}

    for class_name, file_paths in train_data.items():
        current_count = len(file_paths)

        if current_count > min_samples:
            # Reducir aleatoriamente a min_samples
            import random
            selected_paths = random.sample(file_paths, min_samples)
            balanced_data[class_name] = selected_paths
            print(f"  [REDUCCIÓN] Categoría '{class_name}' reducida: {current_count} -> {len(balanced_data[class_name])} rutas.")
        else:
            balanced_data[class_name] = list(file_paths)

    return balanced_data
