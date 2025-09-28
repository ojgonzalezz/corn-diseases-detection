#####################################################################################
# -------------------------- Data Preprocessing Utilities ---------------------------
#####################################################################################


#########################
# ---- Depdendencies ----
#########################

import pathlib
import shutil
import ast
import random
from tqdm import tqdm
import collections
from collections import defaultdict
from typing import Dict, List, Any
import numpy as np
from PIL import Image
from src.adapters.data_loader import load_raw_data
from src.utils.image_modifier import ImageAugmentor
from src.utils.data_augmentator import DataAugmenter
from src.core.load_env import EnvLoader

#####################################
# ---- Data DownsamplingFunction ----
#####################################


def downsample_dataset(datasets: Dict[str, Any], split_ratios: tuple) -> Dict[str, Any]:
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
    env_vars = EnvLoader().get_all()
    # Usar ast.literal_eval para obtener la lista de clases de forma segura
    try:
        categories = set(ast.literal_eval(env_vars.get("CLASS_NAMES", "[]")))
    except Exception:
        # Si falla la carga, derivar categorías de los datos
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
    
    print(f"📐 Tamaño objetivo total por clase (Total): {minimum_lenght} imágenes.")
    
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
        
        print(f"  ⚖️  Categoría '{category}' submuestreada a {minimum_lenght} imágenes.")


    # 5. Generar un resumen de la nueva distribución (opcional, pero útil)
    print("\n✅ Resumen de la Nueva Distribución Downsampled:")
    for dataset_key, data in downsampled_datasets.items():
        total = sum(len(imgs) for imgs in data["images"].values())
        print(f"  {dataset_key} ({data['dataset_consideration']}): Total {total} imágenes")
        for cat, imgs in data["images"].items():
            print(f"    - {cat}: {len(imgs)} imágenes")
            
    return downsampled_datasets


#####################################
# ---- Data DownsamplingFunction ----
#####################################

def downsample_dataset(datasets, split_datios:tuple):
    categories = set(EnvLoader().get_all().get("CLASS_NAMES", "[]"))
    raw_keys_to_iterate = [key for key in datasets.keys() if key.startswith("data_")]

    datasets_annotations = {
        dataset_key: {
            "consideration": datasets[dataset_key].get("dataset_consideration","undefined"),
            "lenghts_per_category":{cat: len(datasets[dataset_key]["images"].get(cat, 0)) for cat in categories}
            } 
        for dataset_key in raw_keys_to_iterate
        }
    
    total_lenghts = {
        cat : sum(
            datasets_annotations[dataset_key]["lenghts_per_category"][cat] for  dataset_key in datasets_annotations.keys()
        )
        for cat in categories
    }

    minimum_lenght = min(list(total_lenghts.values()))


    for dataset_key in raw_keys_to_iterate:
        consideration = datasets[dataset_key]["dataset_consideration"]
        unavailable_classes = differences[dataset_key]

        if len(unavailable_classes) == 0:
            pass
        else:
            e

        min_class_size = min(len(images) for images in dataset_key.values())
        print(f"\n⚖️  Modo balanceado: Todas las clases se reducirán a {min_class_size} imágenes.")
        
        for class_name, images in dataset.items():
            random.shuffle(images)
            sampled_images = random.sample(images, min_class_size)
            
            n_train = int(min_class_size * split_ratios[0])
            n_val = int(min_class_size * split_ratios[1])
            
            train_set[class_name] = sampled_images[:n_train]
            val_set[class_name] = sampled_images[n_train : n_train + n_val]
            test_set[class_name] = sampled_images[n_train + n_val:]
            
            final_counts[class_name] = {'train': len(train_set[class_name]), 'val': len(val_set[class_name]), 'test': len(test_set[class_name])}



def split_and_balance_dataset(split_ratios: tuple = (0.7, 0.15, 0.15), balanced: str = "downsample"):
    """
    Realiza una división estratificada de un dataset de imágenes.

    Args:
        base_path (pathlib.Path): La ruta del directorio que contiene las carpetas de las clases.
        split_ratios (tuple): Una tupla con los ratios de división para train, val, y test.
        balanced (bool): Si es True, balancea el dataset usando submuestreo. Si es False, usa todas las imágenes.

    Returns:
        dict: Un diccionario con los conjuntos de datos divididos ('train', 'val', 'test'), 
              donde cada conjunto es un diccionario de la forma {'clase': [lista de imágenes]}.
    """


    print("\n📦 Llamando a la función de carga de datos...")
    dataset = load_raw_data()
    env_load = EnvLoader().get_all()
    print("✅ Carga de datos completada.")

    if not dataset:
        raise ValueError("No se cargó ninguna imagen. Verifica las rutas y los tipos de archivo.")

    # Lógica de división
    train_set, val_set, test_set = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
    final_counts = collections.defaultdict(dict)

    if balanced == "downsample":
        min_class_size = min(len(images) for images in dataset.values())
        print(f"\n⚖️  Modo balanceado: Todas las clases se reducirán a {min_class_size} imágenes.")
        
        for class_name, images in dataset.items():
            random.shuffle(images)
            sampled_images = random.sample(images, min_class_size)
            
            n_train = int(min_class_size * split_ratios[0])
            n_val = int(min_class_size * split_ratios[1])
            
            train_set[class_name] = sampled_images[:n_train]
            val_set[class_name] = sampled_images[n_train : n_train + n_val]
            test_set[class_name] = sampled_images[n_train + n_val:]
            
            final_counts[class_name] = {'train': len(train_set[class_name]), 'val': len(val_set[class_name]), 'test': len(test_set[class_name])}

    elif balanced == "oversample":
        print("\n📈 Modo balanceado: Aplicando aumento de datos (oversampling)...")
        augmenter = DataAugmenter()
        image_modifier = ImageAugmentor()
        
        # 1. Dividir el dataset original
        for class_name, images in dataset.items():
            random.shuffle(images)
            total_images = len(images)
            n_train = int(total_images * split_ratios[0])
            n_val = int(total_images * split_ratios[1])
            
            train_set[class_name] = images[:n_train]
            val_set[class_name] = images[n_train : n_train + n_val]
            test_set[class_name] = images[n_train + n_val:]

        # 2. Encontrar el número de imágenes en la clase mayoritaria
        max_train_size = max(len(images) for images in train_set.values())
        MAX_ADDED_BALANCE = int(env_load.get("MAX_ADDED_BALANCE", 50))
        target_size = max_train_size + MAX_ADDED_BALANCE
        
        print(f"🎯 Tamaño objetivo para las clases minoritarias: {target_size} imágenes por clase.")

        # 3. Aplicar oversampling a las clases minoritarias del conjunto de entrenamiento
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

        # Iterar sobre las clases del conjunto de entrenamiento para aplicar oversampling
        for class_name, images in train_set.items():
            if len(images) < target_size:
                needed = target_size - len(images)
                
                # Iterar hasta alcanzar el tamaño objetivo
                for _ in tqdm(range(needed), desc=f"Aumentando '{class_name}'", unit="img"):
                    # Seleccionar una imagen aleatoria para la aumentación
                    img = random.choice(images)
                    
                    # ⚠️ Paso 1: Aplicar dos transformaciones de calidad consecutivamente
                    # Elegir 2 transformaciones de calidad diferentes
                    chosen_quality_transforms = random.sample(quality_transforms, 2)
                    
                    transformed_img = img
                    for transform_func in chosen_quality_transforms:
                        img_np = np.array(transformed_img)
                        try:
                            # Algunas funciones requieren parámetros específicos
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
                             # Fallback en caso de que alguna función necesite argumentos
                             transformed_img_np = transform_func(img_np)
                        transformed_img = Image.fromarray(transformed_img_np)

                    # ⚠️ Paso 2: Aplicar una transformación espacial aleatoria
                    chosen_spatial_transform = random.choice(spatial_transforms)
                    final_augmented_img = chosen_spatial_transform(transformed_img)

                    # Añadir la imagen aumentada al conjunto de entrenamiento
                    train_set[class_name].append(final_augmented_img)
            
            # Limitar el tamaño de la clase mayoritaria
            if len(train_set[class_name]) > target_size:
                train_set[class_name] = random.sample(train_set[class_name], target_size)

        # 4. Recolectar el conteo final para el resumen
        for class_name in dataset.keys():
            final_counts[class_name] = {
                'train': len(train_set[class_name]),
                'val': len(val_set[class_name]),
                'test': len(test_set[class_name])
            }

    else: # balanced=False, modo desbalanceado
        print("\n📈 Modo desbalanceado: Usando todas las imágenes disponibles.")

        for class_name, images in dataset.items():
            random.shuffle(images)
            total_images = len(images)
            
            n_train = int(total_images * split_ratios[0])
            n_val = int(total_images * split_ratios[1])
            
            train_set[class_name] = images[:n_train]
            val_set[class_name] = images[n_train : n_train + n_val]
            test_set[class_name] = images[n_train + n_val:]

            final_counts[class_name] = {'train': len(train_set[class_name]), 'val': len(val_set[class_name]), 'test': len(test_set[class_name])}

    # Resumen y retorno
    print("\n" + "="*60)
    print("✅ Proceso de división completado exitosamente.")
    print("="*60)
    print("📊 Resumen de la Distribución Final:")
    header = f"{'Clase':<20} | {'Train':>7} | {'Val':>7} | {'Test':>7} | {'Total':>7}"
    print(header)
    print("-" * len(header))
    
    totals = collections.defaultdict(int)
    for class_name, counts in sorted(final_counts.items()):
        total_class = sum(counts.values())
        totals['train'] += counts['train']
        totals['val'] += counts['val']
        totals['test'] += counts['test']
        print(f"{class_name:<20} | {counts['train']:>7} | {counts['val']:>7} | {counts['test']:>7} | {total_class:>7}")
    
    print("-" * len(header))
    total_all = sum(totals.values())
    print(f"{'TOTAL':<20} | {totals['train']:>7} | {totals['val']:>7} | {totals['test']:>7} | {total_all:>7}")
    print("="*60)

    return {'train': train_set, 'val': val_set, 'test': test_set}



####################################
# ---- Split and balance datast ----
####################################


def split_and_balance_dataset(split_ratios: tuple = (0.7, 0.15, 0.15), balanced: str = "downsample"):
    """
    Realiza una división estratificada de un dataset de imágenes.

    Args:
        base_path (pathlib.Path): La ruta del directorio que contiene las carpetas de las clases.
        split_ratios (tuple): Una tupla con los ratios de división para train, val, y test.
        balanced (bool): Si es True, balancea el dataset usando submuestreo. Si es False, usa todas las imágenes.

    Returns:
        dict: Un diccionario con los conjuntos de datos divididos ('train', 'val', 'test'), 
              donde cada conjunto es un diccionario de la forma {'clase': [lista de imágenes]}.
    """


    print("\n📦 Llamando a la función de carga de datos...")
    dataset = load_raw_data()
    env_load = EnvLoader().get_all()
    print("✅ Carga de datos completada.")

    if not dataset:
        raise ValueError("No se cargó ninguna imagen. Verifica las rutas y los tipos de archivo.")

    # Lógica de división
    train_set, val_set, test_set = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
    final_counts = collections.defaultdict(dict)

    if balanced == "downsample":
        min_class_size = min(len(images) for images in dataset.values())
        print(f"\n⚖️  Modo balanceado: Todas las clases se reducirán a {min_class_size} imágenes.")
        
        for class_name, images in dataset.items():
            random.shuffle(images)
            sampled_images = random.sample(images, min_class_size)
            
            n_train = int(min_class_size * split_ratios[0])
            n_val = int(min_class_size * split_ratios[1])
            
            train_set[class_name] = sampled_images[:n_train]
            val_set[class_name] = sampled_images[n_train : n_train + n_val]
            test_set[class_name] = sampled_images[n_train + n_val:]
            
            final_counts[class_name] = {'train': len(train_set[class_name]), 'val': len(val_set[class_name]), 'test': len(test_set[class_name])}

    elif balanced == "oversample":
        print("\n📈 Modo balanceado: Aplicando aumento de datos (oversampling)...")
        augmenter = DataAugmenter()
        image_modifier = ImageAugmentor()
        
        # 1. Dividir el dataset original
        for class_name, images in dataset.items():
            random.shuffle(images)
            total_images = len(images)
            n_train = int(total_images * split_ratios[0])
            n_val = int(total_images * split_ratios[1])
            
            train_set[class_name] = images[:n_train]
            val_set[class_name] = images[n_train : n_train + n_val]
            test_set[class_name] = images[n_train + n_val:]

        # 2. Encontrar el número de imágenes en la clase mayoritaria
        max_train_size = max(len(images) for images in train_set.values())
        MAX_ADDED_BALANCE = int(env_load.get("MAX_ADDED_BALANCE", 50))
        target_size = max_train_size + MAX_ADDED_BALANCE
        
        print(f"🎯 Tamaño objetivo para las clases minoritarias: {target_size} imágenes por clase.")

        # 3. Aplicar oversampling a las clases minoritarias del conjunto de entrenamiento
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

        # Iterar sobre las clases del conjunto de entrenamiento para aplicar oversampling
        for class_name, images in train_set.items():
            if len(images) < target_size:
                needed = target_size - len(images)
                
                # Iterar hasta alcanzar el tamaño objetivo
                for _ in tqdm(range(needed), desc=f"Aumentando '{class_name}'", unit="img"):
                    # Seleccionar una imagen aleatoria para la aumentación
                    img = random.choice(images)
                    
                    # ⚠️ Paso 1: Aplicar dos transformaciones de calidad consecutivamente
                    # Elegir 2 transformaciones de calidad diferentes
                    chosen_quality_transforms = random.sample(quality_transforms, 2)
                    
                    transformed_img = img
                    for transform_func in chosen_quality_transforms:
                        img_np = np.array(transformed_img)
                        try:
                            # Algunas funciones requieren parámetros específicos
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
                             # Fallback en caso de que alguna función necesite argumentos
                             transformed_img_np = transform_func(img_np)
                        transformed_img = Image.fromarray(transformed_img_np)

                    # ⚠️ Paso 2: Aplicar una transformación espacial aleatoria
                    chosen_spatial_transform = random.choice(spatial_transforms)
                    final_augmented_img = chosen_spatial_transform(transformed_img)

                    # Añadir la imagen aumentada al conjunto de entrenamiento
                    train_set[class_name].append(final_augmented_img)
            
            # Limitar el tamaño de la clase mayoritaria
            if len(train_set[class_name]) > target_size:
                train_set[class_name] = random.sample(train_set[class_name], target_size)

        # 4. Recolectar el conteo final para el resumen
        for class_name in dataset.keys():
            final_counts[class_name] = {
                'train': len(train_set[class_name]),
                'val': len(val_set[class_name]),
                'test': len(test_set[class_name])
            }

    else: # balanced=False, modo desbalanceado
        print("\n📈 Modo desbalanceado: Usando todas las imágenes disponibles.")

        for class_name, images in dataset.items():
            random.shuffle(images)
            total_images = len(images)
            
            n_train = int(total_images * split_ratios[0])
            n_val = int(total_images * split_ratios[1])
            
            train_set[class_name] = images[:n_train]
            val_set[class_name] = images[n_train : n_train + n_val]
            test_set[class_name] = images[n_train + n_val:]

            final_counts[class_name] = {'train': len(train_set[class_name]), 'val': len(val_set[class_name]), 'test': len(test_set[class_name])}

    # Resumen y retorno
    print("\n" + "="*60)
    print("✅ Proceso de división completado exitosamente.")
    print("="*60)
    print("📊 Resumen de la Distribución Final:")
    header = f"{'Clase':<20} | {'Train':>7} | {'Val':>7} | {'Test':>7} | {'Total':>7}"
    print(header)
    print("-" * len(header))
    
    totals = collections.defaultdict(int)
    for class_name, counts in sorted(final_counts.items()):
        total_class = sum(counts.values())
        totals['train'] += counts['train']
        totals['val'] += counts['val']
        totals['test'] += counts['test']
        print(f"{class_name:<20} | {counts['train']:>7} | {counts['val']:>7} | {counts['test']:>7} | {total_class:>7}")
    
    print("-" * len(header))
    total_all = sum(totals.values())
    print(f"{'TOTAL':<20} | {totals['train']:>7} | {totals['val']:>7} | {totals['test']:>7} | {total_all:>7}")
    print("="*60)

    return {'train': train_set, 'val': val_set, 'test': test_set}