#####################################################################################
# -------------------------- Data Preprocessing Utilities ---------------------------
#####################################################################################


#########################
# ---- Depdendencies ----
#########################

import pathlib
import shutil
import random
from tqdm import tqdm
import collections
import numpy as np
from PIL import Image
from src.adapters.data_loader import load_raw_data
from src.utils.image_modifier import ImageAugmentor
from src.utils.data_augmentator import DataAugmenter
from src.core.load_env import EnvLoader


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