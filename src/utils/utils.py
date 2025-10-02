#####################################################################################
# -------------------------------- Project Utilities --------------------------------
#####################################################################################


#########################
# ---- Depdendencies ----
#########################

# system
import os
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from PIL import Image
import torch
import tensorflow as tf
import random
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import math

####################################
# ---- System-related utilities ----
####################################

def check_cuda_availability():
    """
    Verifica si PyTorch y TensorFlow pueden usar una GPU con soporte CUDA.
    """
    print("⚡⚡ Verificando si la GPU está disponible... ⚡⚡\n")

    print("🔹 Verificando GPU en PyTorch 🔹")
    
    is_available = torch.cuda.is_available()
    
    if is_available:
        print("✅ ¡La GPU está disponible! PyTorch puede usar CUDA 🎉🚀")
        # Opcional: muestra el nombre de la GPU que se está utilizando
        print(f"   💻 Nombre de la GPU: {torch.cuda.get_device_name(0)}\n")
    else:
        print("❌ La GPU no está disponible ❌. PyTorch se ejecutará en CPU 🖥️")
        print("   ⚠️ Asegúrate de haber instalado la versión correcta de PyTorch con soporte CUDA.\n")

    print("🔹 Verificando GPU en TensorFlow 🔹")
     
    print(f"📌 TensorFlow version: {tf.__version__}")
    print(f"⚙️ Built with CUDA: {tf.test.is_built_with_cuda()} 🌟")
    print(f"⚙️ Built with cuDNN: {tf.test.is_built_with_gpu_support()} 🌟\n")

    if tf.test.is_built_with_cuda() and tf.test.is_built_with_gpu_support():
        print("🎯 ¡TensorFlow puede usar la GPU con CUDA y cuDNN! 🚀🔥")
    else:
        print("⚠️ TensorFlow no puede usar la GPU. Se ejecutará en CPU 🖥️")


#################################
# ---- Stratified data split ----
#################################

def stratified_split_dataset(
    dataset: Dict[str, List[Any]], 
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)
) -> Dict[str, Dict[str, List[Any]]]:
    """
    Realiza una división estratificada de un dataset unificado por clase.

    Asegura que la proporción de cada clase en los conjuntos de train, val y test 
    sea la misma que en el dataset original.

    Args:
        dataset (Dict[str, List[Any]]): Diccionario unificado de imágenes, 
                                        donde las claves son los nombres de las categorías.
        split_ratios (Tuple[float, float, float]): Ratios de división para 
                                                   (train, val, test). Deben sumar 1.0.

    Returns:
        Dict[str, Dict[str, List[Any]]]: Diccionario con los conjuntos divididos:
                                        {'train': {'cat_1': [...], ...}, 'val': {...}, 'test': {...}}
    """
    
    # 1. Validación de ratios
    if not math.isclose(sum(split_ratios), 1.0):
        raise ValueError(f"Los ratios de división deben sumar 1.0. Suma actual: {sum(split_ratios):.2f}")

    train_ratio, val_ratio, test_ratio = split_ratios
    
    # 2. Inicializar los diccionarios de salida
    split_sets = {
        'train': defaultdict(list),
        'val': defaultdict(list),
        'test': defaultdict(list)
    }

    print("📊 Iniciando división estratificada...")
    
    # 3. Iterar por categoría para realizar la división
    for category, image_list in dataset.items():
        
        # Mezclar la lista de imágenes de la categoría para aleatoriedad
        random.shuffle(image_list)
        total_count = len(image_list)
        
        if total_count == 0:
            print(f"⚠️ Advertencia: La categoría '{category}' está vacía. Se saltará.")
            continue

        # Calcular los tamaños de los subconjuntos
        # Usamos math.floor para el entrenamiento y la validación para asegurar que el resto 
        # se asigne al conjunto de prueba.
        n_train = math.floor(total_count * train_ratio)
        n_val = math.floor(total_count * val_ratio)
        # El resto de las imágenes se asigna a test
        n_test = total_count - n_train - n_val
        
        # Ajustar para asegurar que n_test sea al menos 0 (solo por robustez)
        if n_test < 0:
            n_test = 0
            n_val = total_count - n_train # Reajustar val si los floats causaron exceso
            
        print(f"  [{category:<20}] Total: {total_count} -> Train: {n_train}, Val: {n_val}, Test: {n_test}")

        # 4. Asignar las imágenes a los conjuntos
        
        # Asignar a Train
        split_sets['train'][category] = image_list[:n_train]
        
        # Asignar a Val (comienza donde Train termina)
        start_val = n_train
        end_val = n_train + n_val
        split_sets['val'][category] = image_list[start_val:end_val]
        
        # Asignar a Test (comienza donde Val termina)
        start_test = end_val
        split_sets['test'][category] = image_list[start_test:]


    print("✅ División estratificada completada.")
    return split_sets

##################################
# ---- Data-related Utilities ----
##################################

def load_images_from_folder(folder_path):
    """
    Carga todas las imágenes de una carpeta en una lista.
    
    Args:
        folder_path (str): La ruta del directorio que contiene las imágenes.

    Returns:
        list: Una lista de objetos de imagen de Pillow.
    """
    # Usar pathlib para manejar la ruta de manera segura
    p = pathlib.Path(folder_path)
    if not p.is_dir():
        print(f"Error: La ruta '{folder_path}' no es un directorio válido.")
        return []

    images = []
    # Usar glob para encontrar todos los archivos con las extensiones de imagen
    for image_path in p.glob('*.[jp][pn]g'):
        try:
            # Abrir y cargar la imagen
            with Image.open(image_path) as img:
                # La función .convert('RGB') asegura que todas las imágenes
                # tengan 3 canales de color, lo cual es útil para el entrenamiento
                # de modelos de deep learning.
                images.append(img.convert('RGB'))
                
        except (IOError, OSError) as e:
            print(f"Error al cargar la imagen {image_path}: {e}")
            continue

    print(f"Se cargaron {len(images)} imágenes desde '{folder_path}'.")
    return images


##################################
# ---- Data-related Utilities ----
##################################

def pil_to_numpy(pil_img: Image.Image) -> np.ndarray:
    """Convierte una imagen PIL.Image en un array de NumPy."""
    return np.array(pil_img)

############################
# ---- Image Visualizer ----
############################

def plot_images(images, titles=None, cols=2, figsize=(10, 5)):
    """
    Función para graficar imágenes salientes de un método del ImageAugmentator.

    Parámetros
    ----------
    images : list o np.ndarray o PIL.Image
        Lista de imágenes, una sola imagen o PIL.Image.
    titles : list, opcional
        Lista de títulos para cada imagen.
    cols : int, opcional
        Número de columnas en la grilla de plots.
    figsize : tuple, opcional
        Tamaño de la figura (ancho, alto).
    """
    # Si se pasa una sola imagen, convertirla en lista
    if not isinstance(images, (list, tuple)):
        images = [images]

    # Normalizar imágenes a numpy arrays
    processed_images = []
    for img in images:
        if isinstance(img, Image.Image):  # Si es PIL.Image → convertir
            img = pil_to_numpy(img)
        elif not isinstance(img, np.ndarray):
            raise TypeError(f"Formato de imagen no soportado: {type(img)}")
        processed_images.append(img)

    n = len(processed_images)
    rows = (n + cols - 1) // cols  # calcular filas necesarias

    plt.figure(figsize=figsize)

    for i, img in enumerate(processed_images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap="gray" if len(img.shape) == 2 else None)
        if titles and i < len(titles):
            plt.title(titles[i])
        plt.axis("off")

    plt.tight_layout()
    plt.show()

#####################################
# ---- Data Augmentation Plotter ----
#####################################

def plot_augmented_images(generator):
    """Función de utilidad para visualizar el efecto del Data Augmentation."""
    print("\n🎨 Mostrando ejemplos de imágenes aumentadas del primer lote de entrenamiento:")
    images, labels = next(generator)
    
    plt.figure(figsize=(12, 12))
    for i in range(min(9, len(images))): # Asegura no exceder el tamaño del lote
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        class_index = tf.argmax(labels[i]).numpy()
        class_name = list(generator.class_indices.keys())[class_index]
        plt.title(class_name)
        plt.axis("off")
    plt.suptitle("Visualización del Aumento de Datos en Tiempo Real", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
