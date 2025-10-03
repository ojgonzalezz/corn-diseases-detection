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
import tensorflow as tf
import random
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import math

####################################
# ---- System-related utilities ----
####################################

def check_cuda_availability(check_pytorch: bool = False):
    """
    Verifica si TensorFlow (y opcionalmente PyTorch) pueden usar una GPU con soporte CUDA.

    Args:
        check_pytorch (bool): Si True, también verifica PyTorch (requiere instalación).
                             Por defecto False ya que PyTorch no es una dependencia del proyecto.

    Note:
        Este proyecto usa principalmente TensorFlow. PyTorch es opcional y no está
        incluido en requirements.txt para reducir el tamaño de instalación (~2GB).
    """
    print("[GPU] Verificando disponibilidad de GPU...\n")

    # Verificar TensorFlow (siempre)
    print("[INFO] Verificando GPU en TensorFlow")
    print(f"  TensorFlow version: {tf.__version__}")
    print(f"  Built with CUDA: {tf.test.is_built_with_cuda()}")

    # Obtener lista de GPUs físicas
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        print(f"[OK] ✓ GPU disponible para TensorFlow")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
            # Mostrar detalles de memoria si está disponible
            try:
                gpu_details = tf.config.experimental.get_memory_info(gpu.name)
                current_mb = gpu_details['current'] / (1024**2)
                peak_mb = gpu_details['peak'] / (1024**2)
                print(f"    Memoria actual: {current_mb:.2f} MB")
                print(f"    Memoria pico: {peak_mb:.2f} MB")
            except:
                pass
    else:
        print("[ADVERTENCIA] ✗ GPU no disponible. TensorFlow se ejecutará en CPU")
        print("  Para usar GPU, asegúrate de tener:")
        print("  1. CUDA Toolkit instalado")
        print("  2. TensorFlow-GPU instalado (pip install tensorflow[and-cuda])")

    # Verificar PyTorch (opcional)
    if check_pytorch:
        print("\n[INFO] Verificando GPU en PyTorch")
        try:
            import torch
            is_available = torch.cuda.is_available()

            if is_available:
                print(f"[OK] ✓ GPU disponible para PyTorch")
                print(f"  GPU: {torch.cuda.get_device_name(0)}")
                print(f"  CUDA version: {torch.version.cuda}")
            else:
                print("[ADVERTENCIA] ✗ GPU no disponible para PyTorch")
        except ImportError:
            print("[INFO] PyTorch no instalado (opcional)")
            print("  Para instalarlo: pip install torch")

    print("\n" + "="*60)
    print(f"[CONFIG] Built with cuDNN: {tf.test.is_built_with_gpu_support()} [INFO]\n")

    if tf.test.is_built_with_cuda() and tf.test.is_built_with_gpu_support():
        print("[OBJETIVO] ¡TensorFlow puede usar la GPU con CUDA y cuDNN! [INICIO][ACTIVO]")
    else:
        print("[ADVERTENCIA] TensorFlow no puede usar la GPU. Se ejecutará en CPU [CPU]")


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

    print("[EVAL] Iniciando división estratificada...")
    
    # 3. Iterar por categoría para realizar la división
    for category, image_list in dataset.items():
        
        # Mezclar la lista de imágenes de la categoría para aleatoriedad
        random.shuffle(image_list)
        total_count = len(image_list)
        
        if total_count == 0:
            print(f"[ADVERTENCIA] Advertencia: La categoría '{category}' está vacía. Se saltará.")
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


    print("[OK] División estratificada completada.")
    return split_sets

##################################
# ---- Data-related Utilities ----
##################################

def flatten_data(data_dict: Dict[str, List[Any]], image_size: Tuple[int, int] = (224, 224)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flatten a dictionary of images by class into numpy arrays.

    Args:
        data_dict: Dictionary mapping class names to lists of PIL Images
        image_size: Target size for resizing images (width, height)

    Returns:
        Tuple of (images array, labels array)
    """
    images = []
    labels = []
    for class_name, image_list in data_dict.items():
        for img in image_list:
            resized_img = img.resize(image_size)
            images.append(np.array(resized_img))
            labels.append(class_name)

    return np.array(images), np.array(labels)


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
    print("\n[VISUAL] Mostrando ejemplos de imágenes aumentadas del primer lote de entrenamiento:")
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


def create_efficient_dataset_from_dict(data_dict: Dict[str, List[Any]],
                                       image_size: Tuple[int, int] = (224, 224),
                                       batch_size: int = 32,
                                       num_classes: int = None,
                                       shuffle: bool = True,
                                       augment: bool = False) -> Tuple[tf.data.Dataset, Dict]:
    """
    Crea un tf.data.Dataset eficiente desde un diccionario de imágenes,
    evitando cargar todo en memoria RAM.

    Args:
        data_dict: Diccionario con clase -> lista de rutas de archivos PIL Images
        image_size: Tamaño objetivo de las imágenes
        batch_size: Tamaño del batch
        num_classes: Número de clases para one-hot encoding. Si None, usa len(unique_labels)
        shuffle: Si True, baraja los datos
        augment: Si True, aplica aumentación de datos básica

    Returns:
        Tuple de (tf.data.Dataset listo para entrenamiento, label_to_int mapping)
    """
    # Extraer rutas de archivos y labels
    file_paths = []
    labels = []

    for class_name, image_list in data_dict.items():
        for img in image_list:
            # Si es una ruta de archivo (string), usarla directamente
            if isinstance(img, str):
                file_paths.append(img)
                labels.append(class_name)
            # Si es un objeto PIL Image, necesitamos guardarlo temporalmente o procesarlo
            else:
                # Para compatibilidad, procesar PIL Images en memoria pero eficientemente
                import tempfile
                import os
                from PIL import Image

                # Crear archivo temporal
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                    img.save(tmp_file.name)
                    file_paths.append(tmp_file.name)
                    labels.append(class_name)

    # Crear mapeo de labels a integers
    unique_labels = sorted(list(set(labels)))
    label_to_int = {label: i for i, label in enumerate(unique_labels)}

    # Determinar número de clases
    if num_classes is None:
        num_classes = len(unique_labels)

    # Convertir labels a integers
    labels_int = [label_to_int[label] for label in labels]

    # Crear tf.data.Dataset desde rutas de archivos
    def load_and_preprocess_image(file_path, label):
        # Cargar imagen
        image = tf.io.read_file(file_path)
        image = tf.image.decode_jpeg(image, channels=3)

        # Redimensionar
        image = tf.image.resize(image, image_size)

        # Normalizar a [0,1]
        image = tf.cast(image, tf.float32) / 255.0

        # Aplicar aumentación básica si está habilitada
        if augment:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_brightness(image, max_delta=0.1)
            image = tf.image.random_contrast(image, lower=0.9, upper=1.1)

        # Convertir label a one-hot encoding
        label_onehot = tf.one_hot(label, depth=num_classes)

        return image, label_onehot

    # Crear dataset
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels_int))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(file_paths), seed=42)

    # Mapear función de carga
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch y prefetch para optimización
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset, label_to_int


def create_efficient_dataset_from_paths(data_dir: str,
                                       image_size: Tuple[int, int] = (224, 224),
                                       batch_size: int = 32,
                                       shuffle: bool = True,
                                       augment: bool = False,
                                       validation_split: float = 0.0) -> Tuple[tf.data.Dataset, Dict]:
    """
    Crea un tf.data.Dataset eficiente directamente desde un directorio de imágenes.

    Args:
        data_dir: Directorio raíz con subdirectorios por clase
        image_size: Tamaño objetivo de las imágenes
        batch_size: Tamaño del batch
        shuffle: Si True, baraja los datos
        augment: Si True, aplica aumentación de datos básica
        validation_split: Fracción de datos para validación (0.0 para no dividir)

    Returns:
        Tuple de (dataset, class_names_dict)
    """
    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='int',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=shuffle,
        seed=42,
        validation_split=validation_split,
        subset='training' if validation_split > 0 else None,
    )

    # Normalizar imágenes
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    dataset = dataset.map(lambda x, y: (normalization_layer(x), y))

    # Aplicar aumentación si está habilitada
    if augment:
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomBrightness(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ])
        dataset = dataset.map(lambda x, y: (data_augmentation(x), y))

    # Prefetch para optimización
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    # Obtener nombres de clases
    class_names = dataset.class_names
    class_to_int = {name: i for i, name in enumerate(class_names)}

    return dataset, class_to_int
