#####################################################################################
# ------------------------------- Project Data Loader -------------------------------
#####################################################################################

#########################
# ---- Depdendencies ----
#########################

# system
import os
import sys
import pathlib
from typing import Dict, List, Any
from dotenv import load_dotenv
from ast import literal_eval
from PIL import Image

# Environment variables loading
sys.path.append(os.path.abspath(os.path.join("..", "src")))
from src.core.load_env import EnvLoader
from src.utils.utils import *
from src.utils.paths import paths

################################
# ---- InMemory Data loader ----
################################


def load_split_data() -> Dict[str, Dict[str, List[Image.Image]]]:
    """
    Carga imágenes de los directorios 'train', 'val' y 'test' directamente en memoria.

    Esta función es para cuando los datos ya están divididos y procesados.

    Returns:
        Dict[str, Dict[str, List[Image.Image]]]: Diccionario con las claves 'train', 'val', 'test',
            donde cada una contiene un diccionario de {'categoria': [lista_imagenes]}.

    Example:
        >>> data = load_split_data()
        >>> print(f"Train images: {len(data['train']['Blight'])}")
    """
    print("[BUSQUEDA] Cargando datos ya divididos (train/val/test)...")

    # Obtener rutas del sistema centralizado
    env_vars = EnvLoader().get_all()

    # Obtener las categorías esperadas
    try:
        categories = literal_eval(env_vars.get("CLASS_NAMES", "[]"))
    except (ValueError, SyntaxError, TypeError) as e:
        print(f"[ADVERTENCIA] No se pudieron parsear CLASS_NAMES: {e}")
        categories = None

    # Inicializar estructura de datos
    split_data = {
        'train': {},
        'val': {},
        'test': {}
    }

    # Cargar cada split
    for split_name in ['train', 'val', 'test']:
        split_path = getattr(paths, f'data_{split_name}')

        if not split_path.exists():
            print(f"[ADVERTENCIA] No se encontró el directorio {split_name} en: {split_path}")
            continue

        print(f"\n[INFO] Procesando split: {split_name}")

        # Iterar sobre las categorías (directorios dentro de train/val/test)
        for category_dir in split_path.iterdir():
            if not category_dir.is_dir():
                continue

            category = category_dir.name

            try:
                # Cargar imágenes de esta categoría
                images = load_images_from_folder(str(category_dir))
                split_data[split_name][category] = images
                print(f"  [OK] '{category}' cargadas: {len(images)} imágenes")
            except Exception as e:
                print(f"  [ERROR] Fallo al cargar '{category}' en {split_name}: {e}")
                split_data[split_name][category] = []

    # Verificar que se cargaron datos
    total_images = sum(len(imgs) for split in split_data.values() for imgs in split.values())

    if total_images == 0:
        print("\n[ADVERTENCIA] No se cargó ninguna imagen. Verifique las rutas.")
    else:
        print(f"\n[OK] Datos cargados exitosamente. Total: {total_images} imágenes")

    return split_data


def load_raw_data() -> Dict[str, Dict[str, Any]]:
    """
    Carga imágenes de los directorios 'data_1' y 'data_2' en memoria,
    manteniendo la separación e incluyendo las consideraciones de balanceo.

    Returns:
        Dict[str, Dict[str, Any]]: Diccionario con estructura:
            {
                'data_1': {
                    'dataset_consideration': str,
                    'images': Dict[str, List[Image.Image]]
                },
                'data_2': {...}
            }

    Raises:
        FileNotFoundError: Si no se encuentra el directorio raw o subdirectorios data_*.

    Example:
        >>> raw_data = load_raw_data()
        >>> print(f"Datasets: {list(raw_data.keys())}")
    """
    
    print("[BUSQUEDA] Inicializando adaptador de rutas y variables de entorno...")

    # 1. Obtener rutas del sistema centralizado y cargar variables de entorno
    data_raw_path = paths.data_raw
    env_vars = EnvLoader().get_all()

    # 2. Cargar y parsear las consideraciones
    datasets_consideration_str = env_vars.get("DATASETS_CONSIDERATION", "[]")
    print("consideraciones =", datasets_consideration_str)

    try:
        # Convertir la cadena de la lista de consideraciones a una lista/tupla real de Python
        datasets_consideration = literal_eval(datasets_consideration_str)
        if not isinstance(datasets_consideration, (list, tuple)):
             raise ValueError("DATASETS_CONSIDERATION no es una lista/tupla válida.")
    except (ValueError, SyntaxError, TypeError) as e:
        print(f"[ERROR] ERROR: Fallo al parsear DATASETS_CONSIDERATION. Error: {e}")
        datasets_consideration = [] # Fallback seguro

    # 3. Inicializar la estructura de datos final
    raw_data = {}

    # Verificar que el directorio raw existe
    if not data_raw_path.exists():
        raise FileNotFoundError(f"No se encontró el directorio raw en: {data_raw_path}")

    # Determinar qué carpetas en 'raw' deben iterarse (data_1, data_2)
    raw_keys_to_iterate = [d.name for d in data_raw_path.iterdir() if d.is_dir() and d.name.startswith("data_")]

    if not raw_keys_to_iterate:
        raise FileNotFoundError("No se encontraron subdirectorios 'data_1', 'data_2', etc., dentro de la carpeta 'raw'.")

    # 4. Iterar sobre data_1, data_2, etc.
    for i, dataset_key in enumerate(raw_keys_to_iterate):

        # Asignar la consideración de balanceo si está disponible
        consideration = datasets_consideration[i] if i < len(datasets_consideration) else "unknown"

        raw_data[dataset_key] = {
            "dataset_consideration": consideration,
            "images": {}
        }

        dataset_path = data_raw_path / dataset_key

        print(f"\n[INFO] Procesando dataset: {dataset_key} (Consideración: {consideration})")

        # 5. Iterar sobre las categorías (Blight, Common_Rust, etc.)
        for category_dir in dataset_path.iterdir():
            if not category_dir.is_dir():
                continue

            category = category_dir.name

            try:
                # Se asume que 'load_images_from_folder' convierte las imágenes en objetos PIL.Image
                images = load_images_from_folder(str(category_dir))
                raw_data[dataset_key]["images"][category] = images
                print(f"  [OK] '{category}' cargadas: {len(images)} imágenes.")
            except Exception as e:
                print(f"  [ERROR] Fallo al cargar '{category}' en {dataset_key}: {e}")
                # Mantener la entrada vacía para indicar el fallo
                raw_data[dataset_key]["images"][category] = [] 

    # 6. Finalización
    if any(len(d["images"]) > 0 for d in raw_data.values()):
        print("\n[OK] Las imágenes de todos los datasets considerados se han cargado exitosamente.")
    else:
        print("\n[ADVERTENCIA] No se cargó ninguna imagen. Verifique las rutas o el contenido de las carpetas.")
        
    return raw_data


