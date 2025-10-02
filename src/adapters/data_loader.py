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
from dotenv import load_dotenv
from ast import literal_eval 

# Environment variables loading
sys.path.append(os.path.abspath(os.path.join("..", "src")))
from src.core.load_env import EnvLoader
from src.utils.utils import *

from src.core.path_finder import ProjectPaths

################################
# ---- InMemory Data loader ----
################################


def load_raw_data():
    """
    Carga imágenes de los directorios 'data_1' y 'data_2' en memoria, 
    manteniendo la separación e incluyendo las consideraciones de balanceo.
    """
    
    print("[BUSQUEDA] Inicializando adaptador de rutas y variables de entorno...")
    
    # 1. Inicializar Path Finder y Environment Loader
    pp = ProjectPaths(data_subpath=("data", "raw")) 
    data_paths = pp.get_structure()
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
    
    # Determinar qué carpetas en 'raw' deben iterarse (data_1, data_2)
    raw_keys_to_iterate = [key for key in data_paths.keys() if key.startswith("data_")]
    
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
        
        dataset_path_structure = data_paths[dataset_key]
        
        print(f"\n[INFO] Procesando dataset: {dataset_key} (Consideración: {consideration})")

        # 5. Iterar sobre las categorías (Blight, Common_Rust, etc.)
        for category, path in dataset_path_structure.items():
            if isinstance(path, dict):
                # Esto es una carpeta intermedia, saltar
                continue 
            
            # El valor de 'path' es la ruta absoluta (str) a la carpeta de clase
            try:
                # Se asume que 'load_images_from_folder' convierte las imágenes en objetos PIL.Image
                images = load_images_from_folder(path)
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


