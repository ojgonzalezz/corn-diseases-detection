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

# Environment variables loading
sys.path.append(os.path.abspath(os.path.join("..", "src")))
from src.core.load_env import EnvLoader
from src.utils.utils import *

################################
# ---- InMemory Data loader ----
################################

# Data folder detections
from src.core.path_finder import ProjectPaths

pp = ProjectPaths()
data_paths = pp.get_structure()

def load_raw_data():
    raw_data = {}
    for category, path in data_paths['raw']['data'].items():
        raw_data[category] = load_images_from_folder(path)
    print("✅ Las imágenes de todas las categorías se han cargado exitosamente")
    return raw_data

