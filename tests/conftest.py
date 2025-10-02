"""
Configuración y fixtures compartidas para pytest.
"""
import sys
import os
from pathlib import Path
import pytest
import numpy as np
from PIL import Image

# Agregar el directorio raíz al path para imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_image():
    """Genera una imagen de prueba RGB."""
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(img_array, mode='RGB')


@pytest.fixture
def sample_image_batch():
    """Genera un lote de imágenes de prueba."""
    images = []
    for _ in range(5):
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        images.append(Image.fromarray(img_array, mode='RGB'))
    return images


@pytest.fixture
def sample_dataset():
    """Genera un dataset de prueba con múltiples clases."""
    dataset = {
        'Blight': [],
        'Common_Rust': [],
        'Gray_Leaf_Spot': [],
        'Healthy': []
    }

    # Generar 10 imágenes por clase
    for class_name in dataset.keys():
        for _ in range(10):
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array, mode='RGB')
            dataset[class_name].append(img)

    return dataset


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock de variables de entorno para pruebas."""
    env_vars = {
        'IMAGE_SIZE': '(224, 224)',
        'NUM_CLASSES': '4',
        'CLASS_NAMES': "['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']",
        'BATCH_SIZE': '32',
        'MAX_TRIALS': '10',
        'TUNER_EPOCHS': '10',
        'FACTOR': '3',
        'MAX_EPOCHS': '20',
        'SPLIT_RATIOS': '(0.7, 0.15, 0.15)',
        'IM_SIM_TRESHOLD': '0.95',
        'MAX_ADDED_BALANCE': '50'
    }

    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    return env_vars
