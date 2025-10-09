"""
Configuración común para el entrenamiento de modelos
Compatible con Google Colab (con Drive) y entornos locales
"""

import os
from pathlib import Path

# Detectar si estamos en Google Colab
try:
    import google.colab
    IN_COLAB = True

    # Verificar si Drive está montado
    drive_path = Path('/content/drive/MyDrive')
    if not drive_path.exists():
        print("\n" + "!" * 70)
        print("ERROR: Google Drive no está montado")
        print("!" * 70)
        print("\nAntes de ejecutar el script, debes montar Google Drive:")
        print("\n  from google.colab import drive")
        print("  drive.mount('/content/drive')")
        print("\nLuego vuelve a ejecutar el script de entrenamiento.")
        print("!" * 70 + "\n")
        raise RuntimeError("Google Drive no está montado. Monta Drive primero.")

    # Rutas en Google Drive
    DRIVE_BASE = Path('/content/drive/MyDrive/corn-diseases-detection')
    BASE_DIR = Path('/content/corn-diseases-detection')

    # Dataset desde Drive
    DATA_DIR = DRIVE_BASE / 'data_processed'

    # Salidas en Drive (persistentes)
    MODELS_DIR = DRIVE_BASE / 'models'
    LOGS_DIR = DRIVE_BASE / 'logs'
    MLFLOW_DIR = DRIVE_BASE / 'mlruns'

    # Verificar que el dataset existe
    if not DATA_DIR.exists():
        print("\n" + "!" * 70)
        print("ERROR: No se encontró el dataset en Google Drive")
        print("!" * 70)
        print(f"Ruta esperada: {DATA_DIR}")
        print("\nAsegúrate de haber subido la carpeta 'data_processed' a:")
        print("  Mi unidad/corn-diseases-detection/data_processed/")
        print("\nLa estructura debe ser:")
        print("  Mi unidad/")
        print("  └── corn-diseases-detection/")
        print("      └── data_processed/")
        print("          ├── Blight/")
        print("          ├── Common_Rust/")
        print("          ├── Gray_Leaf_Spot/")
        print("          └── Healthy/")
        print("!" * 70 + "\n")
        raise FileNotFoundError(f"Dataset no encontrado en {DATA_DIR}")

    print("\n" + "=" * 70)
    print("CONFIGURACIÓN GOOGLE COLAB + DRIVE")
    print("=" * 70)
    print(f"✓ Drive montado correctamente")
    print(f"✓ Dataset encontrado: {DATA_DIR}")
    print(f"✓ Modelos se guardarán en: {MODELS_DIR}")
    print(f"✓ Logs se guardarán en: {LOGS_DIR}")
    print("=" * 70 + "\n")

except ImportError:
    IN_COLAB = False
    BASE_DIR = Path(__file__).parent.parent

    # Rutas locales
    DATA_DIR = BASE_DIR / 'data_processed'
    MODELS_DIR = BASE_DIR / 'entrenamiento_modelos' / 'models'
    LOGS_DIR = BASE_DIR / 'entrenamiento_modelos' / 'logs'
    MLFLOW_DIR = BASE_DIR / 'entrenamiento_modelos' / 'mlruns'

# Crear directorios
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
MLFLOW_DIR.mkdir(parents=True, exist_ok=True)

# Clases
CLASSES = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']
NUM_CLASSES = len(CLASSES)

# Hiperparámetros comunes
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5

# División del dataset
TRAIN_SPLIT = 0.70  # 70% entrenamiento
VAL_SPLIT = 0.15    # 15% validación
TEST_SPLIT = 0.15   # 15% prueba

# Semilla para reproducibilidad
RANDOM_SEED = 42

# GPU Configuration
# En Google Colab, la GPU se gestiona automáticamente
# En entornos locales, puedes ajustar este valor según tu GPU
GPU_MEMORY_LIMIT = None  # None = usar toda la memoria disponible

# MLflow
MLFLOW_TRACKING_URI = f"file:///{MLFLOW_DIR}"
MLFLOW_EXPERIMENT_NAME = "corn_disease_classification"

# Data Augmentation
DATA_AUGMENTATION = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'horizontal_flip': True,
    'vertical_flip': True,
    'zoom_range': 0.2,
    'fill_mode': 'nearest'
}
