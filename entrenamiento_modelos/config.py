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
    BASE_DIR = Path('/content/corn-diseases-detection')

    # Dataset desde Drive (raíz de MyDrive)
    DATA_DIR = Path('/content/drive/MyDrive/data_processed')

    # Salidas en Drive (dentro de carpeta del proyecto)
    DRIVE_PROJECT = Path('/content/drive/MyDrive/corn-diseases-detection')
    MODELS_DIR = DRIVE_PROJECT / 'models'
    LOGS_DIR = DRIVE_PROJECT / 'logs'
    MLFLOW_DIR = DRIVE_PROJECT / 'mlruns'

    # Verificar que el dataset existe
    if not DATA_DIR.exists():
        print("\n" + "!" * 70)
        print("ERROR: No se encontró el dataset en Google Drive")
        print("!" * 70)
        print(f"Ruta esperada: {DATA_DIR}")
        print("\nAsegúrate de haber subido la carpeta 'data_processed' a:")
        print("  Mi unidad/data_processed/")
        print("\nLa estructura debe ser:")
        print("  Mi unidad/")
        print("  └── data_processed/")
        print("      ├── Blight/")
        print("      ├── Common_Rust/")
        print("      ├── Gray_Leaf_Spot/")
        print("      └── Healthy/")
        print("!" * 70 + "\n")
        raise FileNotFoundError(f"Dataset no encontrado en {DATA_DIR}")

    # Verificar que hay GPU disponible (OBLIGATORIO en Colab)
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("\n" + "!" * 70)
        print("ERROR: NO SE DETECTÓ GPU")
        print("!" * 70)
        print("\nPara usar GPU en Google Colab:")
        print("1. Ve a: Runtime > Change runtime type")
        print("2. Selecciona: Hardware accelerator > GPU")
        print("3. Haz clic en 'Save'")
        print("4. Reconecta y vuelve a ejecutar")
        print("!" * 70 + "\n")
        raise RuntimeError("GPU no disponible. Configura GPU en Colab primero.")

    print("\n" + "=" * 70)
    print("CONFIGURACIÓN GOOGLE COLAB + DRIVE")
    print("=" * 70)
    print(f"✓ Drive montado correctamente")
    print(f"✓ Dataset encontrado: {DATA_DIR}")
    print(f"✓ GPU detectada: {gpus[0].name}")
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

# Hiperparámetros comunes - OPTIMIZADOS para >85% accuracy y >80% recall
IMAGE_SIZE = (256, 256)  # Mantener según preprocesamiento
BATCH_SIZE = 64  # Sweet spot para A100: balance entre velocidad y generalización
EPOCHS = 40  # Aumentado de 20 a 40 para permitir mejor convergencia
LEARNING_RATE = 0.001  # Mantener para fase inicial
EARLY_STOPPING_PATIENCE = 15  # Aumentado para dar más tiempo a mejorar
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

# Data Augmentation - DESACTIVADO porque ya se aplicó en preprocessing
# El dataset en data_processed YA tiene:
# - Imágenes balanceadas (3690 por clase)
# - Augmentation aplicado (rotación, flips, brillo, contraste)
# - Normalización de brillo y dimensiones (256x256)
# Aplicar augmentation aquí causaría DOBLE transformación y PEOR rendimiento
DATA_AUGMENTATION = {
    # Sin transformaciones adicionales - solo rescale se aplica en utils.py
}
