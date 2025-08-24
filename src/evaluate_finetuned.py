# Archivo: src/evaluate.py

import tensorflow as tf
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import argparse # Biblioteca para manejar argumentos de la terminal

# Importar la función para crear los generadores
from data_pipeline import create_data_generators

def evaluate_model(model_filename: str):
    """
    Carga un modelo específico y lo evalúa en el conjunto de prueba.
    Genera y muestra una matriz de confusión y un reporte de clasificación.
    
    Args:
        model_filename (str): Nombre del archivo del modelo a evaluar (ej. 'fine_tuned_best_model.keras').
    """
    # --- 1. CONFIGURACIÓN ---
    PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
    SPLIT_DATA_DIR = PROJECT_ROOT / 'dataset_split_balanced'
    MODEL_PATH = PROJECT_ROOT / 'models' / model_filename
    
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"El modelo no fue encontrado en '{MODEL_PATH}'. Verifica el nombre del archivo.")

    # --- 2. PREPARAR EL CONJUNTO DE PRUEBA ---
    _, _, test_generator = create_data_generators(
        base_dir=SPLIT_DATA_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    # --- 3. CARGAR EL MODELO Y EVALUAR ---
    print(f"\n🧠 Cargando el modelo desde: '{MODEL_PATH.name}'")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("\n" + "="*70)
    print("📊 Evaluando el modelo en el conjunto de prueba...")
    print("="*70)
    
    loss, accuracy = model.evaluate(test_generator)
    print(f"\nExactitud en el conjunto de prueba: {accuracy * 100:.2f}%")
    print(f"Pérdida en el conjunto de prueba: {loss:.4f}")

    # --- 4. GENERAR MATRIZ DE CONFUSIÓN Y REPORTE ---
    print("\n" + "="*70)
    print("📈 Generando reporte de clasificación y matriz de confusión...")
    print("="*70)

    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    class_names = list(test_generator.class_indices.keys())

    print("\nReporte de Clasificación:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Matriz de Confusión - {MODEL_PATH.name}', fontsize=16)
    plt.ylabel('Clase Verdadera')
    plt.xlabel('Clase Predicha')
    plt.show()

if __name__ == '__main__':
    # --- Configuración de Argumentos de la Terminal ---
    parser = argparse.ArgumentParser(description="Evaluar un modelo de clasificación de imágenes.")
    parser.add_argument(
        '--model',
        type=str,
        default='fine_tuned_best_model.keras', # El modelo por defecto será el mejor y más reciente
        help="Nombre del archivo del modelo a evaluar dentro de la carpeta 'models'."
    )
    args = parser.parse_args()
    
    evaluate_model(model_filename=args.model)