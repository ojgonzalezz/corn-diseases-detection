#####################################################################################
# ----------------------------------- Model Trainer ---------------------------------
#####################################################################################

#########################
# ---- Depdendencies ----
#########################

import tensorflow as tf
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import argparse 

from src.pipelines.preprocess import split_and_balance_dataset
from src.utils.data_augmentator import DataAugmenter
from src.utils.image_modifier import ImageAugmentor
from src.utils.utils import flatten_data

##########################
# ---- Evaluate model ----
##########################

def evaluate_model(model_filename: str, split_ratios=(0.7, 0.15, 0.15)):
    """
    Carga un modelo específico y lo evalúa en el conjunto de prueba.
    Genera y muestra una matriz de confusión y un reporte de clasificación.
    
    Args:
        model_filename (str): Nombre del archivo del modelo a evaluar (ej. 'best_model.keras').
    """
    # --- 1. CONFIGURACIÓN ---
    PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
    DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
    MODEL_PATH = PROJECT_ROOT / 'models' / 'exported' / model_filename 
    IMAGE_SIZE = (224, 224)
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"El modelo no fue encontrado en '{MODEL_PATH}'. Verifica el nombre del archivo.")

    # --- 2. CARGAR Y PREPARAR EL CONJUNTO DE PRUEBA ---
    print("\n📦 Cargando y preparando los datos de prueba en memoria...")
    
    # Cargar el dataset usando la función de preprocesamiento
    raw_dataset = split_and_balance_dataset(
        # Usamos 1.0 para el ratio de prueba porque solo necesitamos este set
        balanced="downsample",  # Changed from True to string
        split_ratios=split_ratios
    )

    # Get class names from the dataset
    class_names = list(raw_dataset['test'].keys())

    # Use shared flatten_data function from utils
    X_test, y_test_labels = flatten_data(raw_dataset['test'], image_size=IMAGE_SIZE)
    
    # Codificar etiquetas y asegurar el formato correcto
    label_to_int = {label: i for i, label in enumerate(class_names)}
    y_test = np.array([label_to_int[l] for l in y_test_labels])
    
    print("✅ Datos de prueba cargados y listos para evaluación.")

    # --- 3. CARGAR EL MODELO Y EVALUAR ---
    print(f"\n🧠 Cargando el modelo desde: '{MODEL_PATH.name}'")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("\n" + "="*70)
    print("📊 Evaluando el modelo en el conjunto de prueba...")
    print("="*70)
    
    # Evaluación con los datos en arrays
    loss, accuracy = model.evaluate(x=X_test, y=tf.keras.utils.to_categorical(y_test))
    print(f"\nExactitud en el conjunto de prueba: {accuracy * 100:.2f}%")
    print(f"Pérdida en el conjunto de prueba: {loss:.4f}")

    # --- 4. GENERAR MATRIZ DE CONFUSIÓN Y REPORTE ---
    print("\n" + "="*70)
    print("📈 Generando reporte de clasificación y matriz de confusión...")
    print("="*70)

    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)
    
    # y_true son las etiquetas numéricas originales
    y_true = y_test
    
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
        default='best_VGG16.keras', # Asume que este es el nombre de tu modelo
        help="Nombre del archivo del modelo a evaluar dentro de la carpeta 'models/exported'."
    )
    args = parser.parse_args()
    
    evaluate_model(model_filename=args.model)


###############################################
# ---- Evaluate model in augmented dataset ----
###############################################

def augmented_evaluation(model_filename: str, aug_type="spacial"):
    """
    Carga y evalúa un modelo en un conjunto de datos de prueba aumentado.
    """
    # --- 1. CONFIGURACIÓN ---
    PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
    DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
    MODEL_PATH = PROJECT_ROOT / 'models' / 'exported' / model_filename 
    IMAGE_SIZE = (224, 224)
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"El modelo no fue encontrado en '{MODEL_PATH}'. Verifica el nombre del archivo.")

    # --- 2. CARGAR Y PREPARAR EL CONJUNTO DE PRUEBA ---
    print("\n📦 Cargando y preparando los datos de prueba en memoria...")
    
    raw_dataset = split_and_balance_dataset(
        # Solo necesitas cargar el set de prueba para la evaluacion
        balanced="downsample",  # Changed from True to string
        split_ratios=(0.7, 0.15, 0.15)
        # Note: base_path parameter removed - function doesn't accept it
    )
    
    test_data = raw_dataset['test']

    augmenter = {
        "spacial":DataAugmenter(),
        "quality": ImageAugmentor()
    }

    # Get class names from the dataset
    class_names = list(raw_dataset['test'].keys())

    # Use shared flatten_data function from utils
    X_test_original, y_test_labels = flatten_data(raw_dataset['test'], image_size=IMAGE_SIZE)
    
    label_to_int = {label: i for i, label in enumerate(class_names)}
    y_test_original = np.array([label_to_int[l] for l in y_test_labels])
    
    # 💥 Aplicar data augmentation al conjunto de prueba original
    X_test_augmented, y_test_augmented = augmenter.augment_dataset(
        images=X_test_original,
        labels=y_test_original,
        p=0.4
    )
    
    print("✅ Datos de prueba originales y aumentados listos para la evaluación.")

    # --- 3. CARGAR EL MODELO Y EVALUAR ---
    print(f"\n🧠 Cargando el modelo desde: '{MODEL_PATH.name}'")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("\n" + "="*70)
    print("📊 Evaluando el modelo en el conjunto de prueba AUMENTADO...")
    print("="*70)
    
    # ⭐ La evaluación y predicción se hacen sobre los datos aumentados ⭐
    loss, accuracy = model.evaluate(x=X_test_augmented, y=tf.keras.utils.to_categorical(y_test_augmented))
    print(f"\nExactitud en el conjunto de prueba AUMENTADO: {accuracy * 100:.2f}%")
    print(f"Pérdida en el conjunto de prueba AUMENTADO: {loss:.4f}")

    # --- 4. GENERAR MATRIZ DE CONFUSIÓN Y REPORTE ---
    print("\n" + "="*70)
    print("📈 Generando reporte de clasificación y matriz de confusión (con datos AUMENTADOS)...")
    print("="*70)

    predictions = model.predict(X_test_augmented)
    y_pred = np.argmax(predictions, axis=1)
    
    y_true = y_test_augmented
    
    print("\nReporte de Clasificación:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Matriz de Confusión - {MODEL_PATH.name} (Datos AUMENTADOS)', fontsize=16)
    plt.ylabel('Clase Verdadera')
    plt.xlabel('Clase Predicha')
    plt.show()
    