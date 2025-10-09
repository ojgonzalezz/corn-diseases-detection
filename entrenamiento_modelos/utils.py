"""
Utilidades comunes para el entrenamiento de modelos
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf

def setup_gpu(memory_limit=None):
    """
    Configurar GPU para entrenamiento
    Compatible con Google Colab y entornos locales

    Args:
        memory_limit: Límite de memoria en MB. Si es None, usa toda la memoria disponible.
                     En Google Colab, se recomienda None para usar la GPU completa.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            if memory_limit is not None:
                # Configurar límite de memoria (solo en entornos locales)
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                )
                print(f"GPU configurada con límite de {memory_limit}MB")
            else:
                # Permitir crecimiento dinámico de memoria (recomendado para Colab)
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU configurada con crecimiento dinámico de memoria")

            print(f"GPUs disponibles: {len(gpus)}")
            print(f"GPU en uso: {gpus[0].name}")
        except RuntimeError as e:
            print(f"Error configurando GPU: {e}")
    else:
        print("No se detectaron GPUs. Usando CPU.")

def create_data_generators(data_dir, image_size, batch_size, train_split, val_split,
                          test_split, random_seed, augmentation_params):
    """
    Crear generadores de datos para entrenamiento, validación y prueba

    IMPORTANTE: El dataset en data_processed YA TIENE data augmentation aplicado
    en el preprocesamiento (rotación, flips, brillo, contraste + balanceo de clases).
    Por lo tanto, NO aplicamos augmentation adicional aquí para evitar:
    - Doble transformación de imágenes
    - Sobre-regularización que reduce accuracy
    - Imágenes artificialmente distorsionadas

    NOTA: Esta implementación usa ImageDataGenerator con validation_split.
    - train_generator: usa el subset 'training' SIN augmentation (solo rescale)
    - validation_data: usa el subset 'validation' sin augmentation
    - test_generator: REUTILIZA el subset 'validation' sin augmentation

    Los conjuntos de validación y prueba son idénticos en esta implementación.
    Esto es común en transfer learning cuando el dataset no es muy grande.
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    # Generador SOLO con rescale para entrenamiento (NO augmentation)
    # El augmentation YA fue aplicado en preprocessing
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=val_split + test_split
        # NO se pasa **augmentation_params porque causaría doble augmentation
    )

    # Generador sin augmentation para validación/prueba
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=val_split + test_split
    )

    # Generador de entrenamiento
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=random_seed
    )

    # Generador de validación (mismo que test, sin augmentation)
    validation_data = val_datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=random_seed
    )

    # Generador de prueba (reutiliza el conjunto de validación)
    test_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=random_seed
    )

    return train_generator, validation_data, test_generator

def plot_training_history(history, save_path):
    """Graficar historial de entrenamiento"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Graficar matriz de confusión"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return cm

def evaluate_model(model, test_generator, class_names):
    """Evaluar modelo y generar métricas"""
    # Predicciones
    y_pred_probs = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_generator.classes

    # Métricas
    report = classification_report(y_true, y_pred,
                                   target_names=class_names,
                                   output_dict=True)

    # Accuracy total
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)

    return {
        'test_loss': float(test_loss),
        'test_accuracy': float(test_accuracy),
        'y_true': y_true,
        'y_pred': y_pred,
        'classification_report': report
    }

def save_training_log(log_path, model_name, hyperparameters, history,
                     evaluation_results, confusion_matrix, training_time):
    """Guardar log de entrenamiento"""

    log_data = {
        'model_name': model_name,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'training_time_seconds': training_time,
        'hyperparameters': hyperparameters,
        'training_history': {
            'final_train_accuracy': float(history.history['accuracy'][-1]),
            'final_val_accuracy': float(history.history['val_accuracy'][-1]),
            'final_train_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
            'best_val_accuracy': float(max(history.history['val_accuracy'])),
            'epochs_trained': len(history.history['accuracy'])
        },
        'test_results': {
            'test_accuracy': evaluation_results['test_accuracy'],
            'test_loss': evaluation_results['test_loss'],
            'classification_report': evaluation_results['classification_report']
        },
        'confusion_matrix': confusion_matrix.tolist()
    }

    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2)

    # También crear versión legible
    txt_log_path = str(log_path).replace('.json', '.txt')
    with open(txt_log_path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*60}\n")
        f.write(f"LOG DE ENTRENAMIENTO - {model_name}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Fecha: {log_data['timestamp']}\n")
        f.write(f"Tiempo de entrenamiento: {training_time:.2f} segundos ({training_time/60:.2f} minutos)\n\n")

        f.write(f"--- Hiperparametros ---\n")
        for key, value in hyperparameters.items():
            f.write(f"  {key}: {value}\n")

        f.write(f"\n--- Resultados de Entrenamiento ---\n")
        f.write(f"  Epocas entrenadas: {log_data['training_history']['epochs_trained']}\n")
        f.write(f"  Train Accuracy final: {log_data['training_history']['final_train_accuracy']:.4f}\n")
        f.write(f"  Val Accuracy final: {log_data['training_history']['final_val_accuracy']:.4f}\n")
        f.write(f"  Mejor Val Accuracy: {log_data['training_history']['best_val_accuracy']:.4f}\n")
        f.write(f"  Train Loss final: {log_data['training_history']['final_train_loss']:.4f}\n")
        f.write(f"  Val Loss final: {log_data['training_history']['final_val_loss']:.4f}\n")

        f.write(f"\n--- Resultados en Test ---\n")
        f.write(f"  Test Accuracy: {log_data['test_results']['test_accuracy']:.4f}\n")
        f.write(f"  Test Loss: {log_data['test_results']['test_loss']:.4f}\n")

        f.write(f"\n--- Classification Report ---\n")
        report = log_data['test_results']['classification_report']
        for class_name, metrics in report.items():
            if isinstance(metrics, dict):
                f.write(f"\n{class_name}:\n")
                f.write(f"  Precision: {metrics.get('precision', 0):.4f}\n")
                f.write(f"  Recall: {metrics.get('recall', 0):.4f}\n")
                f.write(f"  F1-Score: {metrics.get('f1-score', 0):.4f}\n")
                f.write(f"  Support: {metrics.get('support', 0)}\n")

        f.write(f"\n--- Matriz de Confusion ---\n")
        f.write(f"{confusion_matrix}\n")

    return log_data
