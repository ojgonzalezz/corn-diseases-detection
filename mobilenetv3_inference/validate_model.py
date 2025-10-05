#!/usr/bin/env python3
"""
Sistema de validación y evaluación para modelos MobileNetV3Large de enfermedades del maíz.

Este script proporciona herramientas completas para evaluar el rendimiento de modelos
TensorFlow Lite optimizados:

- Cálculo de métricas de clasificación (accuracy, precision, recall, F1-score)
- Generación de matrices de confusión con visualización
- Validación de requisitos de rendimiento (tamaño, accuracy)
- Análisis comparativo de rendimiento vs tamaño del modelo

Características principales:
- Evaluación completa en conjunto de test
- Visualización de matrices de confusión
- Reportes detallados de clasificación por clase
- Validación de objetivos de optimización
- Guardado automático de resultados

Uso:
    python validate_model.py --model model.tflite --test-data /path/to/test --output results.json

Autor: Sistema de Detección de Enfermedades del Maíz
"""

import os
import sys
import time
import yaml
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Configuración del sistema de logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class ModelValidator:
    """
    Clase para validar y evaluar modelos MobileNetV3Large en TensorFlow Lite.

    Proporciona métricas completas de evaluación, validación de requisitos
    y generación de reportes para modelos optimizados.
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Inicializa el validador de modelos.

        Args:
            config_path: Ruta al archivo de configuración YAML
        """
        self.config = self._load_config(config_path)
        self.interpreter = None
        self.class_names = self.config['data']['classes']  # Nombres de las 4 clases de enfermedades
        self.target_accuracy = self.config['validation']['target_accuracy']  # Accuracy objetivo (85%)
        self.target_size_reduction = self.config['validation']['target_size_reduction']  # Reducción de tamaño objetivo

    def _load_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def load_tflite_model(self, model_path: str):
        self.interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=self.config['inference']['num_threads'])
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def load_test_data(self, test_path: str, max_samples: int = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Carga datos de test para validación.

        Args:
            test_path: Ruta al directorio de test
            max_samples: Número máximo de muestras (None para todas)

        Returns:
            Tuple de (imágenes, etiquetas, rutas)
        """
        from tensorflow.keras.preprocessing import image
        from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenet_v3_preprocess

        logger.info(f"Cargando datos de test desde {test_path}")

        images = []
        labels = []
        image_paths = []

        test_dir = Path(test_path)

        if not test_dir.exists():
            raise FileNotFoundError(f"Directorio de test no encontrado: {test_dir}")

        class_to_idx = {class_name: idx for idx, class_name in enumerate(self.class_names)}

        for class_dir in test_dir.iterdir():
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            if class_name not in class_to_idx:
                logger.warning(f"Clase '{class_name}' no encontrada en configuración, omitiendo")
                continue

            class_idx = class_to_idx[class_name]

            # Obtener archivos de imagen
            image_files = list(class_dir.glob('*.jpg')) + \
                         list(class_dir.glob('*.jpeg')) + \
                         list(class_dir.glob('*.png'))

            for img_path in image_files:
                try:
                    # Cargar y preprocesar imagen
                    img = image.load_img(
                        str(img_path),
                        target_size=tuple(self.config['model']['input_shape'][:2]),
                        interpolation=self.config['preprocessing']['interpolation']
                    )

                    img_array = image.img_to_array(img)

                    # Preprocesamiento MobileNetV3
                    if self.config['preprocessing']['mode'] == 'tf':
                        img_array = mobilenet_v3_preprocess(img_array)
                    else:
                        img_array = img_array / 255.0

                    images.append(img_array)
                    labels.append(class_idx)
                    image_paths.append(str(img_path))

                    if max_samples and len(images) >= max_samples:
                        break

                except Exception as e:
                    logger.warning(f"Error al cargar imagen {img_path}: {e}")
                    continue

            if max_samples and len(images) >= max_samples:
                break

        images = np.array(images)
        labels = np.array(labels)

        logger.info(f"Datos de test cargados: {len(images)} imágenes")
        logger.info(f"Distribución de clases: {np.bincount(labels)}")

        return images, labels, image_paths

    def predict_batch(self, images: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Realiza predicciones en lote.

        Args:
            images: Array de imágenes preprocesadas

        Returns:
            Tuple de (predicciones, tiempo_promedio_ms)
        """
        if self.interpreter is None:
            raise ValueError("Modelo no cargado. Use load_tflite_model() primero.")

        predictions = []
        inference_times = []

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        input_dtype = input_details[0]['dtype']

        logger.info(f"Realizando inferencia en {len(images)} imágenes...")

        for i, img in enumerate(images):
            start_time = time.time()

            # Preparar input según tipo esperado
            input_data = np.expand_dims(img, axis=0)

            if input_dtype != input_data.dtype:
                if input_dtype == np.uint8:
                    input_data = ((input_data + 1) * 127.5).astype(np.uint8)
                elif input_dtype == np.int8:
                    input_data = (input_data * 127).astype(np.int8)

            # Ejecutar inferencia
            self.interpreter.set_tensor(input_details[0]['index'], input_data)
            self.interpreter.invoke()

            # Obtener resultados
            output_data = self.interpreter.get_tensor(output_details[0]['index'])

            # Desnormalizar si es cuantizado
            if output_data.dtype in [np.uint8, np.int8]:
                output_data = output_data.astype(np.float32) / 255.0

            predictions.append(output_data[0])
            inference_times.append((time.time() - start_time) * 1000)

        predictions = np.array(predictions)
        avg_inference_time = np.mean(inference_times)

        logger.info(f"Tiempo promedio de inferencia: {avg_inference_time:.1f} ms")
        return predictions, avg_inference_time

    def evaluate_accuracy(self, predictions: np.ndarray, true_labels: np.ndarray, show_confusion_matrix: bool = True) -> Dict:
        """
        Evalúa la precisión del modelo.

        Args:
            predictions: Predicciones del modelo
            true_labels: Etiquetas verdaderas
            show_confusion_matrix: Si mostrar la matriz de confusión

        Returns:
            Diccionario con métricas de evaluación
        """
        # Obtener clases predichas
        pred_classes = np.argmax(predictions, axis=1)

        # Calcular accuracy
        accuracy = accuracy_score(true_labels, pred_classes)

        # Generar reporte de clasificación
        report = classification_report(
            true_labels,
            pred_classes,
            target_names=self.class_names,
            output_dict=True
        )

        # Generar matriz de confusión
        cm = confusion_matrix(true_labels, pred_classes)

        evaluation = {
            'accuracy': accuracy,
            'accuracy_percent': accuracy * 100,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': pred_classes.tolist(),
            'true_labels': true_labels.tolist()
        }

        logger.info(f"Accuracy: {accuracy:.2f}")
        # Mostrar matriz de confusión si se solicita
        if show_confusion_matrix:
            self._display_confusion_matrix(cm)

        return evaluation

    def _display_confusion_matrix(self, cm: np.ndarray):
        """
        Muestra la matriz de confusión de forma legible.

        Args:
            cm: Matriz de confusión como array numpy
        """
        print("\n" + "="*60)
        print("MATRIZ DE CONFUSIÓN")
        print("="*60)

        # Encabezado
        header = "Real\\Pred".ljust(12)
        for class_name in self.class_names:
            header += f"{class_name[:12]:<12}"
        print(header)
        print("-" * len(header))

        # Filas de la matriz
        for i, class_name in enumerate(self.class_names):
            row = f"{class_name[:12]:<12}"
            for j in range(len(self.class_names)):
                row += f"{cm[i, j]:<12}"
            print(row)

        print("="*60)

        # Métricas por clase
        print("\nMÉTRICAS POR CLASE:")
        print("-" * 40)

        for i, class_name in enumerate(self.class_names):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - (tp + fp + fn)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            print("12"                   ".3f"                   ".3f"                   ".3f")
        print("="*60)

    def check_model_size(self, model_path: str) -> Dict:
        """
        Verifica el tamaño del modelo y calcula reducción.

        Args:
            model_path: Ruta al modelo TFLite

        Returns:
            Diccionario con información de tamaño
        """
        # Tamaño del modelo TFLite
        tflite_size = os.path.getsize(model_path)
        tflite_size_mb = tflite_size / (1024 * 1024)

        # Estimación del tamaño del modelo original (MobileNetV3Large ~20-25MB)
        # Esta es una estimación aproximada
        estimated_original_size_mb = 22.0  # MobileNetV3Large típico
        size_reduction = estimated_original_size_mb / tflite_size_mb if tflite_size_mb > 0 else 0

        size_info = {
            'tflite_size_bytes': tflite_size,
            'tflite_size_mb': tflite_size_mb,
            'estimated_original_size_mb': estimated_original_size_mb,
            'size_reduction_factor': size_reduction
        }

        logger.info(f"Modelo TFLite: {tflite_size_mb:.2f} MB")
        logger.info(f"Reducción de tamaño: {size_reduction:.1f}x")
        return size_info

    def validate_requirements(self, evaluation: Dict, size_info: Dict) -> Dict:
        """
        Valida que el modelo cumple con los requisitos especificados.

        Args:
            evaluation: Resultados de evaluación de precisión
            size_info: Información de tamaño del modelo

        Returns:
            Diccionario con resultados de validación
        """
        validation_results = {
            'accuracy_requirement_met': evaluation['accuracy'] >= self.target_accuracy,
            'size_reduction_requirement_met': size_info['size_reduction_factor'] >= self.target_size_reduction,
            'overall_pass': False
        }

        validation_results['overall_pass'] = (
            validation_results['accuracy_requirement_met'] and
            validation_results['size_reduction_requirement_met']
        )

        logger.info("=== VALIDACIÓN DE REQUISITOS ===")
        logger.info(f"Precisión objetivo: {self.target_accuracy:.0%}")
        logger.info(f"Reducción de tamaño objetivo: {self.target_size_reduction:.2f}x")
        logger.info(f"Precisión requerida: {'✅ CUMPLIDA' if validation_results['accuracy_requirement_met'] else '❌ NO CUMPLIDA'}")
        logger.info(f"Reducción requerida: {'✅ CUMPLIDA' if validation_results['size_reduction_requirement_met'] else '❌ NO CUMPLIDA'}")
        logger.info(f"VALIDACIÓN GENERAL: {'✅ PASADA' if validation_results['overall_pass'] else '❌ FALLIDA'}")

        return validation_results

    def run_full_validation(self, model_path: str, test_data_path: str = "data/test",
                          max_samples: int = None) -> Dict:
        """
        Ejecuta validación completa del modelo.

        Args:
            model_path: Ruta al modelo TFLite
            test_data_path: Ruta a datos de test
            max_samples: Número máximo de muestras para validación

        Returns:
            Diccionario con resultados completos de validación
        """
        logger.info("=== INICIANDO VALIDACIÓN COMPLETA DEL MODELO ===")

        try:
            # 1. Cargar modelo
            self.load_tflite_model(model_path)

            # 2. Cargar datos de test
            images, labels, image_paths = self.load_test_data(test_data_path, max_samples)

            if len(images) == 0:
                raise ValueError("No se pudieron cargar imágenes de test")

            # 3. Realizar predicciones
            predictions, avg_inference_time = self.predict_batch(images)

            # 4. Evaluar precisión
            evaluation = self.evaluate_accuracy(predictions, labels)

            # 5. Verificar tamaño del modelo
            size_info = self.check_model_size(model_path)

            # 6. Validar requisitos
            validation_results = self.validate_requirements(evaluation, size_info)

            # 7. Compilar resultados finales
            results = {
                'model_path': model_path,
                'test_samples': len(images),
                'evaluation': evaluation,
                'size_info': size_info,
                'validation': validation_results,
                'inference_time_ms': avg_inference_time,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }

            logger.info("=== VALIDACIÓN COMPLETADA ===")
            return results

        except Exception as e:
            logger.error(f"Error durante validación: {e}")
            raise

def main():
    """Función principal para validación."""
    import argparse

    parser = argparse.ArgumentParser(description='MobileNetV3Large Model Validator')
    parser.add_argument('--config', default='config.yaml', help='Ruta al archivo de configuración')
    parser.add_argument('--model', required=True, help='Ruta al modelo TFLite')
    parser.add_argument('--test-data', default='data/test', help='Ruta a datos de test')
    parser.add_argument('--max-samples', type=int, help='Número máximo de muestras para validación')
    parser.add_argument('--output', help='Ruta para guardar resultados JSON')

    args = parser.parse_args()

    try:
        # Inicializar validador
        validator = ModelValidator(args.config)

        # Ejecutar validación completa
        results = validator.run_full_validation(
            args.model,
            args.test_data,
            args.max_samples
        )

        # Mostrar resultados principales
        print("\n=== RESULTADOS DE VALIDACIÓN ===")
        print(f"Modelo: {results['model_path']}")
        print(f"Muestras de test: {results['test_samples']}")
        print(f"Precisión: {results['evaluation']['accuracy']:.3f}")
        print(f"Precisión por clase: {results['evaluation']['precision_per_class']}")
        print(f"Recall por clase: {results['evaluation']['recall_per_class']}")
        print(f"Tamaño del modelo: {results['size_info']['tflite_size_mb']:.2f} MB")
        print(f"Precisión objetivo cumplida: {results['validation']['accuracy_requirement_met']}")
        print(f"Reducción de tamaño cumplida: {results['validation']['size_reduction_requirement_met']}")
        print(f"VALIDACIÓN GENERAL: {'PASADA' if results['validation']['overall_pass'] else 'FALLIDA'}")

        # Guardar resultados si se especifica
        if args.output:
            import json
            with open(args.output, 'w') as f:
                # Convertir arrays numpy a listas para serialización JSON
                json_results = results.copy()
                json_results['evaluation']['predictions'] = results['evaluation']['predictions']
                json_results['evaluation']['true_labels'] = results['evaluation']['true_labels']
                json.dump(json_results, f, indent=2, default=str)
            logger.info(f"Resultados guardados en: {args.output}")

        # Exit code basado en validación
        sys.exit(0 if results['validation']['overall_pass'] else 1)

    except Exception as e:
        logger.error(f"Error en validación: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
