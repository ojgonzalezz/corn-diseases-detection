#!/usr/bin/env python3
"""
MobileNetV3Large Inference Pipeline para Edge Computing
Implementación optimizada para dispositivos edge con baja latencia y bajo consumo.
"""

import os
import sys
import time
import yaml
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenet_v3_preprocess

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MobileNetV3Inference:
    """Clase para inferencia optimizada con MobileNetV3Large en edge devices."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Inicializa el pipeline de inferencia.

        Args:
            config_path: Ruta al archivo de configuración YAML
        """
        self.config = self._load_config(config_path)
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.class_names = self.config['data']['classes']

        logger.info("Inicializando MobileNetV3Large Inference Pipeline")
        logger.info(f"Modelo: {self.config['model']['name']}")
        logger.info(f"Input shape: {self.config['model']['input_shape']}")
        logger.info(f"Threads: {self.config['inference']['num_threads']}")

    def _load_config(self, config_path: str) -> dict:
        """Carga la configuración desde archivo YAML."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuración cargada desde {config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Archivo de configuración no encontrado: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error al parsear configuración YAML: {e}")
            raise

    def load_model(self, model_path: str):
        """
        Carga el modelo TFLite optimizado.

        Args:
            model_path: Ruta al archivo .tflite
        """
        try:
            # Cargar el modelo TFLite
            self.interpreter = tf.lite.Interpreter(
                model_path=model_path,
                num_threads=self.config['inference']['num_threads']
            )

            # Configurar delegates si están disponibles
            if self.config['inference']['use_xnnpack_delegate']:
                try:
                    xnnpack_delegate = tf.lite.experimental.load_delegate('libXNNPACK.so')
                    self.interpreter = tf.lite.Interpreter(
                        model_path=model_path,
                        experimental_delegates=[xnnpack_delegate],
                        num_threads=self.config['inference']['num_threads']
                    )
                    logger.info("XNNPACK delegate cargado exitosamente")
                except Exception as e:
                    logger.warning(f"No se pudo cargar XNNPACK delegate: {e}")

            # Asignar tensores
            self.interpreter.allocate_tensors()

            # Obtener detalles de entrada/salida
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            logger.info(f"Modelo TFLite cargado desde {model_path}")
            logger.info(f"Input shape: {self.input_details[0]['shape']}")
            logger.info(f"Output shape: {self.output_details[0]['shape']}")

        except Exception as e:
            logger.error(f"Error al cargar modelo TFLite: {e}")
            raise

    def preprocess_image(self, img_path: str) -> np.ndarray:
        """
        Preprocesa una imagen para MobileNetV3Large.

        Args:
            img_path: Ruta a la imagen

        Returns:
            Array preprocesado listo para inferencia
        """
        try:
            # Cargar imagen
            img = image.load_img(
                img_path,
                target_size=tuple(self.config['model']['input_shape'][:2]),
                interpolation=self.config['preprocessing']['interpolation']
            )

            # Convertir a array
            img_array = image.img_to_array(img)

            # Aplicar preprocesamiento específico de MobileNetV3
            if self.config['preprocessing']['mode'] == 'tf':
                # Preprocesamiento MobileNetV3: escala [-1, 1]
                img_array = mobilenet_v3_preprocess(img_array)
            else:
                # Normalización básica [0,1]
                img_array = img_array / 255.0

            # Expandir dimensiones para batch
            img_array = np.expand_dims(img_array, axis=0)

            return img_array.astype(np.float32)

        except Exception as e:
            logger.error(f"Error al preprocesar imagen {img_path}: {e}")
            raise

    def predict(self, img_path: str) -> dict:
        """
        Realiza predicción en una imagen.

        Args:
            img_path: Ruta a la imagen

        Returns:
            Diccionario con resultados de predicción
        """
        if self.interpreter is None:
            raise ValueError("Modelo no cargado. Use load_model() primero.")

        try:
            start_time = time.time()

            # Preprocesar imagen
            input_data = self.preprocess_image(img_path)

            # Verificar tipo de datos esperado por el modelo
            input_dtype = self.input_details[0]['dtype']
            if input_dtype != input_data.dtype:
                if input_dtype == np.uint8:
                    # Convertir a uint8 para cuantización
                    input_data = ((input_data + 1) * 127.5).astype(np.uint8)
                elif input_dtype == np.int8:
                    # Convertir a int8
                    input_data = (input_data * 127).astype(np.int8)

            # Establecer tensor de entrada
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

            # Ejecutar inferencia
            self.interpreter.invoke()

            # Obtener resultados
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

            # Calcular tiempo de inferencia
            inference_time = time.time() - start_time

            # Procesar resultados
            if output_data.dtype in [np.uint8, np.int8]:
                # Desnormalizar si es cuantizado
                probabilities = output_data[0].astype(np.float32) / 255.0
            else:
                probabilities = output_data[0]

            # Obtener clase predicha
            predicted_class_idx = np.argmax(probabilities)
            predicted_class = self.class_names[predicted_class_idx]
            confidence = float(probabilities[predicted_class_idx])

            # Crear diccionario de probabilidades por clase
            class_probabilities = {
                class_name: float(prob)
                for class_name, prob in zip(self.class_names, probabilities)
            }

            result = {
                'predicted_class': predicted_class,
                'predicted_index': int(predicted_class_idx),
                'confidence': confidence,
                'probabilities': class_probabilities,
                'inference_time_ms': inference_time * 1000,
                'image_path': img_path
            }

            logger.info(".3f"            logger.info(f"Predicción: {predicted_class} (confianza: {confidence:.3f})")

            return result

        except Exception as e:
            logger.error(f"Error durante predicción en {img_path}: {e}")
            return {
                'error': str(e),
                'image_path': img_path
            }

    def predict_batch(self, image_paths: List[str]) -> List[dict]:
        """
        Realiza predicciones en un lote de imágenes.

        Args:
            image_paths: Lista de rutas de imágenes

        Returns:
            Lista de diccionarios con resultados
        """
        results = []
        total_time = 0

        logger.info(f"Procesando {len(image_paths)} imágenes...")

        for img_path in image_paths:
            result = self.predict(img_path)
            results.append(result)

            if 'inference_time_ms' in result:
                total_time += result['inference_time_ms']

        avg_time = total_time / len(image_paths) if image_paths else 0
        logger.info(".2f"
        return results

    def load_sample_images(self, num_samples: int = 10) -> List[str]:
        """
        Carga rutas de imágenes de muestra desde el directorio de datos.

        Args:
            num_samples: Número de imágenes a cargar

        Returns:
            Lista de rutas de imágenes
        """
        image_paths = []

        # Buscar imágenes en los directorios de datos
        data_dirs = [
            Path(self.config['data']['test_path']),
            Path(self.config['data']['val_path']),
            Path(self.config['data']['train_path'])
        ]

        for data_dir in data_dirs:
            if data_dir.exists():
                for class_dir in data_dir.iterdir():
                    if class_dir.is_dir():
                        # Obtener archivos de imagen
                        image_files = list(class_dir.glob('*.jpg')) + \
                                    list(class_dir.glob('*.jpeg')) + \
                                    list(class_dir.glob('*.png'))

                        image_paths.extend([str(f) for f in image_files])

                        if len(image_paths) >= num_samples:
                            break

                if len(image_paths) >= num_samples:
                    break

        return image_paths[:num_samples]

def main():
    """Función principal para ejecutar inferencia."""
    import argparse

    parser = argparse.ArgumentParser(description='MobileNetV3Large Inference Pipeline')
    parser.add_argument('--config', default='config.yaml', help='Ruta al archivo de configuración')
    parser.add_argument('--model', required=True, help='Ruta al modelo TFLite')
    parser.add_argument('--image', help='Ruta a una imagen para predicción')
    parser.add_argument('--batch', action='store_true', help='Procesar imágenes de muestra')
    parser.add_argument('--num-samples', type=int, default=10, help='Número de muestras para procesamiento batch')

    args = parser.parse_args()

    try:
        # Inicializar pipeline
        inference = MobileNetV3Inference(args.config)

        # Cargar modelo
        inference.load_model(args.model)

        if args.image:
            # Predicción en imagen individual
            result = inference.predict(args.image)
            print("\nResultado de predicción:")
            print(f"Imagen: {result['image_path']}")
            print(f"Clase predicha: {result['predicted_class']}")
            print(".3f"            print(".1f"

        elif args.batch:
            # Predicción en lote de imágenes de muestra
            image_paths = inference.load_sample_images(args.num_samples)
            if not image_paths:
                logger.error("No se encontraron imágenes en los directorios de datos")
                return

            logger.info(f"Procesando {len(image_paths)} imágenes de muestra...")
            results = inference.predict_batch(image_paths)

            # Calcular estadísticas
            successful_predictions = [r for r in results if 'predicted_class' in r]
            avg_confidence = np.mean([r['confidence'] for r in successful_predictions])
            avg_time = np.mean([r['inference_time_ms'] for r in successful_predictions])

            print(f"\nEstadísticas del lote:")
            print(f"Imágenes procesadas: {len(successful_predictions)}/{len(results)}")
            print(f"Confianza promedio: {avg_confidence:.3f}")
            print(f"Tiempo promedio: {avg_time:.1f} ms")
        else:
            logger.error("Especifique --image para predicción individual o --batch para procesamiento de lote")

    except Exception as e:
        logger.error(f"Error en ejecución principal: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
