#!/usr/bin/env python3
"""
Conversión de MobileNetV3Large a TensorFlow Lite con Optimizaciones
Incluye poda (pruning), cuantización INT8 y optimizaciones para edge computing.
"""

import os
import sys
import yaml
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenet_v3_preprocess
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
# Configurar logging ANTES de usarlo
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import tensorflow_model_optimization as tfmot
    PRUNING_AVAILABLE = True
except ImportError:
    logger.warning("tensorflow-model-optimization no disponible. El pruning será omitido.")
    PRUNING_AVAILABLE = False
    tfmot = None

class MobileNetV3TFLiteConverter:
    """Clase para conversión optimizada de MobileNetV3Large a TFLite."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Inicializa el convertidor TFLite.

        Args:
            config_path: Ruta al archivo de configuración YAML
        """
        self.config = self._load_config(config_path)
        self.model = None
        self.pruned_model = None

        logger.info("Inicializando MobileNetV3Large TFLite Converter")
        logger.info(f"Modelo base: {self.config['model']['name']}")

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

    def build_model(self) -> tf.keras.Model:
        """
        Construye el modelo MobileNetV3Large con configuración específica.

        Returns:
            Modelo Keras compilado
        """
        logger.info("Construyendo modelo MobileNetV3Large...")

        # Cargar modelo base
        base_model = MobileNetV3Large(
            input_shape=self.config['model']['input_shape'],
            alpha=self.config['model']['alpha'],
            minimalistic=self.config['model']['minimalistic'],
            include_top=self.config['model']['include_top'],
            weights=self.config['model']['weights'],
            pooling=self.config['model']['pooling']
        )

        # Congelar capas base si es necesario
        if not self.config['model']['include_top']:
            # Añadir capas de clasificación personalizadas
            x = base_model.output
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dense(1024, activation='relu')(x)
            x = layers.Dropout(0.5)(x)
            predictions = layers.Dense(
                len(self.config['data']['classes']),
                activation='softmax'
            )(x)

            self.model = models.Model(inputs=base_model.input, outputs=predictions)

            # Congelar capas del modelo base
            for layer in base_model.layers:
                layer.trainable = False
        else:
            self.model = base_model

        # Compilar modelo
        self.model.compile(
            optimizer=self.config['training']['optimizer'],
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        logger.info(f"Modelo construido con {self.model.count_params()} parámetros")
        return self.model

    def apply_pruning(self) -> tf.keras.Model:
        """
        Aplica poda (pruning) al modelo.

        Returns:
            Modelo con poda aplicada
        """
        if not self.config['pruning']['enabled'] or not PRUNING_AVAILABLE:
            if not PRUNING_AVAILABLE:
                logger.warning("tensorflow-model-optimization no disponible, omitiendo pruning")
            else:
                logger.info("Poda deshabilitada, retornando modelo original")
            return self.model

        logger.info("Aplicando poda (pruning) al modelo...")

        # Definir schedule de poda
        pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=self.config['pruning']['initial_sparsity'],
            final_sparsity=self.config['pruning']['final_sparsity'],
            begin_step=self.config['pruning']['begin_step'],
            end_step=self.config['pruning']['end_step'],
            frequency=self.config['pruning']['frequency']
        )

        # Aplicar poda
        self.pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
            self.model,
            pruning_schedule=pruning_schedule
        )

        # Compilar modelo podado
        self.pruned_model.compile(
            optimizer=self.config['training']['optimizer'],
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        logger.info("Poda aplicada exitosamente")
        return self.pruned_model

    def create_representative_dataset(self, data_path: str, num_samples: int = 100):
        """
        Crea dataset representativo para cuantización.

        Args:
            data_path: Ruta al directorio de datos
            num_samples: Número de muestras

        Returns:
            Función generadora para dataset representativo
        """
        def representative_dataset():
            # Buscar imágenes de muestra
            image_paths = []
            data_dir = Path(data_path)

            if data_dir.exists():
                for class_dir in data_dir.iterdir():
                    if class_dir.is_dir():
                        image_files = list(class_dir.glob('*.jpg')) + \
                                    list(class_dir.glob('*.jpeg')) + \
                                    list(class_dir.glob('*.png'))
                        image_paths.extend(image_files)

            # Limitar número de muestras
            image_paths = image_paths[:num_samples]

            for img_path in image_paths:
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

                # Añadir dimensión de batch
                img_array = np.expand_dims(img_array, axis=0)

                yield [img_array.astype(np.float32)]

        return representative_dataset

    def convert_to_tflite(self, output_path: str, data_path: str = "data/train") -> str:
        """
        Convierte el modelo a TensorFlow Lite con optimizaciones.

        Args:
            output_path: Ruta de salida para el archivo .tflite
            data_path: Ruta a datos para calibración de cuantización

        Returns:
            Ruta al archivo TFLite generado
        """
        logger.info("Convirtiendo modelo a TensorFlow Lite...")

        # Usar modelo podado si existe, sino el modelo original
        model_to_convert = self.pruned_model if self.pruned_model else self.model

        if model_to_convert is None:
            raise ValueError("No hay modelo para convertir. Use build_model() primero.")

        # Aplicar strip pruning si el modelo fue podado
        if self.pruned_model:
            logger.info("Aplicando strip pruning...")
            model_to_convert = tfmot.sparsity.keras.strip_pruning(model_to_convert)

        # Crear convertidor TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model_to_convert)

        if self.config['quantization']['enabled']:
            logger.info("Aplicando cuantización INT8...")

            # Configurar optimizaciones
            converter.optimizations = self.config['quantization']['optimization']

            # Configurar tipos de entrada/salida
            input_type = getattr(tf, self.config['quantization']['inference_input_type'].split('.')[-1])
            output_type = getattr(tf, self.config['quantization']['inference_output_type'].split('.')[-1])

            converter.inference_input_type = input_type
            converter.inference_output_type = output_type

            # Configurar operaciones soportadas
            converter.target_spec.supported_ops = [
                getattr(tf.lite.OpsSet, op)
                for op in self.config['quantization']['supported_ops']
            ]

            # Dataset representativo para calibración
            representative_dataset = self.create_representative_dataset(
                data_path,
                self.config['quantization']['representative_dataset_samples']
            )
            converter.representative_dataset = representative_dataset

        # Convertir modelo
        tflite_model = converter.convert()

        # Guardar modelo TFLite
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        # Calcular tamaño del modelo
        model_size_mb = len(tflite_model) / (1024 * 1024)

        logger.info(f"Modelo TFLite guardado en: {output_path}")
        logger.info(f"Tamaño del modelo: {model_size_mb:.2f} MB")
        return str(output_path)

    def get_model_info(self) -> dict:
        """
        Obtiene información detallada del modelo.

        Returns:
            Diccionario con información del modelo
        """
        info = {}

        if self.model:
            info['original_params'] = self.model.count_params()

        if self.pruned_model:
            info['pruned_params'] = self.pruned_model.count_params()

        return info

def main():
    """Función principal para conversión TFLite."""
    import argparse

    parser = argparse.ArgumentParser(description='MobileNetV3Large TFLite Converter')
    parser.add_argument('--config', default='config.yaml', help='Ruta al archivo de configuración')
    parser.add_argument('--output', default='mobilenetv3_large.tflite', help='Ruta de salida del modelo TFLite')
    parser.add_argument('--data-path', default='data/train', help='Ruta a datos para calibración')

    args = parser.parse_args()

    try:
        # Inicializar convertidor
        converter = MobileNetV3TFLiteConverter(args.config)

        # Construir modelo base
        converter.build_model()

        # Aplicar poda si está habilitada
        if converter.config['pruning']['enabled']:
            converter.apply_pruning()

        # Convertir a TFLite con optimizaciones
        tflite_path = converter.convert_to_tflite(args.output, args.data_path)

        # Mostrar información del modelo
        info = converter.get_model_info()
        logger.info("Información del modelo:")
        for key, value in info.items():
            logger.info(f"  {key}: {value}")

        logger.info("Conversión completada exitosamente!")
        logger.info(f"Modelo optimizado guardado en: {tflite_path}")

    except Exception as e:
        logger.error(f"Error durante conversión: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
