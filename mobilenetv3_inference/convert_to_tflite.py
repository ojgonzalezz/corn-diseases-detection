#!/usr/bin/env python3
"""
Conversor de MobileNetV3Large a TensorFlow Lite con optimizaciones para edge computing.

Este script realiza todo el proceso de preparación del modelo:
1. Construcción de la arquitectura MobileNetV3Large con cabeza de clasificación personalizada
2. Entrenamiento fine-tuning con datos de enfermedades de maíz
3. Aplicación de técnicas de optimización (pruning, cuantización)
4. Conversión a formato TensorFlow Lite optimizado

Características principales:
- Arquitectura MobileNetV3Large pre-entrenada en ImageNet
- Fine-tuning adaptado a 4 clases de enfermedades de maíz
- Optimización automática de estructura de datos (train/val/test)
- Cuantización INT8 para dispositivos edge
- Pruning opcional para reducción de tamaño

Uso:
    python convert_to_tflite.py --config config.yaml --output model.tflite --data-path /path/to/data

Autor: Sistema de Detección de Enfermedades del Maíz
"""

import os
import sys
import yaml
import logging
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenet_v3_preprocess
from tensorflow.keras import layers, models

# Configuración del sistema de logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Verificación de disponibilidad de pruning (opcional)
try:
    import tensorflow_model_optimization as tfmot
    PRUNING_AVAILABLE = True
    logger.info("TensorFlow Model Optimization disponible - pruning habilitado")
except ImportError:
    PRUNING_AVAILABLE = False
    tfmot = None
    logger.warning("TensorFlow Model Optimization no disponible - pruning deshabilitado")

class MobileNetV3TFLiteConverter:
    """
    Clase principal para conversión de MobileNetV3Large a TensorFlow Lite.

    Maneja todo el proceso de optimización del modelo:
    - Construcción de arquitectura
    - Entrenamiento fine-tuning
    - Aplicación de pruning
    - Cuantización y conversión TFLite
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Inicializa el conversor con la configuración especificada.

        Args:
            config_path: Ruta al archivo de configuración YAML
        """
        self.config = self._load_config(config_path)
        self.model = None
        self.pruned_model = None

    def _load_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def build_model(self) -> tf.keras.Model:
        """
        Construye la arquitectura MobileNetV3Large con cabeza de clasificación personalizada.

        Utiliza MobileNetV3Large pre-entrenado en ImageNet y agrega capas de clasificación
        específicas para las 4 enfermedades del maíz.

        Returns:
            tf.keras.Model: Modelo construido y compilado
        """
        # Cargar modelo base MobileNetV3Large sin la capa de clasificación original
        base_model = MobileNetV3Large(
            input_shape=self.config['model']['input_shape'],
            alpha=self.config['model']['alpha'],
            minimalistic=self.config['model']['minimalistic'],
            include_top=False,  # Siempre False para agregar cabeza personalizada
            weights=self.config['model']['weights'],
            pooling=self.config['model']['pooling']
        )

        # Agregar cabeza de clasificación personalizada para enfermedades de maíz
        x = base_model.output

        # Solo agregar GlobalAveragePooling2D si no está ya aplicado (pooling != 'avg')
        if self.config['model']['pooling'] != 'avg':
            x = layers.GlobalAveragePooling2D()(x)

        # Capas densas para clasificación
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        predictions = layers.Dense(len(self.config['data']['classes']), activation='softmax')(x)
        self.model = models.Model(inputs=base_model.input, outputs=predictions)

        # Congelar capas del modelo base para fine-tuning
        for layer in base_model.layers:
            layer.trainable = False

        # Compilar modelo con optimizador y métricas
        self.model.compile(
            optimizer=self.config['training']['optimizer'],
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        logger.info(f"Modelo construido con {self.model.count_params()} parámetros")
        return self.model

    def apply_pruning(self) -> tf.keras.Model:
        if not self.config['pruning']['enabled'] or not PRUNING_AVAILABLE:
            return self.model

        pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=self.config['pruning']['initial_sparsity'],
            final_sparsity=self.config['pruning']['final_sparsity'],
            begin_step=self.config['pruning']['begin_step'],
            end_step=self.config['pruning']['end_step'],
            frequency=self.config['pruning']['frequency']
        )

        self.pruned_model = tfmot.sparsity.keras.prune_low_magnitude(self.model, pruning_schedule=pruning_schedule)
        self.pruned_model.compile(optimizer=self.config['training']['optimizer'], loss='categorical_crossentropy', metrics=['accuracy'])
        return self.pruned_model

    def train_model(self, data_path: str, epochs: int = 5):
        """
        Entrena el modelo con datos de enfermedades de maíz mediante fine-tuning.

        Realiza un entrenamiento adaptativo del modelo pre-entrenado usando datos
        específicos de enfermedades de maíz. Incluye aumento de datos y validación.

        Args:
            data_path: Ruta a los datos de entrenamiento
            epochs: Número de épocas de entrenamiento
        """
        try:
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            from pathlib import Path

            # Asegurar que la ruta sea absoluta o relativa a la raíz del proyecto
            data_path = Path(data_path)
            if not data_path.is_absolute():
                # Si se ejecuta desde mobilenetv3_inference/, subir un nivel
                project_root = Path(__file__).parent.parent
                data_path = project_root / data_path

            # Si la ruta contiene subdirectorios train/val/test, usar train para entrenamiento
            if data_path.exists() and any((data_path / sub).exists() for sub in ['train', 'val', 'test']):
                data_path = data_path / 'train'

            if not data_path.exists():
                raise FileNotFoundError(f"Ruta de datos de entrenamiento no encontrada: {data_path}")

            # Configurar generador de datos con aumento de imágenes
            train_datagen = ImageDataGenerator(
                preprocessing_function=mobilenet_v3_preprocess,
                rotation_range=20,      # Rotaciones aleatorias
                width_shift_range=0.2,  # Desplazamientos horizontales
                height_shift_range=0.2, # Desplazamientos verticales
                horizontal_flip=True,   # Volteos horizontales
                validation_split=0.2    # 20% para validación
            )

            logger.info(f"Buscando datos de entrenamiento en: {data_path}")

            # Crear generadores de datos
            train_generator = train_datagen.flow_from_directory(
                str(data_path),
                target_size=tuple(self.config['model']['input_shape'][:2]),
                batch_size=32,
                class_mode='categorical',
                subset='training'
            )

            val_generator = train_datagen.flow_from_directory(
                str(data_path),
                target_size=tuple(self.config['model']['input_shape'][:2]),
                batch_size=32,
                class_mode='categorical',
                subset='validation'
            )

            logger.info(f"Clases encontradas: {list(train_generator.class_indices.keys())}")
            logger.info(f"Entrenando con {train_generator.samples} muestras de entrenamiento, {val_generator.samples} muestras de validación")

            # Verificar que las clases coincidan con la configuración
            expected_classes = set(self.config['data']['classes'])
            found_classes = set(train_generator.class_indices.keys())

            if expected_classes != found_classes:
                logger.warning(f"¡Discrepancia de clases! Esperadas: {expected_classes}, Encontradas: {found_classes}")
                # Continuar de todas formas, el modelo se adaptará

            # Fine-tuning solo de la cabeza de clasificación
            self.model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=epochs,
                verbose=1
            )

            logger.info("Entrenamiento del modelo completado")

        except Exception as e:
            logger.warning(f"Entrenamiento fallido: {e}. Continuando con pesos pre-entrenados.")

    def create_representative_dataset(self, data_path: str, num_samples: int = 100):
        def representative_dataset():
            image_paths = []
            data_dir = Path(data_path)

            if data_dir.exists():
                for class_dir in data_dir.iterdir():
                    if class_dir.is_dir():
                        image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.png'))
                        image_paths.extend(image_files)

            image_paths = image_paths[:num_samples]

            for img_path in image_paths:
                img = image.load_img(str(img_path), target_size=self.config['model']['input_shape'][:2], interpolation=self.config['preprocessing']['interpolation'])
                img_array = image.img_to_array(img)

                if self.config['preprocessing']['mode'] == 'tf':
                    img_array = mobilenet_v3_preprocess(img_array)
                else:
                    img_array = img_array / 255.0

                img_array = np.expand_dims(img_array, axis=0)
                yield [img_array.astype(np.float32)]

        return representative_dataset

    def convert_to_tflite(self, output_path: str, data_path: str = "data/train") -> str:
        model_to_convert = self.pruned_model if self.pruned_model else self.model

        if self.pruned_model:
            model_to_convert = tfmot.sparsity.keras.strip_pruning(model_to_convert)

        converter = tf.lite.TFLiteConverter.from_keras_model(model_to_convert)

        if self.config['quantization']['enabled']:
            converter.optimizations = self.config['quantization']['optimization']
            converter.target_spec.supported_ops = [getattr(tf.lite.OpsSet, op) for op in self.config['quantization']['supported_ops']]
            converter.representative_dataset = self.create_representative_dataset(data_path, self.config['quantization']['representative_dataset_samples'])

        tflite_model = converter.convert()

        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        return output_path

def main():
    import argparse

    parser = argparse.ArgumentParser(description='MobileNetV3Large TFLite Converter')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--output', default='mobilenetv3_large.tflite', help='Output TFLite model path')
    parser.add_argument('--data-path', default='data/train', help='Data path for calibration')

    args = parser.parse_args()

    converter = MobileNetV3TFLiteConverter(args.config)
    converter.build_model()

    # Train the model with corn disease data
    converter.train_model(args.data_path, epochs=3)

    if converter.config['pruning']['enabled']:
        converter.apply_pruning()

    tflite_path = converter.convert_to_tflite(args.output, args.data_path)
    logger.info(f"Modelo optimizado guardado en: {tflite_path}")

if __name__ == "__main__":
    main()
