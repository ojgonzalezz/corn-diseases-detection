#!/usr/bin/env python3

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

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

try:
    import tensorflow_model_optimization as tfmot
    PRUNING_AVAILABLE = True
except ImportError:
    PRUNING_AVAILABLE = False
    tfmot = None

class MobileNetV3TFLiteConverter:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.model = None
        self.pruned_model = None

    def _load_config(self, config_path: str) -> dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def build_model(self) -> tf.keras.Model:
        base_model = MobileNetV3Large(
            input_shape=self.config['model']['input_shape'],
            alpha=self.config['model']['alpha'],
            minimalistic=self.config['model']['minimalistic'],
            include_top=False,  # Always use False to add custom classification head
            weights=self.config['model']['weights'],
            pooling=self.config['model']['pooling']
        )

        # Add custom classification head for corn diseases
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        predictions = layers.Dense(len(self.config['data']['classes']), activation='softmax')(x)
        self.model = models.Model(inputs=base_model.input, outputs=predictions)

        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False

        # Compilar modelo
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
        """Simple training to adapt the model to corn disease classification."""
        try:
            from tensorflow.keras.preprocessing.image import ImageDataGenerator

            train_datagen = ImageDataGenerator(
                preprocessing_function=mobilenet_v3_preprocess,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                validation_split=0.2
            )

            train_generator = train_datagen.flow_from_directory(
                data_path,
                target_size=tuple(self.config['model']['input_shape'][:2]),
                batch_size=32,
                class_mode='categorical',
                subset='training'
            )

            val_generator = train_datagen.flow_from_directory(
                data_path,
                target_size=tuple(self.config['model']['input_shape'][:2]),
                batch_size=32,
                class_mode='categorical',
                subset='validation'
            )

            logger.info(f"Training with {train_generator.samples} training samples, {val_generator.samples} validation samples")

            # Fine-tune only the classification head
            self.model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=epochs,
                verbose=1
            )

            logger.info("Model training completed")

        except Exception as e:
            logger.warning(f"Training failed: {e}. Continuing with pre-trained weights.")

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
