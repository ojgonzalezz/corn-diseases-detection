"""
Script de entrenamiento para modelos edge computing.

Este script entrena y evalúa arquitecturas livianas optimizadas para
dispositivos de borde (edge computing).

Arquitecturas soportadas:
- MobileNetV3Small
- MobileNetV3Large  
- EfficientNet-Lite (B0, B1, B2)
- MobileViT
- PMVT (Plant-based MobileViT)

Requisitos mínimos:
- Precisión global ≥ 85%
- Recall por clase ≥ 0.80
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import mlflow
import mlflow.keras
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, recall_score

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.config import config
from src.utils.paths import paths
from src.utils.logger import get_logger, log_section, log_dict
from src.pipelines.preprocess import split_and_balance_dataset_efficient
from src.utils.utils import flatten_data, create_efficient_dataset_from_dict
from src.builders.base_models import (
    load_mobilenetv3_large,
    load_efficientnet_lite_b2,
    load_mobilevit,
    load_pmvt,
    get_model_info
)

# Logger
logger = get_logger(__name__)


def build_classification_head(backbone: Model, num_classes: int, dropout_rate: float = 0.3) -> Model:
    """
    Construye cabeza de clasificación simple para modelos edge.
    
    Args:
        backbone: Modelo base preentrenado.
        num_classes: Número de clases.
        dropout_rate: Tasa de dropout.
        
    Returns:
        Modelo completo con cabeza de clasificación.
    """
    # Congelar backbone
    backbone.trainable = False
    
    # Detectar si el backbone ya tiene pooling (output shape 2D)
    output_shape = backbone.output_shape
    needs_pooling = len(output_shape) == 4  # (None, H, W, C) necesita pooling
    
    # Construir cabeza según la salida del backbone
    if needs_pooling:
        # Backbone retorna 4D (None, H, W, C) - necesita GlobalAveragePooling2D
        model = models.Sequential([
            backbone,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(dropout_rate),
            layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.Dropout(dropout_rate / 2),
            layers.Dense(num_classes, activation='softmax')
        ], name=f'{backbone.name}_classifier')
    else:
        # Backbone ya retorna 2D (None, features) - no necesita pooling
        model = models.Sequential([
            backbone,
            layers.Dropout(dropout_rate),
            layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            layers.Dropout(dropout_rate / 2),
            layers.Dense(num_classes, activation='softmax')
        ], name=f'{backbone.name}_classifier')
    
    return model


def train_edge_model(
    model_name: str,
    learning_rate: float = 0.001,
    dropout_rate: float = 0.3,
    epochs: int = 30,
    batch_size: int = 32,
    fine_tune: bool = False,
    fine_tune_epochs: int = 10
):
    """
    Entrena un modelo edge con hiperparámetros específicos.
    
    Args:
        model_name: Nombre del modelo ('MobileNetV3Small', 'MobileNetV3Large', 
                    'EfficientNetLite0', 'EfficientNetLite1', 'EfficientNetLite2',
                    'MobileViT', 'PMVT')
        learning_rate: Learning rate inicial.
        dropout_rate: Tasa de dropout.
        epochs: Número de épocas.
        batch_size: Tamaño del batch.
        fine_tune: Si True, hace fine-tuning del backbone.
        fine_tune_epochs: Épocas adicionales para fine-tuning.
        
    Returns:
        Dict con resultados del entrenamiento.
    """
    log_section(logger, f"ENTRENAMIENTO: {model_name}")
    
    # Configurar MLflow
    mlflow.set_tracking_uri(f"file:///{paths.mlruns.as_posix()}")
    mlflow.set_experiment("edge_models_comparison")
    
    # Mapeo de nombres a loaders (4 arquitecturas seleccionadas)
    model_loaders = {
        'MobileNetV3Large': lambda: load_mobilenetv3_large(),
        'EfficientNetLiteB2': lambda: load_efficientnet_lite_b2(),
        'MobileViT': lambda: load_mobilevit(variant='small'),
        'PMVT': lambda: load_pmvt()
    }
    
    if model_name not in model_loaders:
        raise ValueError(f"Modelo '{model_name}' no soportado. Opciones: {list(model_loaders.keys())}")
    
    # Cargar datos de manera eficiente
    logger.info("Cargando datos eficientemente...")
    raw_dataset, label_to_int = split_and_balance_dataset_efficient(
        balanced='oversample',
        split_ratios=(0.7, 0.15, 0.15)
    )

    IMAGE_SIZE = config.data.image_size
    NUM_CLASSES = config.data.num_classes
    CLASS_NAMES = list(label_to_int.keys())

    # Crear datasets eficientes con tf.data
    train_dataset, _ = create_efficient_dataset_from_dict(
        raw_dataset['train'],
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=True,
        augment=True  # Aumentación solo para entrenamiento
    )
    val_dataset, _ = create_efficient_dataset_from_dict(
        raw_dataset['val'],
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=False,
        augment=False
    )
    test_dataset, _ = create_efficient_dataset_from_dict(
        raw_dataset['test'],
        image_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=False,
        augment=False
    )

    # Calcular steps por epoch
    train_steps = sum(len(paths) for paths in raw_dataset['train'].values()) // batch_size
    val_steps = sum(len(paths) for paths in raw_dataset['val'].values()) // batch_size
    test_steps = sum(len(paths) for paths in raw_dataset['test'].values()) // batch_size

    logger.info(f"Datos cargados: Train={sum(len(paths) for paths in raw_dataset['train'].values())}, "
                f"Val={sum(len(paths) for paths in raw_dataset['val'].values())}, "
                f"Test={sum(len(paths) for paths in raw_dataset['test'].values())}")
    logger.info(f"Steps por epoch: Train={train_steps}, Val={val_steps}, Test={test_steps}")
    
    # Iniciar MLflow run
    with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Log hiperparámetros
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("dropout_rate", dropout_rate)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("fine_tune", fine_tune)
        mlflow.log_param("image_size", IMAGE_SIZE)
        
        # Cargar backbone
        logger.info(f"Cargando backbone: {model_name}")
        backbone = model_loaders[model_name]()
        
        # Info del modelo
        model_info = get_model_info(backbone)
        log_dict(logger, model_info, f"Información del Backbone {model_name}")
        
        mlflow.log_param("backbone_params", model_info['total_params'])
        mlflow.log_param("backbone_size_mb", round(model_info['size_mb'], 2))
        
        # Construir modelo completo
        model = build_classification_head(backbone, NUM_CLASSES, dropout_rate)
        
        # Compilar
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("Modelo compilado")
        model.summary(print_fn=logger.info)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Fase 1: Entrenar solo cabeza
        logger.info("=" * 70)
        logger.info("FASE 1: Entrenamiento de cabeza de clasificación")
        logger.info("=" * 70)

        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            steps_per_epoch=train_steps,
            validation_steps=val_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        # Log métricas de entrenamiento
        for epoch in range(len(history.history['loss'])):
            mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
            mlflow.log_metric("train_accuracy", history.history['accuracy'][epoch], step=epoch)
            mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
            mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)
        
        # Fase 2: Fine-tuning (opcional)
        if fine_tune:
            logger.info("=" * 70)
            logger.info("FASE 2: Fine-tuning del backbone")
            logger.info("=" * 70)
            
            # Descongelar las últimas capas del backbone
            backbone.trainable = True
            
            # Re-compilar con learning rate más bajo
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate / 10),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            history_ft = model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=fine_tune_epochs,
                steps_per_epoch=train_steps,
                validation_steps=val_steps,
                callbacks=callbacks,
                verbose=1,
                initial_epoch=len(history.history['loss'])
            )
            
            # Log métricas de fine-tuning
            for epoch in range(len(history_ft.history['loss'])):
                step = len(history.history['loss']) + epoch
                mlflow.log_metric("train_loss", history_ft.history['loss'][epoch], step=step)
                mlflow.log_metric("train_accuracy", history_ft.history['accuracy'][epoch], step=step)
                mlflow.log_metric("val_loss", history_ft.history['val_loss'][epoch], step=step)
                mlflow.log_metric("val_accuracy", history_ft.history['val_accuracy'][epoch], step=step)
        
        # Evaluación en test set
        logger.info("=" * 70)
        logger.info("EVALUACIÓN EN TEST SET")
        logger.info("=" * 70)

        test_loss, test_accuracy = model.evaluate(test_dataset, steps=test_steps, verbose=1)

        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

        # Predicciones - necesitamos obtener las etiquetas reales y predicciones
        # Para esto necesitamos un enfoque diferente con tf.data
        y_true = []
        y_pred = []

        for images, labels in test_dataset.take(test_steps):
            batch_pred = model.predict(images, verbose=0)
            y_pred.extend(np.argmax(batch_pred, axis=1))
            y_true.extend(np.argmax(labels.numpy(), axis=1))

        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        
        # Calcular métricas por clase
        report = classification_report(
            y_true,
            y_pred,
            target_names=CLASS_NAMES,
            output_dict=True
        )
        
        # Recall por clase
        recalls = {}
        for class_name in CLASS_NAMES:
            recall = report[class_name]['recall']
            recalls[class_name] = recall
            logger.info(f"Recall {class_name}: {recall:.4f}")
            mlflow.log_metric(f"recall_{class_name}", recall)
        
        # Verificar requisitos mínimos
        min_recall = min(recalls.values())
        meets_requirements = test_accuracy >= 0.85 and min_recall >= 0.80
        
        logger.info("=" * 70)
        if meets_requirements:
            logger.info(" CUMPLE REQUISITOS MÍNIMOS PARA EDGE COMPUTING")
        else:
            logger.info(" NO CUMPLE REQUISITOS MÍNIMOS")
            if test_accuracy < 0.85:
                logger.info(f"  - Precisión: {test_accuracy:.2%} < 85%")
            if min_recall < 0.80:
                logger.info(f"  - Recall mínimo: {min_recall:.2%} < 80%")
        logger.info("=" * 70)
        
        # Log métricas finales
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("min_recall", min_recall)
        mlflow.log_metric("meets_requirements", 1.0 if meets_requirements else 0.0)
        
        # Guardar modelo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = paths.models_exported / f"{model_name}_{timestamp}_acc{test_accuracy:.4f}.keras"
        model.save(model_path)
        logger.info(f"Modelo guardado en: {model_path}")
        
        # Log modelo en MLflow
        mlflow.keras.log_model(model, f"{model_name}_model")
        
        # Guardar metadata
        metadata = {
            'model_name': model_name,
            'timestamp': timestamp,
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss),
            'min_recall': float(min_recall),
            'recalls': {k: float(v) for k, v in recalls.items()},
            'meets_requirements': meets_requirements,
            'hyperparameters': {
                'learning_rate': learning_rate,
                'dropout_rate': dropout_rate,
                'epochs': epochs,
                'batch_size': batch_size,
                'fine_tune': fine_tune
            },
            'model_info': {
                'total_params': int(model_info['total_params']),
                'size_mb': float(model_info['size_mb'])
            }
        }
        
        metadata_path = paths.models_exported / f"{model_name}_{timestamp}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        mlflow.log_artifact(str(metadata_path))
        
        logger.info(f"Metadata guardada en: {metadata_path}")
        
        return metadata


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Entrenar modelo edge para clasificación de enfermedades del maíz')
    parser.add_argument('--model', type=str, required=True,
                        choices=['MobileNetV3Large', 'EfficientNetLiteB2', 'MobileViT', 'PMVT'],
                        help='Arquitectura a entrenar (4 modelos seleccionados)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=30, help='Número de épocas')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--fine-tune', action='store_true', help='Hacer fine-tuning del backbone')
    parser.add_argument('--fine-tune-epochs', type=int, default=10, help='Épocas de fine-tuning')
    
    args = parser.parse_args()
    
    result = train_edge_model(
        model_name=args.model,
        learning_rate=args.lr,
        dropout_rate=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        fine_tune=args.fine_tune,
        fine_tune_epochs=args.fine_tune_epochs
    )
    
    logger.info("=" * 70)
    logger.info("RESULTADO FINAL")
    logger.info("=" * 70)
    log_dict(logger, result, "Metadata del Modelo")

