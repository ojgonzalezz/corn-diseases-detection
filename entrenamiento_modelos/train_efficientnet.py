"""
Entrenamiento de EfficientNet-Lite para clasificación de enfermedades del maíz
"""

import os
import time
import mlflow
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from datetime import datetime

# Importar configuración y utilidades
from config import *
from utils import *

def create_efficientnet_model(num_classes, image_size, learning_rate):
    """Crear modelo EfficientNet-Lite (usando EfficientNetB0 como base)"""

    # Cargar base preentrenada
    base_model = EfficientNetB0(
        input_shape=(*image_size, 3),
        include_top=False,
        weights='imagenet'
    )

    # Congelar capas base inicialmente
    base_model.trainable = False

    # Construir modelo completo
    inputs = tf.keras.Input(shape=(*image_size, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)

    # Compilar
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_efficientnet():
    """Función principal de entrenamiento"""

    print("="*60)
    print("ENTRENAMIENTO EFFICIENTNET-LITE")
    print("="*60)

    # Configurar GPU
    setup_gpu(GPU_MEMORY_LIMIT)

    # Configurar MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # Crear generadores de datos
    print("\nCreando generadores de datos...")
    train_gen, val_gen, test_gen = create_data_generators(
        DATA_DIR,
        IMAGE_SIZE,
        BATCH_SIZE,
        TRAIN_SPLIT,
        VAL_SPLIT,
        TEST_SPLIT,
        RANDOM_SEED,
        DATA_AUGMENTATION
    )

    print(f"Imagenes de entrenamiento: {train_gen.samples}")
    print(f"Imagenes de validacion: {val_gen.samples}")
    print(f"Imagenes de prueba: {test_gen.samples}")

    # Crear modelo
    print("\nCreando modelo EfficientNet-Lite...")
    model = create_efficientnet_model(NUM_CLASSES, IMAGE_SIZE, LEARNING_RATE)
    print(f"Total de parametros: {model.count_params():,}")

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=REDUCE_LR_PATIENCE,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            str(MODELS_DIR / 'efficientnet_best.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    # Hiperparámetros
    hyperparameters = {
        'model_name': 'EfficientNet-Lite (B0)',
        'image_size': IMAGE_SIZE,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'learning_rate': LEARNING_RATE,
        'optimizer': 'Adam',
        'train_split': TRAIN_SPLIT,
        'val_split': VAL_SPLIT,
        'test_split': TEST_SPLIT,
        'early_stopping_patience': EARLY_STOPPING_PATIENCE,
        'reduce_lr_patience': REDUCE_LR_PATIENCE,
        'data_augmentation': DATA_AUGMENTATION
    }

    # Iniciar MLflow run
    with mlflow.start_run(run_name=f"EfficientNet-Lite_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):

        # Log hiperparámetros
        mlflow.log_params(hyperparameters)

        # Entrenar modelo
        print("\nIniciando entrenamiento...")
        start_time = time.time()

        history = model.fit(
            train_gen,
            epochs=EPOCHS,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )

        training_time = time.time() - start_time
        print(f"\nTiempo de entrenamiento: {training_time:.2f} segundos ({training_time/60:.2f} minutos)")

        # Fine-tuning (descongelar últimas capas)
        print("\nIniciando fine-tuning...")
        # Descongelar últimas 20 capas
        for layer in model.layers[1].layers[-20:]:
            layer.trainable = True

        # Re-compilar con learning rate más bajo
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE * 0.1),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Entrenar con fine-tuning
        history_finetune = model.fit(
            train_gen,
            epochs=20,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )

        # Combinar historiales
        for key in history.history:
            history.history[key].extend(history_finetune.history[key])

        finetune_time = time.time() - start_time - training_time
        total_time = time.time() - start_time

        print(f"Tiempo de fine-tuning: {finetune_time:.2f} segundos")
        print(f"Tiempo total: {total_time:.2f} segundos ({total_time/60:.2f} minutos)")

        # Evaluar modelo
        print("\nEvaluando modelo en test set...")
        evaluation_results = evaluate_model(model, test_gen, CLASSES)

        # Graficar historial
        plot_path = LOGS_DIR / 'efficientnet_training_history.png'
        plot_training_history(history, plot_path)
        mlflow.log_artifact(str(plot_path))

        # Matriz de confusión
        cm_path = LOGS_DIR / 'efficientnet_confusion_matrix.png'
        cm = plot_confusion_matrix(
            evaluation_results['y_true'],
            evaluation_results['y_pred'],
            CLASSES,
            cm_path
        )
        mlflow.log_artifact(str(cm_path))

        # Guardar log
        log_path = LOGS_DIR / 'efficientnet_training_log.json'
        save_training_log(
            log_path,
            'EfficientNet-Lite (B0)',
            hyperparameters,
            history,
            evaluation_results,
            cm,
            total_time
        )
        mlflow.log_artifact(str(log_path))
        mlflow.log_artifact(str(log_path).replace('.json', '.txt'))

        # Log métricas en MLflow
        mlflow.log_metric("train_accuracy", history.history['accuracy'][-1])
        mlflow.log_metric("val_accuracy", history.history['val_accuracy'][-1])
        mlflow.log_metric("test_accuracy", evaluation_results['test_accuracy'])
        mlflow.log_metric("test_loss", evaluation_results['test_loss'])
        mlflow.log_metric("training_time", total_time)

        # Guardar modelo
        model_path = MODELS_DIR / 'efficientnet_final.keras'
        model.save(str(model_path))
        mlflow.log_artifact(str(model_path))

        print(f"\nModelo guardado en: {model_path}")
        print(f"Log guardado en: {log_path}")
        print(f"\nTest Accuracy: {evaluation_results['test_accuracy']:.4f}")

    print("\n" + "="*60)
    print("ENTRENAMIENTO COMPLETADO")
    print("="*60)

if __name__ == "__main__":
    train_efficientnet()
