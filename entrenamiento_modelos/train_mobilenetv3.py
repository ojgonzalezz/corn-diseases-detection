"""
Entrenamiento de MobileNetV3-Large para clasificación de enfermedades del maíz
"""

import os
import time
import mlflow
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from datetime import datetime

# Importar configuración y utilidades
from config import *
from utils import *

def create_mobilenetv3_model(num_classes, image_size, learning_rate):
    """Crear modelo MobileNetV3-Large"""

    # Cargar base preentrenada
    base_model = MobileNetV3Large(
        input_shape=(*image_size, 3),
        include_top=False,
        weights='imagenet'
    )

    # Congelar capas base inicialmente
    base_model.trainable = False

    # Construir modelo completo - Arquitectura 10/10 optimizada
    # Balance perfecto entre capacidad y generalización
    inputs = tf.keras.Input(shape=(*image_size, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)

    # Arquitectura balanceada: 2 capas densas (256→128)
    # Dropout más alto para mejor regularización
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = Dropout(0.35)(x)  # Aumentado de 0.3 a 0.35 para mejor generalización

    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = Dropout(0.3)(x)  # Aumentado de 0.25 a 0.3

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)

    # Compilar
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_mobilenetv3():
    """Función principal de entrenamiento"""

    print("="*60)
    print("ENTRENAMIENTO MOBILENETV3-LARGE")
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

    # Calcular class weights para balanceo de clases (mejora recall)
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"\nClass weights calculados: {class_weight_dict}")

    # Crear modelo
    print("\nCreando modelo MobileNetV3-Large...")
    model = create_mobilenetv3_model(NUM_CLASSES, IMAGE_SIZE, LEARNING_RATE)
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
            str(MODELS_DIR / 'mobilenetv3_best.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    # Hiperparámetros
    hyperparameters = {
        'model_name': 'MobileNetV3-Large',
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
    with mlflow.start_run(run_name=f"MobileNetV3-Large_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):

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
            class_weight=class_weight_dict,  # Usar class weights para mejor recall
            verbose=1
        )

        training_time = time.time() - start_time
        print(f"\nTiempo de entrenamiento: {training_time:.2f} segundos ({training_time/60:.2f} minutos)")

        # Fine-tuning (descongelar últimas capas gradualmente)
        print("\nIniciando fine-tuning...")

        # Descongelar solo las últimas 20 capas del modelo base
        # Balance perfecto para evitar overfitting y permitir adaptación
        base_model = model.layers[1]
        base_model.trainable = True

        # Congelar todas las capas excepto las últimas 20
        for layer in base_model.layers[:-20]:
            layer.trainable = False

        trainable_layers = sum([1 for layer in base_model.layers if layer.trainable])
        print(f"Capas descongeladas para fine-tuning: {trainable_layers} de {len(base_model.layers)}")

        # Re-compilar con learning rate bajo pero no demasiado bajo para fine-tuning
        # Usar 0.00005 (0.001 * 0.05) para permitir ajuste fino sin sobreajuste
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE * 0.05),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Callbacks para fine-tuning - Balance entre exploración y convergencia
        finetune_callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=8,  # Aumentado de 5 a 8 para dar más tiempo
                restore_best_weights=True,
                verbose=1,
                mode='max'
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,  # Aumentado de 3 a 4
                min_lr=1e-7,
                verbose=1,
                mode='min'
            ),
            ModelCheckpoint(
                str(MODELS_DIR / 'mobilenetv3_best.keras'),
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1,
                mode='max'
            )
        ]

        # Entrenar con fine-tuning
        history_finetune = model.fit(
            train_gen,
            epochs=20,  # Aumentado de 15 a 20 para mejor convergencia
            validation_data=val_gen,
            callbacks=finetune_callbacks,
            class_weight=class_weight_dict,  # Usar class weights también en fine-tuning
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
        plot_path = LOGS_DIR / 'mobilenetv3_training_history.png'
        plot_training_history(history, plot_path)
        mlflow.log_artifact(str(plot_path))

        # Matriz de confusión
        cm_path = LOGS_DIR / 'mobilenetv3_confusion_matrix.png'
        cm = plot_confusion_matrix(
            evaluation_results['y_true'],
            evaluation_results['y_pred'],
            CLASSES,
            cm_path
        )
        mlflow.log_artifact(str(cm_path))

        # Guardar log
        log_path = LOGS_DIR / 'mobilenetv3_training_log.json'
        save_training_log(
            log_path,
            'MobileNetV3-Large',
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
        model_path = MODELS_DIR / 'mobilenetv3_final.keras'
        model.save(str(model_path))
        mlflow.log_artifact(str(model_path))

        print(f"\nModelo guardado en: {model_path}")
        print(f"Log guardado en: {log_path}")
        print(f"\nTest Accuracy: {evaluation_results['test_accuracy']:.4f}")

    print("\n" + "="*60)
    print("ENTRENAMIENTO COMPLETADO")
    print("="*60)

if __name__ == "__main__":
    train_mobilenetv3()
