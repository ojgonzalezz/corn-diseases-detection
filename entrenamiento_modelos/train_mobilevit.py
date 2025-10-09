"""
Entrenamiento de MobileViT para clasificación de enfermedades del maíz
Implementación simplificada de Vision Transformer móvil
"""

import os
import time
import mlflow
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from datetime import datetime

# Importar configuración y utilidades
from config import *
from utils import *

def conv_block(x, filters, kernel_size=3, strides=1):
    """Bloque convolucional"""
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    return x

def inverted_residual_block(x, expanded_channels, output_channels, strides=1):
    """Bloque residual invertido (estilo MobileNet)"""
    input_channels = x.shape[-1]

    # Expansion
    expanded = conv_block(x, expanded_channels, kernel_size=1)

    # Depthwise
    depthwise = layers.DepthwiseConv2D(3, strides=strides, padding='same')(expanded)
    depthwise = layers.BatchNormalization()(depthwise)
    depthwise = layers.Activation('swish')(depthwise)

    # Projection
    projected = layers.Conv2D(output_channels, 1, padding='same')(depthwise)
    projected = layers.BatchNormalization()(projected)

    # Skip connection
    if strides == 1 and input_channels == output_channels:
        return layers.Add()([x, projected])
    return projected

def transformer_block(x, num_heads=4, mlp_dim=128):
    """Bloque Transformer simplificado"""
    batch, height, width, channels = x.shape

    # Reshape para attention
    num_patches = height * width
    x_reshaped = layers.Reshape((num_patches, channels))(x)

    # Multi-head attention
    attn_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=channels // num_heads
    )(x_reshaped, x_reshaped)

    attn_output = layers.Dropout(0.1)(attn_output)
    x1 = layers.Add()([x_reshaped, attn_output])
    x1 = layers.LayerNormalization(epsilon=1e-6)(x1)

    # MLP
    mlp_output = layers.Dense(mlp_dim, activation='swish')(x1)
    mlp_output = layers.Dropout(0.1)(mlp_output)
    mlp_output = layers.Dense(channels)(mlp_output)
    mlp_output = layers.Dropout(0.1)(mlp_output)

    x2 = layers.Add()([x1, mlp_output])
    x2 = layers.LayerNormalization(epsilon=1e-6)(x2)

    # Reshape back
    output = layers.Reshape((height, width, channels))(x2)

    return output

def create_mobilevit_model(num_classes, image_size, learning_rate):
    """Crear modelo MobileViT simplificado"""

    inputs = layers.Input(shape=(*image_size, 3))

    # Stem
    x = conv_block(inputs, 32, kernel_size=3, strides=2)

    # Stage 1: MobileNet blocks
    x = inverted_residual_block(x, 64, 32)
    x = inverted_residual_block(x, 128, 64, strides=2)

    # Stage 2: MobileViT block
    x = inverted_residual_block(x, 256, 96)
    x = transformer_block(x, num_heads=4, mlp_dim=192)
    x = inverted_residual_block(x, 384, 128, strides=2)

    # Stage 3: MobileViT block
    x = inverted_residual_block(x, 512, 160)
    x = transformer_block(x, num_heads=4, mlp_dim=256)

    # Stage 4: Final blocks
    x = inverted_residual_block(x, 640, 192, strides=2)
    x = conv_block(x, 384, kernel_size=1)

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='swish')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)

    # Compilar
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_mobilevit():
    """Función principal de entrenamiento"""

    print("="*60)
    print("ENTRENAMIENTO MOBILEVIT")
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
    print("\nCreando modelo MobileViT...")
    model = create_mobilevit_model(NUM_CLASSES, IMAGE_SIZE, LEARNING_RATE)
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
            str(MODELS_DIR / 'mobilevit_best.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    # Hiperparámetros
    hyperparameters = {
        'model_name': 'MobileViT',
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
        'data_augmentation': DATA_AUGMENTATION,
        'transformer_heads': 4,
        'transformer_mlp_dim': '192/256'
    }

    # Iniciar MLflow run
    with mlflow.start_run(run_name=f"MobileViT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):

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

        total_time = time.time() - start_time
        print(f"\nTiempo de entrenamiento: {total_time:.2f} segundos ({total_time/60:.2f} minutos)")

        # Evaluar modelo
        print("\nEvaluando modelo en test set...")
        evaluation_results = evaluate_model(model, test_gen, CLASSES)

        # Graficar historial
        plot_path = LOGS_DIR / 'mobilevit_training_history.png'
        plot_training_history(history, plot_path)
        mlflow.log_artifact(str(plot_path))

        # Matriz de confusión
        cm_path = LOGS_DIR / 'mobilevit_confusion_matrix.png'
        cm = plot_confusion_matrix(
            evaluation_results['y_true'],
            evaluation_results['y_pred'],
            CLASSES,
            cm_path
        )
        mlflow.log_artifact(str(cm_path))

        # Guardar log
        log_path = LOGS_DIR / 'mobilevit_training_log.json'
        save_training_log(
            log_path,
            'MobileViT',
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
        model_path = MODELS_DIR / 'mobilevit_final.keras'
        model.save(str(model_path))
        mlflow.log_artifact(str(model_path))

        print(f"\nModelo guardado en: {model_path}")
        print(f"Log guardado en: {log_path}")
        print(f"\nTest Accuracy: {evaluation_results['test_accuracy']:.4f}")

    print("\n" + "="*60)
    print("ENTRENAMIENTO COMPLETADO")
    print("="*60)

if __name__ == "__main__":
    train_mobilevit()
