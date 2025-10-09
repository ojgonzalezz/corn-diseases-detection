"""
Entrenamiento de PMVT (Plant Mobile Vision Transformer) para clasificación de enfermedades del maíz
Lightweight Vision Transformer optimizado para dispositivos móviles y clasificación de enfermedades de plantas
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

def patch_embedding(x, patch_size, embed_dim):
    """Convertir imagen en patches y embeddings"""
    patches = layers.Conv2D(
        embed_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding='valid'
    )(x)
    batch, height, width, channels = patches.shape
    patches = layers.Reshape((height * width, channels))(patches)
    return patches, height, width

def lightweight_msa(x, num_heads, key_dim):
    """Multi-head Self-Attention ligero"""
    # Reducir dimensionalidad antes de attention para eficiencia
    channels = x.shape[-1]
    reduced_dim = channels // 2

    # Linear projection para reducir computación
    x_reduced = layers.Dense(reduced_dim)(x)

    # Multi-head attention
    attn_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=0.1
    )(x_reduced, x_reduced)

    # Project back
    attn_output = layers.Dense(channels)(attn_output)

    return attn_output

def mlp_block(x, hidden_dim, dropout=0.1):
    """MLP block con GELU activation"""
    x = layers.Dense(hidden_dim)(x)
    x = layers.Activation('gelu')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(x.shape[-1])(x)
    x = layers.Dropout(dropout)(x)
    return x

def pmvt_block(x, num_heads, mlp_dim, key_dim):
    """Bloque PMVT: Attention + MLP con residual connections"""
    # Layer norm + Attention
    x1 = layers.LayerNormalization(epsilon=1e-6)(x)
    attn_output = lightweight_msa(x1, num_heads, key_dim)
    x = layers.Add()([x, attn_output])

    # Layer norm + MLP
    x2 = layers.LayerNormalization(epsilon=1e-6)(x)
    mlp_output = mlp_block(x2, mlp_dim)
    x = layers.Add()([x, mlp_output])

    return x

def depthwise_separable_conv(x, filters, kernel_size=3, strides=1):
    """Convolución depthwise separable (eficiente)"""
    x = layers.DepthwiseConv2D(
        kernel_size,
        strides=strides,
        padding='same'
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('gelu')(x)

    x = layers.Conv2D(filters, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('gelu')(x)

    return x

def create_pmvt_model(num_classes, image_size, learning_rate):
    """
    Crear modelo PMVT (Plant Mobile Vision Transformer)
    Diseñado específicamente para clasificación de enfermedades de plantas en móviles
    """

    inputs = layers.Input(shape=(*image_size, 3))

    # Stem: Reducción inicial ligera
    x = depthwise_separable_conv(inputs, 32, kernel_size=3, strides=2)
    x = depthwise_separable_conv(x, 64, kernel_size=3, strides=1)

    # Stage 1: Extracción de features locales
    x = depthwise_separable_conv(x, 96, kernel_size=3, strides=2)

    # Patch Embedding (dividir en patches para transformer)
    patch_size = 4
    embed_dim = 128

    patches, h, w = patch_embedding(x, patch_size, embed_dim)

    # Positional Embedding (aprendible)
    num_patches = h * w
    positions = tf.range(start=0, limit=num_patches, delta=1)
    pos_embedding = layers.Embedding(
        input_dim=num_patches,
        output_dim=embed_dim
    )(positions)

    # Add positional encoding
    patches = patches + pos_embedding

    # Stage 2: Transformer blocks (ligeros)
    # Menos bloques para mantener el modelo liviano
    x = pmvt_block(patches, num_heads=4, mlp_dim=256, key_dim=32)
    x = pmvt_block(x, num_heads=4, mlp_dim=256, key_dim=32)
    x = pmvt_block(x, num_heads=4, mlp_dim=256, key_dim=32)

    # Layer normalization final
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    # Global pooling
    x = layers.GlobalAveragePooling1D()(x)

    # Classification head (optimizado para plantas)
    x = layers.Dense(256, activation='gelu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='gelu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs, name='PMVT')

    # Compilar
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_pmvt():
    """Función principal de entrenamiento"""

    print("="*60)
    print("ENTRENAMIENTO PMVT (Plant Mobile Vision Transformer)")
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
    print("\nCreando modelo PMVT...")
    model = create_pmvt_model(NUM_CLASSES, IMAGE_SIZE, LEARNING_RATE)
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
            str(MODELS_DIR / 'pmvt_best.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    # Hiperparámetros
    hyperparameters = {
        'model_name': 'PMVT (Plant Mobile Vision Transformer)',
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
        'patch_size': 4,
        'embed_dim': 128,
        'num_transformer_blocks': 3,
        'num_heads': 4,
        'mlp_dim': 256,
        'key_dim': 32
    }

    # Iniciar MLflow run
    with mlflow.start_run(run_name=f"PMVT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):

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
        plot_path = LOGS_DIR / 'pmvt_training_history.png'
        plot_training_history(history, plot_path)
        mlflow.log_artifact(str(plot_path))

        # Matriz de confusión
        cm_path = LOGS_DIR / 'pmvt_confusion_matrix.png'
        cm = plot_confusion_matrix(
            evaluation_results['y_true'],
            evaluation_results['y_pred'],
            CLASSES,
            cm_path
        )
        mlflow.log_artifact(str(cm_path))

        # Guardar log
        log_path = LOGS_DIR / 'pmvt_training_log.json'
        save_training_log(
            log_path,
            'PMVT (Plant Mobile Vision Transformer)',
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
        mlflow.log_metric("num_parameters", model.count_params())

        # Guardar modelo
        model_path = MODELS_DIR / 'pmvt_final.keras'
        model.save(str(model_path))
        mlflow.log_artifact(str(model_path))

        print(f"\nModelo guardado en: {model_path}")
        print(f"Log guardado en: {log_path}")
        print(f"\nTest Accuracy: {evaluation_results['test_accuracy']:.4f}")

    print("\n" + "="*60)
    print("ENTRENAMIENTO COMPLETADO")
    print("="*60)

if __name__ == "__main__":
    train_pmvt()
