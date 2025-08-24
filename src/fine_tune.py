# Archivo: src/fine_tune.py

import tensorflow as tf
import pathlib
from datetime import datetime

# Importar las funciones que ya creamos
from data_pipeline import create_data_generators
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def fine_tune_model():
    """
    Carga el mejor modelo entrenado, descongela capas superiores y
    continúa el entrenamiento con una tasa de aprendizaje baja (fine-tuning).
    """
    # --- 1. CONFIGURACIÓN ---
    PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
    SPLIT_DATA_DIR = PROJECT_ROOT / 'dataset_split_balanced'
    BEST_MODEL_PATH = PROJECT_ROOT / 'models' / 'best_model.keras'
    
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    # Entrenaremos por menos épocas para el ajuste fino
    FINE_TUNE_EPOCHS = 10
    
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo base en '{BEST_MODEL_PATH}'. "
                                "Asegúrate de haber completado el entrenamiento inicial primero.")

    # --- 2. CARGAR DATOS Y MODELO PREVIAMENTE ENTRENADO ---
    print("Cargando y preparando los datos...")
    train_generator, validation_generator, test_generator = create_data_generators(
        base_dir=SPLIT_DATA_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    print(f"\n🧠 Cargando el modelo entrenado desde: '{BEST_MODEL_PATH}'")
    model = tf.keras.models.load_model(BEST_MODEL_PATH)

    # --- 3. DESCONGELAR CAPAS SUPERIORES DEL MODELO BASE ---
    # Accedemos a la base VGG16 por su nombre (por defecto es 'vgg16')
    base_model = model.get_layer('vgg16')
    base_model.trainable = True # Descongelar la base completa
    
    # Congelar todas las capas excepto las del último bloque convolucional
    # VGG16 tiene 19 capas. Congelaremos todas hasta la capa 15.
    print(f"Descongelando las últimas 4 capas de VGG16 para el ajuste fino...")
    for layer in base_model.layers[:-4]:
        layer.trainable = False

    # --- 4. RE-COMPILAR EL MODELO CON UNA TASA DE APRENDIZAJE MUY BAJA ---
    # Es CRÍTICO recompilar después de cambiar el estado 'trainable' de las capas.
    model.compile(
        # Usamos una tasa de aprendizaje 10 veces más pequeña que la original
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n✅ Modelo re-compilado para ajuste fino.")
    model.summary()

    # --- 5. CONFIGURAR CALLBACKS PARA EL AJUSTE FINO ---
    models_dir = PROJECT_ROOT / 'models'
    
    # Guardará el mejor modelo del proceso de fine-tuning
    finetune_checkpoint_cb = ModelCheckpoint(
        filepath=models_dir / 'fine_tuned_best_model.keras',
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )

    # Early stopping para el fine-tuning
    finetune_early_stopping_cb = EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True
    )

    # --- 6. CONTINUAR EL ENTRENAMIENTO (FINE-TUNING) ---
    print("\n" + "="*70)
    print("🚀 ¡Comenzando el ajuste fino (Fine-Tuning)!")
    print("="*70)
    
    history_finetune = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=FINE_TUNE_EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        callbacks=[finetune_checkpoint_cb, finetune_early_stopping_cb]
    )

    print("\n" + "="*70)
    print("✅ ¡Ajuste fino completado exitosamente!")
    print("="*70)
    print(f"🏆 El mejor modelo ajustado se guardó en: {models_dir / 'fine_tuned_best_model.keras'}")

    return history_finetune, model

if __name__ == '__main__':
    fine_tune_model()