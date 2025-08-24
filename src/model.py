# Archivo: src/model.py

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

def build_model(input_shape=(224, 224, 3), num_classes=4):
    """
    Construye un modelo de clasificación de imágenes utilizando Transfer Learning con VGG16.

    Args:
        input_shape (tuple): Dimensiones de las imágenes de entrada (alto, ancho, canales).
        num_classes (int): Número de clases a predecir.

    Returns:
        keras.Model: El modelo Keras compilado y listo para entrenar.
    """
    # --- 1. Cargar el Modelo Base Pre-entrenado (VGG16) ---
    # include_top=False: Excluye la capa de clasificación original de VGG16 (que predecía 1000 clases).
    # weights='imagenet': Carga los pesos aprendidos en el dataset ImageNet.
    # input_shape: Define el tamaño de las imágenes que aceptará el modelo.
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)

    # --- 2. Congelar el Modelo Base ---
    # Marcamos las capas del modelo base como no entrenables para no perder su conocimiento.
    base_model.trainable = False
    
    print(f"✅ Modelo base VGG16 cargado y congelado. Capas: {len(base_model.layers)}")
    # base_model.summary() # Descomenta para ver la arquitectura detallada de VGG16

    # --- 3. Añadir Nuevas Capas de Clasificación ---
    # Creamos un nuevo modelo secuencial que comienza donde termina el modelo base.
    model = models.Sequential([
        base_model,                               # La base convolucional congelada
        layers.Flatten(),                         # Aplana la salida 3D a un vector 1D
        layers.Dense(512, activation='relu'),     # Una capa densa con 512 neuronas para aprender patrones complejos
        layers.Dropout(0.5),                      # Capa de regularización para prevenir el sobreajuste
        layers.Dense(num_classes, activation='softmax') # Capa de salida con una neurona por clase y activación softmax
    ])

    # --- 4. Compilar el Modelo ---
    # Prepara el modelo para el entrenamiento.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), # Optimizador Adam con una tasa de aprendizaje baja
        loss='categorical_crossentropy',                      # Función de pérdida para clasificación multiclase
        metrics=['accuracy']                                  # Métrica para monitorear: la exactitud
    )
    
    print("✅ Modelo final construido y compilado.")
    return model

if __name__ == '__main__':
    # --- Verificación de la Arquitectura del Modelo ---
    print("🧠 Construyendo el modelo para verificación...")
    
    # Crear una instancia del modelo
    cnn_model = build_model()
    
    # Imprimir un resumen de la arquitectura
    # Esto es crucial para entender cómo se conectan las capas y cuántos parámetros tiene el modelo.
    print("\n" + "="*70)
    print("Resumen de la Arquitectura del Modelo Final")
    print("="*70)
    cnn_model.summary()
    print("="*70)