#####################################################################################
# ---------------------------------- Data Pipelines ---------------------------------
#####################################################################################

#########################
# ---- Depdendencies ----
#########################

import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt

from src.utils.utils import *

##################################
# ---- Data loader generators ----
##################################

def create_data_generators(base_dir, image_size=(224, 224), batch_size=32):
    """
    Crea los generadores de datos para entrenamiento, validación y prueba.
    Aplica aumento de datos solo al conjunto de entrenamiento.
    """
    train_dir = base_dir / 'train'
    validation_dir = base_dir / 'val'
    test_dir = base_dir / 'test'

    if not train_dir.exists():
        raise FileNotFoundError(f"El subdirectorio 'train' no fue encontrado en '{base_dir}'. "
                                "Asegúrate de que la división del dataset se completó correctamente.")

    print("[INICIO] Configurando generador de entrenamiento con aumento de datos...")
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        #rescale=1./255,
        #rotation_range=40,
        #width_shift_range=0.2,
        #height_shift_range=0.2,
        #shear_range=0.2,
        #zoom_range=0.2,
        #horizontal_flip=True,
        #fill_mode='nearest'
        #----------------------
        #mejor
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )

    print("[CONFIG] Configurando generadores de validación y prueba (solo normalización)...")
    validation_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    validation_generator = validation_test_datagen.flow_from_directory(
        validation_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_generator = validation_test_datagen.flow_from_directory(
        test_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    print("\n[OK] Generadores de datos creados exitosamente.")
    return train_generator, validation_generator, test_generator

def plot_augmented_images(generator):
    """Función de utilidad para visualizar el efecto del Data Augmentation."""
    print("\n[VISUAL] Mostrando ejemplos de imágenes aumentadas del primer lote de entrenamiento:")
    images, labels = next(generator)
    
    plt.figure(figsize=(12, 12))
    for i in range(min(9, len(images))): # Asegura no exceder el tamaño del lote
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        class_index = tf.argmax(labels[i]).numpy()
        class_name = list(generator.class_indices.keys())[class_index]
        plt.title(class_name)
        plt.axis("off")
    plt.suptitle("Visualización del Aumento de Datos en Tiempo Real", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == '__main__':
    # --- CONFIGURACIÓN ---
    PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
    
    # --- Verificación de Directorio Inteligente ---
    # Busca la carpeta de datos divididos, dando prioridad a la versión balanceada.
    dir_balanced = PROJECT_ROOT / 'dataset_split_balanced'
    dir_standard = PROJECT_ROOT / 'dataset_split'
    
    if dir_balanced.exists():
        SPLIT_DATA_DIR = dir_balanced
    elif dir_standard.exists():
        SPLIT_DATA_DIR = dir_standard
    else:
        raise FileNotFoundError(
            f"[ERROR] No se encontró el directorio de datos divididos. \n"
            f"Asegúrate de haber ejecutado 'preprocessing/preprocess.py' y que la carpeta "
            f"'{dir_balanced.name}' o '{dir_standard.name}' exista en el directorio raíz del proyecto."
        )
    
    print(f"[OK] Directorio de datos encontrado: '{SPLIT_DATA_DIR.name}'")
    
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    # ---------------------

    # --- EJECUCIÓN DEL PROCESO ---
    train_gen, val_gen, test_gen = create_data_generators(
        base_dir=SPLIT_DATA_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    # --- Verificación ---
    print("\n" + "="*50)
    print("VERIFICACIÓN DE LOS GENERADORES")
    print("="*50)
    print(f"Clases encontradas y sus índices: {train_gen.class_indices}")

    sample_images, sample_labels = next(train_gen)
    print(f"Dimensiones de un lote de imágenes: {sample_images.shape}")
    print(f"Dimensiones de un lote de etiquetas: {sample_labels.shape}")
    print("="*50)

    # Visualizar las imágenes aumentadas
    plot_augmented_images(train_gen)