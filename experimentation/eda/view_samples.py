import pathlib
import random
import matplotlib.pyplot as plt
from PIL import Image

# --- CONFIGURACIÓN ---
DATA_ROOT = pathlib.Path('../data')
# Número de imágenes aleatorias que quieres ver por cada clase
SAMPLES_PER_CLASS = 5
# ---------------------

def display_image_samples(root_path: pathlib.Path, num_samples: int):
    """
    Muestra una cuadrícula de imágenes con muestras aleatorias de cada clase.
    """
    print("Seleccionando muestras aleatorias para visualización...")
    
    class_dirs = sorted([d for d in root_path.iterdir() if d.is_dir()])
    num_classes = len(class_dirs)

    if num_classes == 0:
        print(" No se encontraron directorios de clases.")
        return

    # Configurar la cuadrícula de visualización (subplots)
    # Filas = número de clases, Columnas = número de muestras
    fig, axes = plt.subplots(
        nrows=num_classes, 
        ncols=num_samples, 
        figsize=(15, num_classes * 3) # Ancho fijo, altura depende de cuántas clases haya
    )
    
    fig.suptitle('Muestras Aleatorias por Clase de Enfermedad', fontsize=20, fontweight='bold')

    # Iterar sobre cada clase para llenar la cuadrícula
    for i, class_dir in enumerate(class_dirs):
        class_name = class_dir.name
        
        # Obtener todas las rutas de imágenes y seleccionar una muestra aleatoria
        image_paths = list(class_dir.glob('*.[jp][pn]g')) # Busca .jpg, .jpeg, .png
        if not image_paths:
            print(f"⚠️ No se encontraron imágenes en la clase '{class_name}'")
            continue
            
        sample_paths = random.sample(image_paths, min(num_samples, len(image_paths)))
        
        # Llenar cada columna de la fila actual
        for j, image_path in enumerate(sample_paths):
            ax = axes[i, j]
            # Abrir y mostrar la imagen
            ax.imshow(Image.open(image_path))
            ax.set_xticks([]) # Ocultar ejes para una vista más limpia
            ax.set_yticks([])
            
            # Poner el nombre de la clase como título de la primera imagen de cada fila
            if j == 0:
                ax.set_ylabel(class_name, rotation=0, labelpad=80, fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Ajustar para que el título principal no se superponga
    plt.show()


if __name__ == "__main__":
    if not DATA_ROOT.exists():
        print(f" Error: El directorio '{DATA_ROOT}' no fue encontrado.")
    else:
        display_image_samples(root_path=DATA_ROOT, num_samples=SAMPLES_PER_CLASS)