#####################################################################################
# ----------------------------------- EDA Functions ---------------------------------
#####################################################################################

#########################
# ---- Depdendencies ----
#########################

import pathlib
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm

# --- CONFIGURACIÓN ---
DATA_ROOT = pathlib.Path('../data')
# Analizar solo un subconjunto para agilizar el proceso. 1.0 = 100% de las imágenes.
SAMPLE_FRACTION = 0.5 
# ---------------------

##############################
# ---- Analize dimensions ----
##############################

def analyze_dimensions(root_path: pathlib.Path, fraction: float):
    """
    Analiza las dimensiones (ancho y alto) de una fracción de las imágenes.
    """
    print(" Analizando dimensiones de las imágenes...")
    image_paths = list(root_path.glob('*/*.[jp][pn]g'))
    
    # Tomar una muestra aleatoria para no procesar todo si no es necesario
    sample_size = int(len(image_paths) * fraction)
    sampled_paths = random.sample(image_paths, sample_size)
    
    dimensions = []
    for path in tqdm(sampled_paths, desc="Leyendo dimensiones"):
        try:
            with Image.open(path) as img:
                dimensions.append(img.size) # (width, height)
        except (IOError, OSError):
            continue # Ignorar archivos corruptos

    if not dimensions:
        print(" No se pudieron leer las dimensiones de las imágenes.")
        return

    df = pd.DataFrame(dimensions, columns=['Ancho', 'Alto'])
    
    # --- Plot de Dimensiones ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(x='Ancho', y='Alto', data=df, ax=ax, alpha=0.5)
    ax.set_title(f'Distribución de Dimensiones de las Imágenes (Muestra del {fraction*100}%)', fontsize=16)
    ax.set_xlabel('Ancho (píxeles)')
    ax.set_ylabel('Alto (píxeles)')
    print(f"\nDimensiones encontradas: Ancho (promedio: {df['Ancho'].mean():.0f}px), Alto (promedio: {df['Alto'].mean():.0f}px)")
    plt.tight_layout()
    plt.show()

def analyze_color_histograms(root_path: pathlib.Path, fraction: float):
    """
    Calcula y grafica los histogramas de color promedio para cada clase con colores distintivos.
    """
    print("\n Analizando histogramas de color...")
    class_dirs = sorted([d for d in root_path.iterdir() if d.is_dir()])
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # --- Paleta de colores para cada CLASE ---
    class_color_map = {
        'Healthy': '#2ECC71',        # Verde esmeralda
        'Common_Rust': '#E67E22',     # Naranja zanahoria
        'Blight': '#3498DB',         # Azul Peter River
        'Gray_Leaf_Spot': '#E74C3C'  # Rojo Alizarin
    }

    for class_dir in class_dirs:
        class_name = class_dir.name
        image_paths = list(class_dir.glob('*.[jp][pn]g'))
        
        sample_size = int(len(image_paths) * fraction)
        sampled_paths = random.sample(image_paths, sample_size)
        
        histograms_g = [] # Solo necesitamos el canal verde
        
        for path in tqdm(sampled_paths, desc=f"Procesando colores en '{class_name}'"):
            try:
                with Image.open(path) as img:
                    img_array = np.array(img)
                    hist_g, _ = np.histogram(img_array[:, :, 1].ravel(), bins=256, range=[0, 256])
                    histograms_g.append(hist_g)
            except Exception:
                continue
        
        if histograms_g:
            mean_hist = np.mean(histograms_g, axis=0)
            # Obtener el color específico para esta clase del mapa
            plot_color = class_color_map.get(class_name, 'black') # 'black' por si una clase no está en el mapa
            
            ax.plot(mean_hist, color=plot_color, label=f'{class_name} (Verde)', linewidth=2.5)

    ax.set_title('Histograma Promedio del Canal Verde por Clase', fontsize=16, fontweight='bold')
    ax.set_xlabel('Intensidad de Píxel (0-255)')
    ax.set_ylabel('Frecuencia Promedio')
    ax.legend(title='Clases')
    ax.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if not DATA_ROOT.exists():
        print(f" Error: El directorio '{DATA_ROOT}' no fue encontrado.")
    else:
        # Ejecutar ambos análisis
        analyze_dimensions(DATA_ROOT, SAMPLE_FRACTION)
        analyze_color_histograms(DATA_ROOT, SAMPLE_FRACTION)