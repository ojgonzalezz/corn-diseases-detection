import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- CONFIGURACIÓN ---
DATA_ROOT = pathlib.Path('../data')
# ---------------------

def get_class_distribution(root_path: pathlib.Path) -> pd.DataFrame:
    """
    Recorre un directorio, cuenta los archivos de imagen por clase y devuelve
    un DataFrame de pandas con los resultados.
    """
    class_counts = {}
    class_dirs = sorted([d for d in root_path.iterdir() if d.is_dir()])
    
    for class_dir in class_dirs:
        class_name = class_dir.name
        image_count = len([p for p in class_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
        class_counts[class_name] = image_count

    df = pd.DataFrame(list(class_counts.items()), columns=['Clase', 'Cantidad'])
    return df.sort_values(by='Cantidad', ascending=False)

def plot_enhanced_distribution(df: pd.DataFrame):
    """
    Genera un gráfico de barras mejorado para evidenciar el desbalance de clases.
    """
    if df.empty:
        print("El DataFrame está vacío. No se puede generar el gráfico.")
        return
        
    # --- 1. CÁLCULOS PARA EVIDENCIAR EL DESBALANCE ---
    mean_count = df['Cantidad'].mean()
    min_class_name = df.iloc[-1]['Clase'] # La clase con menos imágenes

    # --- 2. CONFIGURACIÓN DEL GRÁFICO ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # Crear una paleta de colores dinámica para resaltar la clase minoritaria
    colors = ['#40B5AD' if c != min_class_name else '#E34A6F' for c in df['Clase']]

    # Crear el gráfico de barras con Seaborn y la paleta personalizada
    sns.barplot(x='Clase', y='Cantidad', data=df, ax=ax, palette=colors)

    # --- 3. MEJORAS VISUALES ---
    # Añadir línea de promedio
    ax.axhline(mean_count, color='red', linestyle='--', linewidth=2, label=f'Promedio: {mean_count:.0f}')
    ax.legend()
    
    # Añadir el número exacto sobre cada barra
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points', fontweight='bold')

    # Títulos y etiquetas
    ax.set_title('Análisis de Desbalance de Clases', fontsize=18, fontweight='bold')
    ax.set_xlabel('Clase de Enfermedad', fontsize=12)
    ax.set_ylabel('Cantidad de Imágenes', fontsize=12)
    plt.xticks(rotation=15)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if not DATA_ROOT.exists():
        print(f" Error: El directorio '{DATA_ROOT}' no fue encontrado.")
    else:
        distribution_df = get_class_distribution(DATA_ROOT)
        
        # --- RESUMEN NUMÉRICO EN TERMINAL ---
        print("\n" + "="*40)
        print("📊 RESUMEN DEL DESBALANCE DE CLASES")
        print("="*40)
        print(f"📈 Clase con más imágenes: '{distribution_df.iloc[0]['Clase']}' ({distribution_df.iloc[0]['Cantidad']})")
        print(f"📉 Clase con menos imágenes: '{distribution_df.iloc[-1]['Clase']}' ({distribution_df.iloc[-1]['Cantidad']})")
        print(f"⚖️ Promedio de imágenes por clase: {distribution_df['Cantidad'].mean():.2f}")
        print(f"📊 Desviación estándar: {distribution_df['Cantidad'].std():.2f}")
        print("="*40, "\n")
        
        plot_enhanced_distribution(distribution_df)