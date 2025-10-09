import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Configuración
DATA_DIR = Path('data')
CLASSES = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']
SAMPLE_SIZE = 500  # Para estadísticas de imágenes

def analyze_dataset_distribution():
    """Análisis de distribución de clases"""
    print("=" * 60)
    print("1. DISTRIBUCIÓN DE CLASES")
    print("=" * 60)

    class_counts = {}
    for cls in CLASSES:
        files = list((DATA_DIR / cls).glob('*'))
        class_counts[cls] = len(files)

    df = pd.DataFrame(list(class_counts.items()), columns=['Clase', 'Cantidad'])
    df['Porcentaje'] = (df['Cantidad'] / df['Cantidad'].sum() * 100).round(2)
    print(df.to_string(index=False))
    print(f"\nTotal de imágenes: {df['Cantidad'].sum()}")
    print(f"Balanceo: {(df['Cantidad'].min() / df['Cantidad'].max() * 100):.2f}%")

    return df

def analyze_image_properties():
    """Análisis de propiedades de las imágenes"""
    print("\n" + "=" * 60)
    print("2. PROPIEDADES DE LAS IMÁGENES")
    print("=" * 60)

    properties = []
    for cls in CLASSES:
        files = list((DATA_DIR / cls).glob('*'))[:SAMPLE_SIZE]
        for file in files:
            try:
                img = Image.open(file)
                properties.append({
                    'clase': cls,
                    'ancho': img.width,
                    'alto': img.height,
                    'modo': img.mode,
                    'aspecto': img.width / img.height,
                    'tamaño_kb': os.path.getsize(file) / 1024
                })
            except:
                pass

    df = pd.DataFrame(properties)

    # Resumen por clase
    print("\n--- Dimensiones por clase ---")
    for cls in CLASSES:
        cls_data = df[df['clase'] == cls]
        print(f"\n{cls}:")
        print(f"  Ancho: {cls_data['ancho'].mean():.1f} ± {cls_data['ancho'].std():.1f} px")
        print(f"  Alto: {cls_data['alto'].mean():.1f} ± {cls_data['alto'].std():.1f} px")
        print(f"  Ratio aspecto: {cls_data['aspecto'].mean():.3f} ± {cls_data['aspecto'].std():.3f}")
        print(f"  Tamaño: {cls_data['tamaño_kb'].mean():.1f} ± {cls_data['tamaño_kb'].std():.1f} KB")

    # Dimensiones únicas
    print("\n--- Resoluciones únicas ---")
    resolutions = df.groupby(['ancho', 'alto']).size().reset_index(name='count')
    resolutions = resolutions.sort_values('count', ascending=False).head(10)
    for _, row in resolutions.iterrows():
        print(f"  {int(row['ancho'])}x{int(row['alto'])}: {row['count']} imágenes")

    # Modos de color
    print("\n--- Modos de color ---")
    mode_counts = Counter(df['modo'])
    for mode, count in mode_counts.most_common():
        print(f"  {mode}: {count} ({count/len(df)*100:.1f}%)")

    return df

def analyze_pixel_statistics():
    """Análisis estadístico de píxeles"""
    print("\n" + "=" * 60)
    print("3. ESTADÍSTICAS DE PÍXELES")
    print("=" * 60)

    stats = []
    for cls in CLASSES:
        files = list((DATA_DIR / cls).glob('*'))[:SAMPLE_SIZE]
        for file in files:
            try:
                img = np.array(Image.open(file).convert('RGB'))
                stats.append({
                    'clase': cls,
                    'media_r': img[:,:,0].mean(),
                    'media_g': img[:,:,1].mean(),
                    'media_b': img[:,:,2].mean(),
                    'std_r': img[:,:,0].std(),
                    'std_g': img[:,:,1].std(),
                    'std_b': img[:,:,2].std(),
                    'brillo': img.mean(),
                    'contraste': img.std()
                })
            except:
                pass

    df = pd.DataFrame(stats)

    print("\n--- Estadisticas por canal RGB ---")
    for cls in CLASSES:
        cls_data = df[df['clase'] == cls]
        print(f"\n{cls}:")
        print(f"  R: media={cls_data['media_r'].mean():.2f}, std={cls_data['std_r'].mean():.2f}")
        print(f"  G: media={cls_data['media_g'].mean():.2f}, std={cls_data['std_g'].mean():.2f}")
        print(f"  B: media={cls_data['media_b'].mean():.2f}, std={cls_data['std_b'].mean():.2f}")
        print(f"  Brillo: {cls_data['brillo'].mean():.2f} +/- {cls_data['brillo'].std():.2f}")
        print(f"  Contraste: {cls_data['contraste'].mean():.2f} +/- {cls_data['contraste'].std():.2f}")

    return df

def create_visualizations(dist_df, prop_df, pixel_df):
    """Crear visualizaciones"""
    print("\n" + "=" * 60)
    print("4. GENERANDO VISUALIZACIONES")
    print("=" * 60)

    fig = plt.figure(figsize=(20, 12))

    # 1. Distribución de clases
    ax1 = plt.subplot(3, 4, 1)
    sns.barplot(data=dist_df, x='Clase', y='Cantidad', palette='viridis', ax=ax1)
    ax1.set_title('Distribución de Clases', fontsize=12, fontweight='bold')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

    # 2. Proporción de clases
    ax2 = plt.subplot(3, 4, 2)
    ax2.pie(dist_df['Cantidad'], labels=dist_df['Clase'], autopct='%1.1f%%', startangle=90)
    ax2.set_title('Proporción de Clases', fontsize=12, fontweight='bold')

    # 3. Distribución de anchos
    ax3 = plt.subplot(3, 4, 3)
    for cls in CLASSES:
        data = prop_df[prop_df['clase'] == cls]['ancho']
        ax3.hist(data, alpha=0.5, label=cls, bins=20)
    ax3.set_title('Distribución de Anchos', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Ancho (px)')
    ax3.legend(fontsize=8)

    # 4. Distribución de altos
    ax4 = plt.subplot(3, 4, 4)
    for cls in CLASSES:
        data = prop_df[prop_df['clase'] == cls]['alto']
        ax4.hist(data, alpha=0.5, label=cls, bins=20)
    ax4.set_title('Distribución de Altos', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Alto (px)')
    ax4.legend(fontsize=8)

    # 5. Ratio de aspecto
    ax5 = plt.subplot(3, 4, 5)
    sns.boxplot(data=prop_df, x='clase', y='aspecto', palette='Set2', ax=ax5)
    ax5.set_title('Ratio de Aspecto por Clase', fontsize=12, fontweight='bold')
    ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45, ha='right')
    ax5.set_xlabel('')

    # 6. Tamaño de archivo
    ax6 = plt.subplot(3, 4, 6)
    sns.violinplot(data=prop_df, x='clase', y='tamaño_kb', palette='Set2', ax=ax6)
    ax6.set_title('Tamaño de Archivo (KB)', fontsize=12, fontweight='bold')
    ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45, ha='right')
    ax6.set_xlabel('')

    # 7-9. Distribución de canales RGB
    colors = ['red', 'green', 'blue']
    channels = ['media_r', 'media_g', 'media_b']
    titles = ['Canal Rojo', 'Canal Verde', 'Canal Azul']

    for i, (channel, color, title) in enumerate(zip(channels, colors, titles)):
        ax = plt.subplot(3, 4, 7 + i)
        sns.boxplot(data=pixel_df, x='clase', y=channel, palette=[color]*4, ax=ax)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_xlabel('')

    # 10. Brillo promedio
    ax10 = plt.subplot(3, 4, 10)
    sns.violinplot(data=pixel_df, x='clase', y='brillo', palette='coolwarm', ax=ax10)
    ax10.set_title('Brillo Promedio', fontsize=12, fontweight='bold')
    ax10.set_xticklabels(ax10.get_xticklabels(), rotation=45, ha='right')
    ax10.set_xlabel('')

    # 11. Contraste
    ax11 = plt.subplot(3, 4, 11)
    sns.violinplot(data=pixel_df, x='clase', y='contraste', palette='coolwarm', ax=ax11)
    ax11.set_title('Contraste', fontsize=12, fontweight='bold')
    ax11.set_xticklabels(ax11.get_xticklabels(), rotation=45, ha='right')
    ax11.set_xlabel('')

    # 12. Correlación RGB
    ax12 = plt.subplot(3, 4, 12)
    corr_data = pixel_df[['media_r', 'media_g', 'media_b']].corr()
    sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', ax=ax12,
                xticklabels=['R', 'G', 'B'], yticklabels=['R', 'G', 'B'])
    ax12.set_title('Correlación Canales RGB', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('eda_corn_diseases.png', dpi=150, bbox_inches='tight')
    print("\nVisualizaciones guardadas en: eda_corn_diseases.png")

def show_sample_images():
    """Mostrar muestras de cada clase"""
    print("\n" + "=" * 60)
    print("5. MUESTRAS DE IMÁGENES")
    print("=" * 60)

    fig, axes = plt.subplots(4, 5, figsize=(15, 12))

    for i, cls in enumerate(CLASSES):
        files = list((DATA_DIR / cls).glob('*'))[:5]
        for j, file in enumerate(files):
            img = Image.open(file)
            axes[i, j].imshow(img)
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_title(cls, fontsize=12, fontweight='bold', loc='left')

    plt.suptitle('Muestras de Imágenes por Clase', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('samples_corn_diseases.png', dpi=150, bbox_inches='tight')
    print("\nMuestras guardadas en: samples_corn_diseases.png")

def generate_summary_report(dist_df, prop_df, pixel_df):
    """Generar reporte resumen"""
    print("\n" + "=" * 60)
    print("6. REPORTE RESUMEN")
    print("=" * 60)

    report = []
    report.append("RESUMEN EJECUTIVO - EDA CORN DISEASES")
    report.append("=" * 60)
    report.append(f"\nTotal de imágenes: {dist_df['Cantidad'].sum()}")
    report.append(f"Número de clases: {len(CLASSES)}")
    report.append(f"Clases: {', '.join(CLASSES)}")

    report.append(f"\n--- Balance de clases ---")
    report.append(f"Clase más frecuente: {dist_df.loc[dist_df['Cantidad'].idxmax(), 'Clase']} ({dist_df['Cantidad'].max()} imgs)")
    report.append(f"Clase menos frecuente: {dist_df.loc[dist_df['Cantidad'].idxmin(), 'Clase']} ({dist_df['Cantidad'].min()} imgs)")
    report.append(f"Ratio de balance: {(dist_df['Cantidad'].min() / dist_df['Cantidad'].max() * 100):.2f}%")

    report.append(f"\n--- Caracteristicas de imagenes ---")
    report.append(f"Ancho promedio: {prop_df['ancho'].mean():.1f} +/- {prop_df['ancho'].std():.1f} px")
    report.append(f"Alto promedio: {prop_df['alto'].mean():.1f} +/- {prop_df['alto'].std():.1f} px")
    report.append(f"Ratio aspecto promedio: {prop_df['aspecto'].mean():.3f}")
    report.append(f"Tamano promedio: {prop_df['tamaño_kb'].mean():.1f} KB")

    report.append(f"\n--- Caracteristicas de pixeles ---")
    report.append(f"Brillo promedio: {pixel_df['brillo'].mean():.2f} +/- {pixel_df['brillo'].std():.2f}")
    report.append(f"Contraste promedio: {pixel_df['contraste'].mean():.2f} +/- {pixel_df['contraste'].std():.2f}")
    report.append(f"Canal R: {pixel_df['media_r'].mean():.2f} +/- {pixel_df['std_r'].mean():.2f}")
    report.append(f"Canal G: {pixel_df['media_g'].mean():.2f} +/- {pixel_df['std_g'].mean():.2f}")
    report.append(f"Canal B: {pixel_df['media_b'].mean():.2f} +/- {pixel_df['std_b'].mean():.2f}")

    report.append(f"\n--- Recomendaciones ---")
    if (dist_df['Cantidad'].min() / dist_df['Cantidad'].max()) < 0.8:
        report.append("! Desbalance de clases detectado: considerar data augmentation o class weights")

    unique_sizes = prop_df[['ancho', 'alto']].drop_duplicates().shape[0]
    if unique_sizes > 5:
        report.append(f"! {unique_sizes} resoluciones diferentes: normalizar tamano en preprocessing")

    if pixel_df['brillo'].std() > 30:
        report.append("! Alta variabilidad en brillo: aplicar normalizacion o ecualizacion")

    report_text = '\n'.join(report)
    print(report_text)

    with open('eda_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    print("\nReporte guardado en: eda_report.txt")

def main():
    """Función principal"""
    print("\n" + "=" * 60)
    print("EDA COMPLETO - CORN DISEASES DETECTION")
    print("=" * 60)

    # Análisis
    dist_df = analyze_dataset_distribution()
    prop_df = analyze_image_properties()
    pixel_df = analyze_pixel_statistics()

    # Visualizaciones
    create_visualizations(dist_df, prop_df, pixel_df)
    show_sample_images()

    # Reporte final
    generate_summary_report(dist_df, prop_df, pixel_df)

    print("\n" + "=" * 60)
    print("EDA COMPLETADO EXITOSAMENTE")
    print("=" * 60)
    print("\nArchivos generados:")
    print("  - eda_corn_diseases.png (visualizaciones)")
    print("  - samples_corn_diseases.png (muestras)")
    print("  - eda_report.txt (reporte resumen)")
    print("\n")

if __name__ == "__main__":
    main()
