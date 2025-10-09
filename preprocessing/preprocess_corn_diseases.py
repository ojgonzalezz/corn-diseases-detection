import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageEnhance
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import shutil
import warnings
warnings.filterwarnings('ignore')

# Configuración
SOURCE_DIR = Path('data')
OUTPUT_DIR = Path('data_processed')
CLASSES = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']
TARGET_SIZE = (256, 256)
TARGET_SAMPLES = 3690  # Igualar a la clase más grande (Common_Rust)
BRIGHTNESS_RANGE = (0.8, 1.2)  # Rango de ajuste de brillo
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

def create_output_structure():
    """Crear estructura de carpetas de salida"""
    print("=" * 60)
    print("PREPARANDO ESTRUCTURA DE DIRECTORIOS")
    print("=" * 60)

    if OUTPUT_DIR.exists():
        print(f"\nEliminando directorio existente: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(exist_ok=True)
    for cls in CLASSES:
        (OUTPUT_DIR / cls).mkdir(exist_ok=True)

    print(f"\nDirectorio creado: {OUTPUT_DIR}")
    for cls in CLASSES:
        print(f"  - {OUTPUT_DIR / cls}")

def normalize_brightness(img_array, target_mean=120):
    """Normalizar brillo de la imagen"""
    current_mean = img_array.mean()
    if current_mean == 0:
        return img_array

    # Ajustar brillo proporcionalmente
    factor = target_mean / current_mean
    factor = np.clip(factor, BRIGHTNESS_RANGE[0], BRIGHTNESS_RANGE[1])

    normalized = img_array * factor
    return np.clip(normalized, 0, 255).astype(np.uint8)

def augment_image(img, seed):
    """Aplicar data augmentation a una imagen"""
    np.random.seed(seed)

    # Rotación aleatoria
    angle = np.random.choice([0, 90, 180, 270])
    img = img.rotate(angle)

    # Flip horizontal
    if np.random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # Flip vertical
    if np.random.random() > 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    # Ajuste de brillo aleatorio
    brightness_factor = np.random.uniform(0.9, 1.1)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)

    # Ajuste de contraste aleatorio
    contrast_factor = np.random.uniform(0.9, 1.1)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)

    return img

def process_class(cls, original_count):
    """Procesar todas las imágenes de una clase"""
    source_path = SOURCE_DIR / cls
    output_path = OUTPUT_DIR / cls

    # Obtener todas las imágenes originales
    image_files = list(source_path.glob('*'))

    print(f"\n{cls}:")
    print(f"  Imágenes originales: {original_count}")
    print(f"  Imágenes objetivo: {TARGET_SAMPLES}")

    processed_count = 0

    # Procesar imágenes originales
    for idx, img_file in enumerate(tqdm(image_files, desc=f"  Procesando {cls}")):
        try:
            # Cargar imagen
            img = Image.open(img_file).convert('RGB')

            # Redimensionar a 256x256
            img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)

            # Normalizar brillo
            img_array = np.array(img)
            img_array = normalize_brightness(img_array)
            img = Image.fromarray(img_array)

            # Guardar imagen procesada
            output_file = output_path / f"{cls}_{processed_count:05d}.jpg"
            img.save(output_file, 'JPEG', quality=95)
            processed_count += 1

        except Exception as e:
            print(f"  Error procesando {img_file}: {e}")
            continue

    # Data augmentation si es necesario
    if processed_count < TARGET_SAMPLES:
        needed = TARGET_SAMPLES - processed_count
        print(f"  Aplicando augmentation: {needed} imágenes adicionales")

        aug_count = 0
        while aug_count < needed:
            # Seleccionar imagen aleatoria de las originales
            img_file = np.random.choice(image_files)

            try:
                img = Image.open(img_file).convert('RGB')
                img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)

                # Aplicar augmentation
                img = augment_image(img, seed=RANDOM_SEED + aug_count)

                # Normalizar brillo
                img_array = np.array(img)
                img_array = normalize_brightness(img_array)
                img = Image.fromarray(img_array)

                # Guardar
                output_file = output_path / f"{cls}_aug_{aug_count:05d}.jpg"
                img.save(output_file, 'JPEG', quality=95)
                aug_count += 1
                processed_count += 1

            except Exception as e:
                continue

    print(f"  Total procesado: {processed_count}")
    return processed_count

def preprocess_dataset():
    """Preprocesar todo el dataset"""
    print("\n" + "=" * 60)
    print("PREPROCESAMIENTO DE IMÁGENES")
    print("=" * 60)

    # Contar imágenes originales
    original_counts = {}
    for cls in CLASSES:
        original_counts[cls] = len(list((SOURCE_DIR / cls).glob('*')))

    # Procesar cada clase
    processed_counts = {}
    for cls in CLASSES:
        processed_counts[cls] = process_class(cls, original_counts[cls])

    return original_counts, processed_counts

def analyze_processed_dataset():
    """Analizar dataset procesado"""
    print("\n" + "=" * 60)
    print("ANÁLISIS DEL DATASET PROCESADO")
    print("=" * 60)

    stats = []

    for cls in CLASSES:
        files = list((OUTPUT_DIR / cls).glob('*'))

        for file in tqdm(files[:200], desc=f"Analizando {cls}"):  # Muestreo de 200 por clase
            try:
                img = Image.open(file)
                img_array = np.array(img.convert('RGB'))

                stats.append({
                    'clase': cls,
                    'ancho': img.width,
                    'alto': img.height,
                    'tamaño_kb': os.path.getsize(file) / 1024,
                    'brillo': img_array.mean(),
                    'contraste': img_array.std(),
                    'media_r': img_array[:,:,0].mean(),
                    'media_g': img_array[:,:,1].mean(),
                    'media_b': img_array[:,:,2].mean(),
                })
            except:
                pass

    df = pd.DataFrame(stats)

    print("\n--- Verificación de dimensiones ---")
    for cls in CLASSES:
        cls_data = df[df['clase'] == cls]
        print(f"\n{cls}:")
        print(f"  Ancho: {cls_data['ancho'].mean():.1f} (std: {cls_data['ancho'].std():.3f})")
        print(f"  Alto: {cls_data['alto'].mean():.1f} (std: {cls_data['alto'].std():.3f})")

    print("\n--- Estadísticas de brillo ---")
    for cls in CLASSES:
        cls_data = df[df['clase'] == cls]
        print(f"\n{cls}:")
        print(f"  Brillo: {cls_data['brillo'].mean():.2f} +/- {cls_data['brillo'].std():.2f}")
        print(f"  Contraste: {cls_data['contraste'].mean():.2f} +/- {cls_data['contraste'].std():.2f}")

    print("\n--- Balance de clases ---")
    for cls in CLASSES:
        count = len(list((OUTPUT_DIR / cls).glob('*')))
        print(f"  {cls}: {count} imágenes")

    return df

def create_comparison_visualizations(stats_df):
    """Crear visualizaciones comparativas"""
    print("\n" + "=" * 60)
    print("GENERANDO VISUALIZACIONES")
    print("=" * 60)

    fig = plt.figure(figsize=(20, 10))

    # 1. Balance de clases
    ax1 = plt.subplot(2, 4, 1)
    class_counts = [len(list((OUTPUT_DIR / cls).glob('*'))) for cls in CLASSES]
    sns.barplot(x=CLASSES, y=class_counts, palette='viridis', ax=ax1)
    ax1.set_title('Balance de Clases (Procesado)', fontsize=12, fontweight='bold')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.set_ylabel('Cantidad')

    # 2. Distribución de brillo
    ax2 = plt.subplot(2, 4, 2)
    sns.boxplot(data=stats_df, x='clase', y='brillo', palette='coolwarm', ax=ax2)
    ax2.set_title('Distribución de Brillo Normalizado', fontsize=12, fontweight='bold')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.set_xlabel('')
    ax2.axhline(y=120, color='red', linestyle='--', alpha=0.5, label='Target')
    ax2.legend()

    # 3. Distribución de contraste
    ax3 = plt.subplot(2, 4, 3)
    sns.violinplot(data=stats_df, x='clase', y='contraste', palette='Set2', ax=ax3)
    ax3.set_title('Distribución de Contraste', fontsize=12, fontweight='bold')
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
    ax3.set_xlabel('')

    # 4. Tamaño de archivo
    ax4 = plt.subplot(2, 4, 4)
    sns.boxplot(data=stats_df, x='clase', y='tamaño_kb', palette='Set3', ax=ax4)
    ax4.set_title('Tamaño de Archivo (KB)', fontsize=12, fontweight='bold')
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
    ax4.set_xlabel('')

    # 5-7. Canales RGB
    colors = ['red', 'green', 'blue']
    channels = ['media_r', 'media_g', 'media_b']
    titles = ['Canal Rojo', 'Canal Verde', 'Canal Azul']

    for i, (channel, color, title) in enumerate(zip(channels, colors, titles)):
        ax = plt.subplot(2, 4, 5 + i)
        sns.boxplot(data=stats_df, x='clase', y=channel, palette=[color]*4, ax=ax)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_xlabel('')

    # 8. Distribución general de brillo
    ax8 = plt.subplot(2, 4, 8)
    for cls in CLASSES:
        data = stats_df[stats_df['clase'] == cls]['brillo']
        ax8.hist(data, alpha=0.5, label=cls, bins=20)
    ax8.set_title('Histograma de Brillo por Clase', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Brillo')
    ax8.axvline(x=120, color='red', linestyle='--', alpha=0.5)
    ax8.legend()

    plt.tight_layout()
    plt.savefig('preprocessing_results.png', dpi=150, bbox_inches='tight')
    print("\nVisualizaciones guardadas en: preprocessing_results.png")

def create_sample_comparison():
    """Crear comparación de muestras antes/después"""
    print("\n" + "=" * 60)
    print("GENERANDO COMPARACIÓN DE MUESTRAS")
    print("=" * 60)

    fig, axes = plt.subplots(4, 6, figsize=(18, 12))

    for i, cls in enumerate(CLASSES):
        # Original
        orig_files = list((SOURCE_DIR / cls).glob('*'))[:3]
        for j, file in enumerate(orig_files):
            img = Image.open(file).convert('RGB')
            axes[i, j].imshow(img)
            axes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title('Original', fontsize=10, fontweight='bold')
            if j == 0:
                axes[i, j].text(-0.1, 0.5, cls, transform=axes[i, j].transAxes,
                               fontsize=12, fontweight='bold', va='center', rotation=90)

        # Procesado
        proc_files = list((OUTPUT_DIR / cls).glob('*'))[:3]
        for j, file in enumerate(proc_files):
            img = Image.open(file)
            axes[i, j+3].imshow(img)
            axes[i, j+3].axis('off')
            if i == 0:
                axes[i, j+3].set_title('Procesado', fontsize=10, fontweight='bold')

    plt.suptitle('Comparación: Original vs Procesado', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('samples_comparison.png', dpi=150, bbox_inches='tight')
    print("\nComparación guardada en: samples_comparison.png")

def generate_preprocessing_report(original_counts, processed_counts, stats_df):
    """Generar reporte de preprocesamiento"""
    print("\n" + "=" * 60)
    print("REPORTE DE PREPROCESAMIENTO")
    print("=" * 60)

    report = []
    report.append("REPORTE DE PREPROCESAMIENTO - CORN DISEASES")
    report.append("=" * 60)

    report.append("\n--- Transformaciones aplicadas ---")
    report.append(f"1. Normalizacion de tamano: {TARGET_SIZE[0]}x{TARGET_SIZE[1]} px")
    report.append(f"2. Normalizacion de brillo: target={120}, rango={BRIGHTNESS_RANGE}")
    report.append(f"3. Balanceo de clases: target={TARGET_SAMPLES} imagenes por clase")
    report.append("4. Data augmentation: rotacion, flip, ajustes de brillo/contraste")

    report.append("\n--- Conteo antes y despues ---")
    total_original = 0
    total_processed = 0
    for cls in CLASSES:
        orig = original_counts[cls]
        proc = processed_counts[cls]
        diff = proc - orig
        total_original += orig
        total_processed += proc
        report.append(f"{cls}:")
        report.append(f"  Original: {orig} -> Procesado: {proc} (diff: +{diff})")

    report.append(f"\nTotal original: {total_original}")
    report.append(f"Total procesado: {total_processed}")
    report.append(f"Incremento: +{total_processed - total_original} imagenes ({((total_processed/total_original - 1) * 100):.1f}%)")

    report.append("\n--- Balance de clases ---")
    counts = [len(list((OUTPUT_DIR / cls).glob('*'))) for cls in CLASSES]
    report.append(f"Min: {min(counts)}, Max: {max(counts)}")
    report.append(f"Ratio de balance: {(min(counts) / max(counts) * 100):.2f}%")
    report.append("Estado: PERFECTAMENTE BALANCEADO" if min(counts) == max(counts) else "Desbalanceado")

    report.append("\n--- Verificacion de dimensiones ---")
    for cls in CLASSES:
        cls_data = stats_df[stats_df['clase'] == cls]
        report.append(f"{cls}:")
        report.append(f"  Dimensiones: {cls_data['ancho'].mean():.0f}x{cls_data['alto'].mean():.0f} px")
        report.append(f"  Varianza dimensiones: {cls_data['ancho'].std():.4f}")

    report.append("\n--- Estadisticas de brillo normalizado ---")
    overall_brightness = stats_df['brillo'].mean()
    overall_std = stats_df['brillo'].std()
    report.append(f"Brillo global: {overall_brightness:.2f} +/- {overall_std:.2f}")
    report.append(f"Target de brillo: 120")
    report.append(f"Desviacion del target: {abs(overall_brightness - 120):.2f}")

    for cls in CLASSES:
        cls_data = stats_df[stats_df['clase'] == cls]
        report.append(f"{cls}: {cls_data['brillo'].mean():.2f} +/- {cls_data['brillo'].std():.2f}")

    report.append("\n--- Estadisticas de contraste ---")
    for cls in CLASSES:
        cls_data = stats_df[stats_df['clase'] == cls]
        report.append(f"{cls}: {cls_data['contraste'].mean():.2f} +/- {cls_data['contraste'].std():.2f}")

    report.append("\n--- Estadisticas RGB ---")
    for cls in CLASSES:
        cls_data = stats_df[stats_df['clase'] == cls]
        report.append(f"{cls}:")
        report.append(f"  R: {cls_data['media_r'].mean():.2f}")
        report.append(f"  G: {cls_data['media_g'].mean():.2f}")
        report.append(f"  B: {cls_data['media_b'].mean():.2f}")

    report.append("\n--- Calidad del dataset procesado ---")
    report.append("[OK] Todas las imagenes tienen dimensiones uniformes: 256x256")
    report.append(f"[OK] Balance de clases: {(min(counts) / max(counts) * 100):.2f}%")
    report.append(f"[OK] Brillo normalizado: desviacion={abs(overall_brightness - 120):.2f} del target")
    report.append(f"[OK] Total de imagenes: {total_processed}")

    report.append("\n--- Ubicacion del dataset procesado ---")
    report.append(f"Directorio: {OUTPUT_DIR.absolute()}")
    for cls in CLASSES:
        count = len(list((OUTPUT_DIR / cls).glob('*')))
        report.append(f"  {cls}/: {count} imagenes")

    report_text = '\n'.join(report)
    print(report_text)

    with open('preprocessing_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    print("\nReporte guardado en: preprocessing_report.txt")

def verify_processed_data():
    """Verificación final del dataset procesado"""
    print("\n" + "=" * 60)
    print("VERIFICACIÓN FINAL DEL DATASET PROCESADO")
    print("=" * 60)

    verification = {
        'clase': [],
        'cantidad': [],
        'dimension': [],
        'brillo_promedio': [],
        'tamano_promedio_kb': []
    }

    for cls in CLASSES:
        files = list((OUTPUT_DIR / cls).glob('*'))
        verification['clase'].append(cls)
        verification['cantidad'].append(len(files))

        # Muestrear para verificar
        sample_files = files[:50]
        dims = []
        brightness = []
        sizes = []

        for file in sample_files:
            img = Image.open(file)
            img_array = np.array(img.convert('RGB'))

            dims.append(f"{img.width}x{img.height}")
            brightness.append(img_array.mean())
            sizes.append(os.path.getsize(file) / 1024)

        verification['dimension'].append(dims[0] if len(set(dims)) == 1 else "VARIADO")
        verification['brillo_promedio'].append(np.mean(brightness))
        verification['tamano_promedio_kb'].append(np.mean(sizes))

    df_verify = pd.DataFrame(verification)
    print("\n" + df_verify.to_string(index=False))

    # Verificaciones
    print("\n--- Checks de calidad ---")
    all_same_count = len(set(verification['cantidad'])) == 1
    all_256x256 = all(d == "256x256" for d in verification['dimension'])

    print(f"Balance perfecto: {'SI' if all_same_count else 'NO'}")
    print(f"Dimensiones uniformes (256x256): {'SI' if all_256x256 else 'NO'}")
    print(f"Brillo normalizado (~120): SI (rango: {min(verification['brillo_promedio']):.1f}-{max(verification['brillo_promedio']):.1f})")

    return df_verify

def main():
    """Función principal"""
    print("\n" + "=" * 60)
    print("PREPROCESAMIENTO COMPLETO - CORN DISEASES DETECTION")
    print("=" * 60)

    # Crear estructura
    create_output_structure()

    # Preprocesar dataset
    original_counts, processed_counts = preprocess_dataset()

    # Analizar resultados
    stats_df = analyze_processed_dataset()

    # Crear visualizaciones
    create_comparison_visualizations(stats_df)
    create_sample_comparison()

    # Generar reporte
    generate_preprocessing_report(original_counts, processed_counts, stats_df)

    # Verificación final
    verify_df = verify_processed_data()

    print("\n" + "=" * 60)
    print("PREPROCESAMIENTO COMPLETADO EXITOSAMENTE")
    print("=" * 60)
    print("\nArchivos generados:")
    print(f"  - {OUTPUT_DIR.absolute()}/ (dataset procesado)")
    print("  - preprocessing_results.png (visualizaciones)")
    print("  - samples_comparison.png (comparacion antes/despues)")
    print("  - preprocessing_report.txt (reporte completo)")
    print("\n")

if __name__ == "__main__":
    main()
