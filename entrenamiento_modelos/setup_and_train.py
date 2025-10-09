"""
Script completo para ejecutar entrenamiento en Google Colab
Ejecuta este script en una celda de Colab y todo se configurará automáticamente
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Ejecutar comando y mostrar resultado"""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0 and result.stderr:
        print(f"Error: {result.stderr}")
        return False
    return True

def main():
    print("""
╔════════════════════════════════════════════════════════════════════╗
║     CONFIGURACIÓN AUTOMÁTICA - ENTRENAMIENTO EN GOOGLE COLAB      ║
╚════════════════════════════════════════════════════════════════════╝
    """)

    # 1. Verificar que estamos en Colab
    try:
        import google.colab
        print("✓ Ejecutando en Google Colab")
    except ImportError:
        print("✗ ERROR: Este script debe ejecutarse en Google Colab")
        sys.exit(1)

    # 2. Montar Google Drive
    print("\n" + "="*70)
    print("PASO 1: Montando Google Drive")
    print("="*70)
    from google.colab import drive
    drive.mount('/content/drive')

    # 3. Verificar GPU
    print("\n" + "="*70)
    print("PASO 2: Verificando GPU")
    print("="*70)
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("\n" + "!" * 70)
        print("ERROR: NO SE DETECTÓ GPU")
        print("!" * 70)
        print("\nPara habilitar GPU:")
        print("1. Runtime > Change runtime type")
        print("2. Hardware accelerator > GPU")
        print("3. Save y reconectar")
        print("!" * 70)
        sys.exit(1)
    print(f"✓ GPU detectada: {gpus[0].name}")

    # 4. Clonar repositorio
    print("\n" + "="*70)
    print("PASO 3: Clonando repositorio")
    print("="*70)
    if Path('/content/corn-diseases-detection').exists():
        print("Repositorio ya existe, actualizando...")
        run_command('cd /content/corn-diseases-detection && git pull origin pipe', 'Git pull')
    else:
        run_command('git clone -b pipe https://github.com/ojgonzalezz/corn-diseases-detection.git /content/corn-diseases-detection', 'Git clone')

    # 5. Cambiar directorio
    import os
    os.chdir('/content/corn-diseases-detection/entrenamiento_modelos')
    print(f"✓ Directorio actual: {os.getcwd()}")

    # 6. Instalar dependencias
    if not run_command('pip install -q -r requirements.txt', 'PASO 4: Instalando dependencias'):
        print("Error instalando dependencias")
        sys.exit(1)

    # 7. Verificar dataset
    print("\n" + "="*70)
    print("PASO 5: Verificando dataset")
    print("="*70)
    data_dir = Path('/content/drive/MyDrive/data_processed')
    if not data_dir.exists():
        print(f"\n✗ ERROR: Dataset no encontrado en {data_dir}")
        print("\nAsegúrate de tener la carpeta 'data_processed' en:")
        print("  Mi unidad/data_processed/")
        sys.exit(1)

    # Contar imágenes
    total_images = sum(1 for _ in data_dir.rglob('*.jpg')) + sum(1 for _ in data_dir.rglob('*.jpeg')) + sum(1 for _ in data_dir.rglob('*.png'))
    print(f"✓ Dataset encontrado: {total_images} imágenes")

    # 8. Iniciar entrenamiento
    print("\n" + "="*70)
    print("PASO 6: INICIANDO ENTRENAMIENTO")
    print("="*70)
    print("\nModelos a entrenar: MobileNetV3, EfficientNet, MobileViT, PMVT")
    print("Épocas por modelo: 20")
    print("Batch size: 64")
    print("\n" + "="*70 + "\n")

    # Ejecutar entrenamiento (sin capturar output para ver progreso en tiempo real)
    subprocess.run(['python', 'train_all_models.py'])

if __name__ == "__main__":
    main()
