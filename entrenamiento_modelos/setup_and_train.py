"""
Script completo para ejecutar entrenamiento en Google Colab
Ejecuta este script en una celda de Colab y todo se configurará automáticamente
Versión mejorada con mejor manejo de errores y timeouts

Para diagnosticar problemas, ejecuta primero:
python diagnostic.py

Esto te dirá exactamente dónde se queda atascado el script.
"""

import subprocess
import sys
import time
from pathlib import Path

def run_command(cmd, description, timeout=300):
    """Ejecutar comando con timeout y mostrar resultado"""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        if result.stdout:
            print(result.stdout)
        if result.returncode != 0 and result.stderr:
            print(f"Error: {result.stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"ERROR: Timeout ({timeout}s) alcanzado en: {description}")
        return False

def check_drive_mount(max_attempts=3):
    """Verificar y montar Google Drive con reintentos"""
    from google.colab import drive

    for attempt in range(max_attempts):
        try:
            print(f"Intento {attempt + 1}/{max_attempts} de montar Drive...")
            drive.mount('/content/drive', force_remount=True)

            # Verificar que Drive está montado
            drive_path = Path('/content/drive/MyDrive')
            if drive_path.exists():
                print("✓ Google Drive montado correctamente")
                return True
            else:
                print("✗ Drive montado pero no se puede acceder a MyDrive")
                continue

        except Exception as e:
            print(f"Error montando Drive (intento {attempt + 1}): {e}")
            if attempt < max_attempts - 1:
                print("Reintentando en 5 segundos...")
                time.sleep(5)
            continue

    return False

def verify_dataset(data_dir, max_wait_time=60):
    """Verificar dataset con timeout"""
    print(f"Verificando dataset en: {data_dir}")
    start_time = time.time()

    while time.time() - start_time < max_wait_time:
        if data_dir.exists():
            # Contar imágenes con timeout
            try:
                total_images = sum(1 for _ in data_dir.rglob('*.jpg')) + \
                              sum(1 for _ in data_dir.rglob('*.jpeg')) + \
                              sum(1 for _ in data_dir.rglob('*.png'))
                if total_images > 0:
                    print(f"✓ Dataset encontrado: {total_images} imágenes")
                    return True
                else:
                    print("Dataset existe pero no tiene imágenes. Esperando...")
            except Exception as e:
                print(f"Error contando imágenes: {e}")
        else:
            print(f"Dataset no encontrado en {data_dir}. Esperando...")

        time.sleep(5)

    print(f"✗ ERROR: Dataset no encontrado después de {max_wait_time} segundos")
    return False

def main():
    print("""
╔════════════════════════════════════════════════════════════════════╗
║     CONFIGURACIÓN AUTOMÁTICA - ENTRENAMIENTO EN GOOGLE COLAB      ║
╚════════════════════════════════════════════════════════════════════╝
    """)

    # 1. Verificar que estamos en Colab
    print("PASO 1: Verificando entorno...")
    try:
        import google.colab
        print("✓ Ejecutando en Google Colab")
    except ImportError:
        print("✗ ERROR: Este script debe ejecutarse en Google Colab")
        sys.exit(1)

    # 2. Montar Google Drive con reintentos
    print("\n" + "="*70)
    print("PASO 2: Montando Google Drive")
    print("="*70)
    if not check_drive_mount():
        print("\n✗ ERROR: No se pudo montar Google Drive después de varios intentos")
        print("Verifica tu conexión a internet y permisos de Drive")
        sys.exit(1)

    # 3. Verificar GPU
    print("\n" + "="*70)
    print("PASO 3: Verificando GPU")
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

    # 4. Clonar/actualizar repositorio
    print("\n" + "="*70)
    print("PASO 4: Configurando repositorio")
    print("="*70)
    if Path('/content/corn-diseases-detection').exists():
        print("Repositorio ya existe, actualizando...")
        if not run_command('cd /content/corn-diseases-detection && git pull origin pipe', 'Git pull', timeout=60):
            print("Error actualizando repositorio")
            sys.exit(1)
    else:
        if not run_command('git clone -b pipe https://github.com/ojgonzalezz/corn-diseases-detection.git /content/corn-diseases-detection', 'Git clone', timeout=120):
            print("Error clonando repositorio")
            sys.exit(1)

    # 5. Cambiar directorio
    import os
    os.chdir('/content/corn-diseases-detection/entrenamiento_modelos')
    print(f"✓ Directorio actual: {os.getcwd()}")

    # 6. Instalar dependencias
    print("\n" + "="*70)
    print("PASO 5: Instalando dependencias")
    print("="*70)
    if not run_command('pip install -q -r requirements.txt', 'Instalando dependencias', timeout=300):
        print("Error instalando dependencias")
        sys.exit(1)

    # 7. Verificar dataset con timeout
    print("\n" + "="*70)
    print("PASO 6: Verificando dataset")
    print("="*70)
    data_dir = Path('/content/drive/MyDrive/data_processed')
    if not verify_dataset(data_dir):
        print("\nAsegúrate de tener la carpeta 'data_processed' en:")
        print("  Mi unidad/data_processed/")
        print("\nLa estructura debe ser:")
        print("  Mi unidad/")
        print("  └── data_processed/")
        print("      ├── Blight/")
        print("      ├── Common_Rust/")
        print("      ├── Gray_Leaf_Spot/")
        print("      └── Healthy/")
        sys.exit(1)

    # 8. Iniciar entrenamiento con timeout
    print("\n" + "="*70)
    print("PASO 7: INICIANDO ENTRENAMIENTO")
    print("="*70)
    print("\nModelos a entrenar: MobileNetV3, EfficientNet, MobileViT, PMVT")
    print("Épocas por modelo: 20")
    print("Batch size: 64")
    print("NOTA: El entrenamiento puede tomar varias horas")
    print("\n" + "="*70 + "\n")

    # Ejecutar entrenamiento con timeout extendido (4 horas máximo)
    try:
        result = subprocess.run(['python', 'train_all_models.py'],
                              timeout=4*3600,  # 4 horas
                              check=True)
        print("\n✓ Entrenamiento completado exitosamente")
    except subprocess.TimeoutExpired:
        print("\n⚠ ADVERTENCIA: El entrenamiento excedió el tiempo límite (4 horas)")
        print("Puede que esté funcionando correctamente pero lentamente.")
        print("Revisa los logs y resultados en Drive.")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ ERROR durante el entrenamiento: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
