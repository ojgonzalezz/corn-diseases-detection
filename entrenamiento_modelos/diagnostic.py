"""
Script de diagnóstico para identificar problemas en el setup de entrenamiento
Ejecuta este script antes del setup_and_train.py para identificar dónde se queda atascado
"""

import sys
import time
import subprocess
from pathlib import Path

def test_step(step_name, test_func):
    """Ejecutar un paso de diagnóstico"""
    print(f"\n{'='*60}")
    print(f"PRUEBA: {step_name}")
    print('='*60)
    start_time = time.time()

    try:
        result = test_func()
        elapsed = time.time() - start_time
        if result:
            print(f"✓ PASÓ ({elapsed:.2f}s)")
        else:
            print(f"✗ FALLÓ ({elapsed:.2f}s)")
        return result
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"✗ ERROR ({elapsed:.2f}s)")
        print(f"Error: {e}")
        return False

def check_colab():
    """Verificar que estamos en Colab"""
    try:
        import google.colab
        print("✓ Estamos en Google Colab")
        return True
    except ImportError:
        print("✗ NO estamos en Google Colab")
        print("Este script está diseñado para Google Colab")
        return False

def check_drive_mount():
    """Verificar montaje de Google Drive"""
    try:
        from google.colab import drive
        drive_path = Path('/content/drive/MyDrive')

        if drive_path.exists():
            print("✓ Google Drive ya está montado")
            return True
        else:
            print("Intentando montar Google Drive...")
            drive.mount('/content/drive')

            if drive_path.exists():
                print("✓ Google Drive montado correctamente")
                return True
            else:
                print("✗ No se pudo montar Google Drive")
                return False
    except Exception as e:
        print(f"✗ Error montando Drive: {e}")
        return False

def check_gpu():
    """Verificar GPU disponible"""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ GPU detectada: {gpus[0].name}")
            # Verificar que funciona
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [1.0, 1.0]])
                c = tf.matmul(a, b)
            print("✓ GPU funciona correctamente")
            return True
        else:
            print("✗ No se detectó GPU")
            print("Ve a Runtime > Change runtime type > Hardware accelerator > GPU")
            return False
    except Exception as e:
        print(f"✗ Error con GPU: {e}")
        return False

def check_repo():
    """Verificar repositorio"""
    repo_path = Path('/content/corn-diseases-detection')

    if repo_path.exists():
        print("✓ Repositorio existe")
        # Verificar que tiene los archivos necesarios
        required_files = [
            'entrenamiento_modelos/train_all_models.py',
            'entrenamiento_modelos/config.py',
            'entrenamiento_modelos/utils.py'
        ]
        missing = []
        for file in required_files:
            if not (repo_path / file).exists():
                missing.append(file)

        if missing:
            print(f"✗ Faltan archivos: {missing}")
            return False
        else:
            print("✓ Todos los archivos necesarios están presentes")
            return True
    else:
        print("✗ Repositorio no encontrado")
        print("Ejecuta el paso de clonado en setup_and_train.py")
        return False

def check_dependencies():
    """Verificar dependencias"""
    required_packages = [
        'tensorflow',
        'mlflow',
        'matplotlib',
        'seaborn',
        'scikit-learn'
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package} disponible")
        except ImportError:
            missing.append(package)
            print(f"✗ {package} no disponible")

    if missing:
        print(f"Paquetes faltantes: {missing}")
        return False
    else:
        print("✓ Todas las dependencias principales están instaladas")
        return True

def check_dataset():
    """Verificar dataset"""
    data_dir = Path('/content/drive/MyDrive/data_processed')

    if not data_dir.exists():
        print(f"✗ Dataset no encontrado en {data_dir}")
        print("Asegúrate de tener 'data_processed' en la raíz de tu Google Drive")
        return False

    # Verificar estructura
    expected_classes = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']
    missing_classes = []
    total_images = 0

    for class_name in expected_classes:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            missing_classes.append(class_name)
            continue

        # Contar imágenes
        images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.png'))
        total_images += len(images)
        print(f"  {class_name}: {len(images)} imágenes")

    if missing_classes:
        print(f"✗ Faltan clases: {missing_classes}")
        return False

    if total_images == 0:
        print("✗ No se encontraron imágenes")
        return False

    print(f"✓ Dataset válido: {total_images} imágenes totales en {len(expected_classes)} clases")
    return True

def check_training_script():
    """Verificar que el script de entrenamiento puede importar sin errores"""
    try:
        # Cambiar al directorio correcto
        import os
        original_dir = os.getcwd()
        os.chdir('/content/corn-diseases-detection/entrenamiento_modelos')

        # Intentar importar módulos
        import config
        print("✓ config.py se importa correctamente")

        import utils
        print("✓ utils.py se importa correctamente")

        # Verificar configuración
        print(f"✓ Clases configuradas: {config.CLASSES}")
        print(f"✓ GPU limit: {config.GPU_MEMORY_LIMIT}")
        print(f"✓ Batch size: {config.BATCH_SIZE}")
        print(f"✓ Epochs: {config.EPOCHS}")

        # Volver al directorio original
        os.chdir(original_dir)
        return True

    except Exception as e:
        print(f"✗ Error importando módulos: {e}")
        return False

def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║               DIAGNÓSTICO DEL ENTORNO                       ║
╚══════════════════════════════════════════════════════════════╝
    """)

    tests = [
        ("Entorno Colab", check_colab),
        ("Montaje Google Drive", check_drive_mount),
        ("GPU disponible", check_gpu),
        ("Repositorio", check_repo),
        ("Dependencias", check_dependencies),
        ("Dataset", check_dataset),
        ("Scripts de entrenamiento", check_training_script),
    ]

    results = []
    for test_name, test_func in tests:
        result = test_step(test_name, test_func)
        results.append((test_name, result))

    # Resumen
    print(f"\n{'='*60}")
    print("RESUMEN DE DIAGNÓSTICO")
    print('='*60)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✓ PASÓ" if result else "✗ FALLÓ"
        print(f"{status}: {test_name}")
        if result:
            passed += 1

    print(f"\n{passed}/{total} pruebas pasaron")

    if passed == total:
        print("\n🎉 ¡Todo está configurado correctamente!")
        print("Puedes ejecutar setup_and_train.py sin problemas")
    else:
        print("\n⚠️  Hay problemas que deben solucionarse antes de ejecutar el entrenamiento")
        print("Revisa los errores arriba y corrígelos")

    print(f"\n{'='*60}")

if __name__ == "__main__":
    main()
