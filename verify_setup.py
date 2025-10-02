#!/usr/bin/env python3
"""
Script de verificación de configuración del proyecto.

Este script verifica que todos los requisitos y configuraciones necesarias
estén correctamente establecidas antes de ejecutar el proyecto.
"""
import sys
import os
from pathlib import Path


def check_python_version():
    """Verifica que la versión de Python sea >= 3.10."""
    print("🐍 Verificando versión de Python...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"  ❌ ERROR: Python {version.major}.{version.minor} detectado")
        print(f"  Se requiere Python 3.10 o superior")
        return False
    print(f"  ✅ Python {version.major}.{version.minor}.{version.micro} OK")
    return True


def check_env_file():
    """Verifica que exista el archivo .env."""
    print("\n⚙️  Verificando archivo de configuración...")
    env_path = Path("src/core/.env")
    if not env_path.exists():
        print(f"  ❌ ERROR: No se encontró {env_path}")
        print(f"  Ejecuta: cp src/core/.env_example src/core/.env")
        return False
    print(f"  ✅ Archivo .env encontrado")
    return True


def check_directories():
    """Verifica que existan los directorios necesarios."""
    print("\n📁 Verificando estructura de directorios...")
    required_dirs = [
        "src",
        "src/adapters",
        "src/builders",
        "src/core",
        "src/pipelines",
        "src/utils",
        "src/api",
        "tests",
        "experimentation",
    ]
    
    all_ok = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if not path.exists():
            print(f"  ❌ Directorio faltante: {dir_path}")
            all_ok = False
        else:
            print(f"  ✅ {dir_path}")
    
    return all_ok


def check_data_directories():
    """Verifica la existencia de directorios de datos."""
    print("\n📊 Verificando directorios de datos...")
    data_path = Path("data")
    
    if not data_path.exists():
        print(f"  ⚠️  WARNING: Directorio 'data/' no encontrado")
        print(f"  El proyecto necesita datos para entrenar/evaluar")
        print(f"  Opciones:")
        print(f"    1. data/train, data/val, data/test (datos ya divididos)")
        print(f"    2. data/raw/data_1, data/raw/data_2 (datos crudos)")
        return False
    
    # Verificar si hay datos divididos
    has_split_data = (
        (data_path / "train").exists() and
        (data_path / "val").exists() and
        (data_path / "test").exists()
    )
    
    # Verificar si hay datos raw
    has_raw_data = (data_path / "raw").exists()
    
    if has_split_data:
        print(f"  ✅ Datos divididos encontrados (train/val/test)")
        return True
    elif has_raw_data:
        print(f"  ✅ Datos raw encontrados (raw/)")
        print(f"  ℹ️  Ejecuta preprocesamiento: python -m src.pipelines.preprocess")
        return True
    else:
        print(f"  ⚠️  WARNING: No se encontraron datos")
        return False


def check_dependencies():
    """Verifica que las dependencias principales estén instaladas."""
    print("\n📦 Verificando dependencias principales...")
    
    dependencies = {
        "tensorflow": "TensorFlow",
        "keras": "Keras",
        "numpy": "NumPy",
        "PIL": "Pillow",
        "pydantic": "Pydantic",
        "mlflow": "MLflow",
    }
    
    all_ok = True
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} no instalado")
            all_ok = False
    
    return all_ok


def check_config():
    """Verifica que la configuración se pueda cargar."""
    print("\n🔧 Verificando configuración del proyecto...")
    try:
        from src.core.config import config
        
        # Verificar algunos valores críticos
        assert config.data.num_classes > 0, "NUM_CLASSES debe ser > 0"
        assert len(config.data.class_names) == config.data.num_classes, \
            "CLASS_NAMES debe tener el mismo número de elementos que NUM_CLASSES"
        assert config.data.image_size[0] > 0 and config.data.image_size[1] > 0, \
            "IMAGE_SIZE debe ser válido"
        
        print(f"  ✅ Configuración cargada correctamente")
        print(f"    - Clases: {config.data.num_classes}")
        print(f"    - Tamaño imagen: {config.data.image_size}")
        print(f"    - Backbone: {config.training.backbone}")
        return True
    except Exception as e:
        print(f"  ❌ Error al cargar configuración: {e}")
        return False


def check_gpu():
    """Verifica disponibilidad de GPU (opcional)."""
    print("\n🎮 Verificando GPU (opcional)...")
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"  ✅ GPU disponible: {len(gpus)} dispositivo(s)")
            for gpu in gpus:
                print(f"    - {gpu.name}")
        else:
            print(f"  ℹ️  GPU no disponible (se usará CPU)")
        return True
    except Exception as e:
        print(f"  ⚠️  No se pudo verificar GPU: {e}")
        return True  # No es crítico


def main():
    """Función principal de verificación."""
    print("="*70)
    print("🔍 VERIFICACIÓN DE CONFIGURACIÓN DEL PROYECTO")
    print("="*70)
    
    checks = [
        ("Versión de Python", check_python_version),
        ("Archivo .env", check_env_file),
        ("Estructura de directorios", check_directories),
        ("Directorios de datos", check_data_directories),
        ("Dependencias", check_dependencies),
        ("Configuración", check_config),
        ("GPU", check_gpu),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n❌ Error en verificación '{name}': {e}")
            results[name] = False
    
    # Resumen
    print("\n" + "="*70)
    print("📋 RESUMEN DE VERIFICACIÓN")
    print("="*70)
    
    critical_checks = [
        "Versión de Python",
        "Archivo .env",
        "Estructura de directorios",
        "Dependencias",
        "Configuración"
    ]
    
    critical_passed = all(results.get(check, False) for check in critical_checks)
    warning_checks = [
        "Directorios de datos",
        "GPU"
    ]
    
    for check_name, passed in results.items():
        status = "✅" if passed else ("⚠️ " if check_name in warning_checks else "❌")
        print(f"{status} {check_name}")
    
    print("="*70)
    
    if critical_passed:
        print("\n🎉 ¡Configuración verificada exitosamente!")
        print("\nPróximos pasos:")
        print("  1. Asegúrate de tener datos en data/train, data/val, data/test")
        print("  2. Entrena un modelo: python -m src.pipelines.train")
        print("  3. Ejecuta tests: pytest tests/")
        print("  4. Inicia la API: python -m src.api.main")
        return 0
    else:
        print("\n⚠️  Hay problemas críticos que deben resolverse")
        print("Por favor, revisa los errores arriba y corrígelos.")
        return 1


if __name__ == "__main__":
    sys.exit(main())


