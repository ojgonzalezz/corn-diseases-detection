#!/usr/bin/env python3
"""
MobileNetV3Large Pipeline Automático
Ejecuta todo el pipeline de conversión, validación e inferencia automáticamente.
"""

import os
import sys
import subprocess
import json
import argparse
from datetime import datetime
from pathlib import Path

def run_command(cmd, description, cwd=None):
    """Ejecuta un comando y maneja errores."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print('='*60)

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        )
        print("✅ Completado exitosamente")
        if result.stdout:
            print("Salida:", result.stdout[-500:])  # Últimos 500 caracteres
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en {description}")
        print(f"Código de error: {e.returncode}")
        if e.stdout:
            print("Salida:", e.stdout[-1000:])
        if e.stderr:
            print("Error:", e.stderr[-1000:])
        return False

def main():
    parser = argparse.ArgumentParser(description='MobileNetV3Large Pipeline Automático')
    parser.add_argument('--config', default='config.yaml', help='Archivo de configuración')
    parser.add_argument('--data-path', default='../data', help='Ruta a los datos')
    parser.add_argument('--max-samples', type=int, default=500, help='Máximo número de muestras para validación')
    parser.add_argument('--inference-samples', type=int, default=20, help='Número de muestras para demo de inferencia')
    args = parser.parse_args()

    # Directorios de salida
    models_dir = Path('models')
    results_dir = Path('results')
    models_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    # Archivos de salida
    model_file = models_dir / 'mobilenetv3_large_optimized.tflite'
    validation_report = results_dir / 'validation_report.json'
    inference_report = results_dir / 'inference_demo.json'

    print("MobileNetV3Large Pipeline Automático")
    print(f"Inicio: {datetime.now()}")
    print(f"Configuración: {args.config}")
    print(f"Datos: {args.data_path}")
    print(f"Modelo de salida: {model_file}")
    print(f"Reporte de validación: {validation_report}")

    # PASO 1: Conversión a TFLite
    success = run_command(
        f"python convert_to_tflite.py --config {args.config} --output {model_file} --data-path {args.data_path}",
        "PASO 1: Conversión del modelo a TensorFlow Lite",
        cwd=Path('.')
    )

    if not success:
        print("\n❌ Falló la conversión del modelo")
        sys.exit(1)

    # Verificar que el modelo se creó
    if not model_file.exists():
        print(f"\n❌ El archivo del modelo no se creó: {model_file}")
        sys.exit(1)

    print(f"✅ Modelo creado: {model_file} ({model_file.stat().st_size / (1024*1024):.1f} MB)")

    # PASO 2: Validación del modelo
    success = run_command(
        f"python validate_model.py --config {args.config} --model {model_file} --test-data {args.data_path}/test --max-samples {args.max_samples} --output {validation_report}",
        f"PASO 2: Validación del modelo (máx. {args.max_samples} muestras)",
        cwd=Path('.')
    )

    if success and validation_report.exists():
        try:
            with open(validation_report, 'r') as f:
                report = json.load(f)

            accuracy = report.get('accuracy', 'N/A')
            print(f"📊 Accuracy: {accuracy:.4f}")
            print(f"📊 Matriz de confusión guardada en: {results_dir}/confusion_matrix.png")
        except Exception as e:
            print(f"⚠️  No se pudo leer el reporte de validación: {e}")
    else:
        print("\n⚠️  La validación falló o no generó reporte")

    # PASO 3: Demo de inferencia
    success = run_command(
        f"python inference.py --config {args.config} --model {model_file} --batch --num-samples {args.inference_samples}",
        f"PASO 3: Demo de inferencia ({args.inference_samples} muestras)",
        cwd=Path('.')
    )

    # PASO 4: Resumen final
    print(f"\n{'='*60}")
    print("📋 RESUMEN FINAL")
    print('='*60)

    print("Archivos generados:")
    print(f"  📁 Modelo optimizado: {model_file}")
    print(f"  📏 Tamaño: {model_file.stat().st_size / (1024*1024):.1f} MB")
    if validation_report.exists():
        print(f"  📊 Reporte de validación: {validation_report}")
        print(f"  🖼️  Matriz de confusión: {results_dir}/confusion_matrix.png")

    print("\nConfiguración utilizada:")
    print(f"  ⚙️  Config: {args.config}")
    print(f"  📂 Datos: {args.data_path}")
    print(f"  🔢 Muestras validación: {args.max_samples}")
    print(f"  🚀 Muestras inferencia: {args.inference_samples}")

    print(f"\n✅ Pipeline completado: {datetime.now()}")
    print("\n💡 Próximos pasos:")
    print("   - Revisar el accuracy en el reporte de validación")
    print("   - Ver la matriz de confusión generada")
    print("   - El modelo está listo para despliegue en dispositivos edge")

if __name__ == "__main__":
    main()
