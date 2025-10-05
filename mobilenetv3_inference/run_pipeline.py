#!/usr/bin/env python3
"""
Script principal del pipeline MobileNetV3Large para detección de enfermedades del maíz.

Este script orquesta automáticamente todo el proceso:
1. Construcción y entrenamiento del modelo
2. Conversión a TensorFlow Lite
3. Validación del modelo optimizado
4. Demo de inferencia

Uso:
    python run_pipeline.py --data-path /path/to/data

Autor: Sistema de Detección de Enfermedades del Maíz
"""

import sys
import subprocess
import argparse
from datetime import datetime
from pathlib import Path

def run_command(cmd, description):
    """
    Ejecuta un comando del sistema y maneja errores.

    Args:
        cmd: Comando a ejecutar
        description: Descripción del comando para logging

    Returns:
        bool: True si el comando fue exitoso, False en caso contrario
    """
    print(f"\n{'='*50}")
    print(f"RUNNING: {description}")
    print('='*50)

    try:
        subprocess.run(cmd, shell=True, check=True, cwd='.')
        print("SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {e.returncode}")
        return False

def main():
    """
    Función principal que ejecuta todo el pipeline MobileNetV3Large.

    El pipeline consta de 3 pasos principales:
    1. Conversión del modelo a TensorFlow Lite (incluye entrenamiento)
    2. Validación del modelo optimizado
    3. Demo de inferencia
    """
    parser = argparse.ArgumentParser(description='MobileNetV3Large Pipeline')
    parser.add_argument('--config', default='config.yaml', help='Archivo de configuración YAML')
    parser.add_argument('--data-path', default='../data', help='Ruta a los datos de entrenamiento')
    parser.add_argument('--max-samples', type=int, default=500, help='Máximo número de muestras para validación')
    parser.add_argument('--inference-samples', type=int, default=20, help='Número de muestras para demo de inferencia')
    args = parser.parse_args()

    print("MobileNetV3Large Pipeline")
    print(f"Start: {datetime.now()}")

    # Crear directorios de salida si no existen
    Path('models').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)

    model_file = 'models/mobilenetv3_large_optimized.tflite'

    # Paso 1: Convertir modelo a TensorFlow Lite (incluye construcción y entrenamiento)
    if not run_command(
        f"python convert_to_tflite.py --config {args.config} --output {model_file} --data-path {args.data_path}",
        "Convert to TensorFlow Lite"
    ):
        sys.exit(1)

    # Paso 2: Validar el modelo optimizado
    run_command(
        f"python validate_model.py --config {args.config} --model {model_file} --test-data {args.data_path} --max-samples {args.max_samples} --output results/validation_report.json",
        "Validate model"
    )

    # Paso 3: Ejecutar demo de inferencia
    run_command(
        f"python inference.py --config {args.config} --model {model_file} --data-path {args.data_path} --batch --num-samples {args.inference_samples}",
        "Run inference demo"
    )

    print(f"\nPipeline completed: {datetime.now()}")

if __name__ == "__main__":
    main()
