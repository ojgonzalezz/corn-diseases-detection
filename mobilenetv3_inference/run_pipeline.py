#!/usr/bin/env python3

import os
import sys
import subprocess
import json
import argparse
from datetime import datetime
from pathlib import Path

def run_command(cmd, description, cwd=None):
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print('='*60)

    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, check=True)
        print("SUCCESS: Completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR in {description}: {e.returncode}")
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

    print("MobileNetV3Large Pipeline")
    print(f"Start: {datetime.now()}")
    print(f"Config: {args.config}")
    print(f"Data: {args.data_path}")

    # Step 1: Convert to TFLite
    success = run_command(
        f"python convert_to_tflite.py --config {args.config} --output {model_file} --data-path {args.data_path}",
        "Step 1: Convert model to TensorFlow Lite"
    )

    if not success or not model_file.exists():
        print("ERROR: Model conversion failed")
        sys.exit(1)

    print(f"Model created: {model_file} ({model_file.stat().st_size / (1024*1024):.1f} MB)")

    # Step 2: Validate model
    success = run_command(
        f"python validate_model.py --config {args.config} --model {model_file} --test-data {args.data_path}/test --max-samples {args.max_samples} --output {validation_report}",
        f"Step 2: Validate model (max {args.max_samples} samples)"
    )

    # Step 3: Run inference demo
    success = run_command(
        f"python inference.py --config {args.config} --model {model_file} --batch --num-samples {args.inference_samples}",
        f"Step 3: Run inference demo ({args.inference_samples} samples)"
    )

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print('='*60)

    print("Generated files:")
    print(f"  Model: {model_file}")
    print(f"  Size: {model_file.stat().st_size / (1024*1024):.1f} MB")
    if validation_report.exists():
        print(f"  Validation report: {validation_report}")
        print(f"  Confusion matrix: {results_dir}/confusion_matrix.png")

    print(f"\nPipeline completed: {datetime.now()}")

if __name__ == "__main__":
    main()
