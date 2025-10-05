#!/usr/bin/env python3

import sys
import subprocess
import argparse
from datetime import datetime
from pathlib import Path

def run_command(cmd, description):
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
    parser = argparse.ArgumentParser(description='MobileNetV3Large Pipeline')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--data-path', default='../data')
    parser.add_argument('--max-samples', type=int, default=500)
    parser.add_argument('--inference-samples', type=int, default=20)
    args = parser.parse_args()

    print("MobileNetV3Large Pipeline")
    print(f"Start: {datetime.now()}")

    # Create output directories
    Path('models').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)

    model_file = 'models/mobilenetv3_large_optimized.tflite'

    # Step 1: Convert to TFLite
    if not run_command(
        f"python convert_to_tflite.py --config {args.config} --output {model_file} --data-path {args.data_path}",
        "Convert to TensorFlow Lite"
    ):
        sys.exit(1)

    # Step 2: Validate model
    run_command(
        f"python validate_model.py --config {args.config} --model {model_file} --test-data {args.data_path} --max-samples {args.max_samples} --output results/validation_report.json",
        "Validate model"
    )

    # Step 3: Run inference
    run_command(
        f"python inference.py --config {args.config} --model {model_file} --data-path {args.data_path} --batch --num-samples {args.inference_samples}",
        "Run inference demo"
    )

    print(f"\nPipeline completed: {datetime.now()}")

if __name__ == "__main__":
    main()
