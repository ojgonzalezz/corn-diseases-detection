#!/usr/bin/env python3
"""
Script de prueba para verificar la carga eficiente de datos.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipelines.preprocess import split_and_balance_dataset_efficient
from src.utils.utils import create_efficient_dataset_from_dict
from src.core import config

def test_data_loading():
    print("Testing efficient data loading...")

    try:
        # Load data
        print("1. Loading data...")
        raw_dataset, label_to_int = split_and_balance_dataset_efficient()
        print(f"   Classes: {list(label_to_int.keys())}")
        print(f"   Train samples: {sum(len(paths) for paths in raw_dataset['train'].values())}")
        print(f"   Val samples: {sum(len(paths) for paths in raw_dataset['val'].values())}")
        print(f"   Test samples: {sum(len(paths) for paths in raw_dataset['test'].values())}")

        # Check if paths are strings
        print("2. Checking data types...")
        sample_train = raw_dataset['train']
        first_class = list(sample_train.keys())[0]
        first_samples = sample_train[first_class][:3]

        for i, sample in enumerate(first_samples):
            print(f"   Sample {i}: {type(sample)} - {sample}")
            if not isinstance(sample, str):
                print(f"   ERROR: Expected string path, got {type(sample)}")
                return False

        # Test dataset creation
        print("3. Creating dataset...")
        train_dataset, _ = create_efficient_dataset_from_dict(
            raw_dataset['train'],
            image_size=config.data.image_size,
            batch_size=16,
            num_classes=config.data.num_classes,
            shuffle=False,
            augment=False
        )

        # Test getting one batch
        print("4. Testing batch extraction...")
        for images, labels in train_dataset.take(1):
            print(f"   Batch shape: images={images.shape}, labels={labels.shape}")
            print(f"   Image range: [{images.numpy().min():.3f}, {images.numpy().max():.3f}]")
            print(f"   Label values: {labels.numpy()[:5]}")
            break

        print("SUCCESS: Data loading is working correctly!")
        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_data_loading()
