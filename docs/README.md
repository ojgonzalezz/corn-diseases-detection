# Corn Diseases Detection - Edge Models

Deep learning system for corn leaf disease classification using lightweight architectures optimized for edge computing.

## Overview

This project implements a complete pipeline for automatic diagnosis of common corn leaf diseases using 4 edge-optimized architectures trained in Google Colab with free GPU access.

## Disease Classes

The model classifies 4 categories:
- Blight (Corn Borer)
- Common_Rust (Common Rust)
- Gray_Leaf_Spot (Gray Leaf Spot)
- Healthy (Healthy leaves)

## Project Structure

```
corn-diseases-detection/
├── data/                    # Dataset (ignored by git)
├── src/                     # Source code
│   ├── adapters/           # Data loaders
│   ├── builders/           # Edge model builders
│   ├── core/               # Central configuration
│   ├── pipelines/          # ML pipelines
│   └── utils/              # Utilities
├── tests/                  # Test suite (10 files)
├── experimentation/        # EDA scripts and notebooks
├── experiments/            # Edge computing experiments
│   └── edge_models/        # Lightweight architecture training
├── notebooks/colab_edge_models_training.ipynb  # Main Colab notebook
├── COLAB_SETUP.md          # Colab setup guide
└── README.md
```

## Quick Start

1. Upload `data/` folder to Google Drive under `MyDrive/corn-diseases-data/`
2. Open `notebooks/colab_edge_models_training.ipynb` in Google Colab
3. Set runtime to GPU (T4)
4. Run all cells
5. Wait 2-3 hours for training completion

See documentation in `docs/` folder for detailed instructions.
