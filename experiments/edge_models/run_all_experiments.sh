#!/bin/bash
#
# Script para ejecutar todos los experimentos edge automáticamente
#

set -e  # Exit on error

echo "=================================================================="
echo "INICIANDO EXPERIMENTOS EDGE COMPUTING"
echo "=================================================================="
echo ""
echo "Paso 1/3: Entrenando los 4 modelos edge..."
python experiments/edge_models/train_all_models.py

echo ""
echo "=================================================================="
echo "Paso 2/3: Comparando resultados..."
python experiments/edge_models/compare_models.py

echo ""
echo "=================================================================="
echo "Paso 3/3: Seleccionando mejor modelo..."
python experiments/edge_models/select_best_model.py

echo ""
echo "=================================================================="
echo "EXPERIMENTOS COMPLETADOS EXITOSAMENTE"
echo "=================================================================="
echo ""
echo "Archivos generados:"
echo "  - experiments/edge_models/best_edge_model.json"
echo "  - experiments/edge_models/comparison_results.csv"
echo ""
echo "Ver resultados en MLflow: http://localhost:5000"
echo ""

