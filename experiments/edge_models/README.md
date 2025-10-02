#  Experimentos de Modelos Edge

Sistema completo de experimentación para seleccionar la mejor arquitectura liviana para edge computing.

---

##  Objetivo

Determinar cuál arquitectura es la **mejor** para deployment en dispositivos edge basándose en:

1. **Precisión global ≥ 85%**
2. **Recall por clase ≥ 0.80**
3. **Tamaño del modelo** (menor es mejor)
4. **Balance precisión/tamaño** (efficiency score)

---

##  Arquitecturas Evaluadas (4 Seleccionadas)

| Modelo | Parámetros | Tamaño | Características |
|--------|------------|--------|-----------------|
| **MobileNetV3Large** | ~5.4M | ~21MB | Balance óptimo tamaño/precisión |
| **EfficientNet-Lite B2** | ~10.1M | ~42MB | Máxima precisión manteniendo eficiencia |
| **MobileViT** | ~6.4M | ~25MB | Vision Transformer para móviles |
| **PMVT** | ~6M | ~24MB | Específico para enfermedades de plantas |

---

##  Uso Rápido (Docker)

### 1. Entrenar Todas las Arquitecturas

```bash
# Desde el root del proyecto
docker-compose --profile edge-experiments up
```

Esto ejecutará automáticamente:
1. Entrenamiento de los 4 modelos seleccionados
2. Comparación de resultados
3. Selección del mejor modelo
4. Generación de `best_edge_model.json`

**Tiempo estimado:** 2-3 horas (dependiendo de tu hardware)

### 2. Ver Resultados en MLflow

```bash
# Iniciar MLflow UI
docker-compose --profile mlflow up -d

# Acceder a:
open http://localhost:5000
```

Filtra por experimento: **edge_models_comparison**

---

##  Scripts Disponibles

### `train_edge_model.py`

Entrena un modelo específico.

```bash
# Ejemplo: Entrenar MobileNetV3Large
docker-compose run --rm training python experiments/edge_models/train_edge_model.py \
  --model MobileNetV3Large \
  --epochs 30 \
  --lr 0.001 \
  --dropout 0.3

# Ejemplo: Entrenar EfficientNetLiteB2
docker-compose run --rm training python experiments/edge_models/train_edge_model.py \
  --model EfficientNetLiteB2 \
  --epochs 30 \
  --lr 0.0008 \
  --dropout 0.25

# Con fine-tuning (recomendado para MobileViT y PMVT)
docker-compose run --rm training python experiments/edge_models/train_edge_model.py \
  --model PMVT \
  --epochs 30 \
  --fine-tune \
  --fine-tune-epochs 10
```

**Opciones:**
- `--model`: Modelo a entrenar (ver lista arriba)
- `--lr`: Learning rate (default: 0.001)
- `--dropout`: Dropout rate (default: 0.3)
- `--epochs`: Número de épocas (default: 30)
- `--batch-size`: Tamaño del batch (default: 32)
- `--fine-tune`: Activar fine-tuning
- `--fine-tune-epochs`: Épocas de fine-tuning (default: 10)

### `train_all_models.py`

Entrena todas las arquitecturas secuencialmente.

```bash
docker-compose run --rm training python experiments/edge_models/train_all_models.py
```

Cada modelo se entrena con hiperparámetros optimizados predefinidos.

### `compare_models.py`

Compara todos los modelos entrenados.

```bash
docker-compose run --rm training python experiments/edge_models/compare_models.py
```

Genera:
- Tabla comparativa en consola
- Archivo `comparison_results.csv`
- Análisis de mejores modelos por criterio

### `select_best_model.py`

Selecciona el mejor modelo y genera archivo de salida.

```bash
docker-compose run --rm training python experiments/edge_models/select_best_model.py
```

Genera: **`best_edge_model.json`** con toda la información para la siguiente fase.

---

##  Archivos de Salida

### `best_edge_model.json`

Archivo principal con el modelo seleccionado:

```json
{
  "selection_info": {
    "timestamp": "2025-10-02T16:30:00",
    "total_models_evaluated": 4,
    "models_meeting_requirements": 3,
    "selection_criteria": "Mejor balance precisión/tamaño"
  },
  
  "selected_model": {
    "name": "MobileNetV3Large",
    "run_id": "abc123...",
    "artifact_uri": "file:///app/models/mlruns/...",
    "model_file": "MobileNetV3Small_20251002_selected.keras"
  },
  
  "performance_metrics": {
    "test_accuracy": 0.8734,
    "min_recall": 0.8245,
    "recall_per_class": {
      "Blight": 0.8567,
      "Common_Rust": 0.8245,
      "Gray_Leaf_Spot": 0.8923,
      "Healthy": 0.9123
    },
    "meets_minimum_requirements": true
  },
  
  "model_characteristics": {
    "total_parameters": 2534567,
    "model_size_mb": 10.2,
    "efficiency_score": 0.2845,
    "suitable_for_edge": true
  },
  
  "training_configuration": {
    "learning_rate": 0.001,
    "dropout_rate": 0.3,
    "epochs_trained": 30,
    "batch_size": 32,
    "image_size": [224, 224],
    "num_classes": 4,
    "class_names": ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]
  },
  
  "all_models_comparison": [...]
}
```

### `comparison_results.csv`

Tabla con todos los modelos y métricas en formato CSV para análisis.

---

##  Interpretación de Resultados

### Efficiency Score

```
efficiency_score = (accuracy × min_recall) / log(size_mb + 1)
```

**Prioriza:**
- Alta precisión
- Alto recall (crítico para detección de enfermedades)
- Modelo pequeño

### Cumplimiento de Requisitos

 **CUMPLE** si:
- `test_accuracy ≥ 0.85` (85%)
- `min_recall ≥ 0.80` (80% en TODAS las clases)

 **NO CUMPLE** si falla alguno de los requisitos.

---

##  Workflow Completo

```bash
# 1. Entrenar todos los modelos
docker-compose --profile edge-experiments up

# 2. Comparar resultados
docker-compose run --rm training python experiments/edge_models/compare_models.py

# 3. Seleccionar mejor modelo
docker-compose run --rm training python experiments/edge_models/select_best_model.py

# 4. Ver en MLflow
docker-compose --profile mlflow up -d
open http://localhost:5000
```

---

##  Personalización

### Modificar Hiperparámetros

Edita `train_all_models.py` en la sección `EXPERIMENTS`:

```python
EXPERIMENTS = [
    {
        'name': 'MobileNetV3Small',
        'lr': 0.002,          # Cambiar learning rate
        'dropout': 0.4,       # Cambiar dropout
        'epochs': 50,         # Más épocas
        'fine_tune': True,    # Activar fine-tuning
    },
    # ...
]
```

### Agregar Nuevas Arquitecturas

1. Agregar loader en `src/builders/base_models.py`
2. Actualizar diccionario en `train_edge_model.py`
3. Agregar configuración en `train_all_models.py`

---

##  Próximos Pasos

Una vez seleccionado el mejor modelo (`best_edge_model.json`):

1. **Exportar a TensorFlow Lite**
   ```bash
   # Convertir a .tflite para deployment
   python scripts/convert_to_tflite.py
   ```

2. **Optimización adicional**
   - Quantization (INT8, FP16)
   - Pruning
   - Knowledge distillation

3. **Deployment**
   - Raspberry Pi
   - Jetson Nano
   - Mobile apps (Android/iOS)
   - Microcontroladores

---

** ¡Listo para encontrar el mejor modelo edge!**

