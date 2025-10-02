# MLflow Tracking - Experimentos Edge Computing

## Configuración de Tracking

**Experimento MLflow:** `edge_models_comparison`  
**Ubicación:** `models/mlruns/`  
**UI:** http://localhost:5000

---

## Métricas Trackeadas por Modelo

### **Por cada modelo se registra:**

#### **Parámetros (mlflow.log_param):**
```
- model_name              # Ej: "MobileNetV3Large"
- learning_rate           # Ej: 0.001
- dropout_rate            # Ej: 0.3
- epochs                  # Ej: 30
- batch_size              # Ej: 32
- fine_tune               # True/False
- image_size              # (224, 224)
- backbone_params         # Total parámetros del backbone
- backbone_size_mb        # Tamaño en MB
```

#### **Métricas de Entrenamiento (por época):**
```
- train_loss              # Loss en training
- train_accuracy          # Accuracy en training
- val_loss                # Loss en validation
- val_accuracy            # Accuracy en validation
```

#### **Métricas Finales en Test:**
```
- test_loss               # Loss final
- test_accuracy           # Accuracy final
- min_recall              # Recall mínimo entre clases
- recall_Blight           # Recall clase Blight
- recall_Common_Rust      # Recall clase Common_Rust
- recall_Gray_Leaf_Spot   # Recall clase Gray_Leaf_Spot
- recall_Healthy          # Recall clase Healthy
- meets_requirements      # 1.0 si cumple, 0.0 si no
```

---

## Dashboard Comparativo en MLflow

### **Cómo Ver el Dashboard:**

```bash
# 1. Iniciar MLflow UI
docker-compose --profile mlflow up -d

# 2. Abrir en navegador
open http://localhost:5000

# 3. Seleccionar experimento "edge_models_comparison"
```

### **Vistas Disponibles:**

#### **1. Tabla Comparativa**
Verás una tabla con:
- Nombre del run (modelo + timestamp)
- test_accuracy
- min_recall
- backbone_size_mb
- meets_requirements

**Ordenar por:** Click en columna "test_accuracy" para ver el mejor

#### **2. Gráficas de Entrenamiento**
Para cada modelo:
- Curva de train_accuracy vs épocas
- Curva de val_accuracy vs épocas
- Curva de loss
- Comparación entre modelos

#### **3. Parallel Coordinates Plot**
Visualiza relación entre:
- learning_rate
- dropout_rate
- test_accuracy
- min_recall
- backbone_size_mb

#### **4. Scatter Plot**
Eje X: backbone_size_mb  
Eje Y: test_accuracy  
Tamaño punto: min_recall

**Identifica visualmente:** Modelo en esquina superior izquierda = mejor (alta precisión, bajo tamaño)

---

## Filtros Útiles en MLflow

### **Filtrar modelos que cumplen requisitos:**
```
metrics.meets_requirements = 1.0
```

### **Filtrar por accuracy mínima:**
```
metrics.test_accuracy >= 0.85
```

### **Filtrar por tamaño máximo:**
```
params.backbone_size_mb <= 30
```

### **Ordenar por eficiencia:**
```
ORDER BY metrics.test_accuracy DESC
```

---

## Comparación Lado a Lado

### **En MLflow UI:**

1. Selecciona múltiples runs (checkbox)
2. Click en "Compare"
3. Verás tabla comparativa con:
   - Todos los parámetros lado a lado
   - Todas las métricas lado a lado
   - Diferencias resaltadas

---

## Ejemplo de Vista Comparativa

```
╔═══════════════════╦══════════════╦═══════════╦══════════╦═══════════╗
║ Model             ║ Accuracy     ║ Min Recall║ Size (MB)║ Cumple    ║
╠═══════════════════╬══════════════╬═══════════╬══════════╬═══════════╣
║ MobileNetV3Large  ║ 87.34%       ║ 82.45%    ║ 21.0     ║ SI        ║
║ EfficientNetLiteB2║ 89.12%       ║ 85.67%    ║ 42.0     ║ SI        ║
║ MobileViT         ║ 88.45%       ║ 83.89%    ║ 25.0     ║ SI        ║
║ PMVT              ║ 90.23%       ║ 87.12%    ║ 24.0     ║ SI        ║
╚═══════════════════╩══════════════╩═══════════╩══════════╩═══════════╝
```

---

## Exportar Resultados

Desde MLflow UI:

1. **Descargar CSV:**
   - Selecciona runs
   - Click "Download CSV"

2. **Descargar Modelos:**
   - Click en run
   - Tab "Artifacts"
   - Download model

3. **Ver Logs:**
   - Click en run
   - Tab "Text"
   - Ver output completo

---

## Acceso Programático

```python
import mlflow
from mlflow.tracking import MlflowClient

# Conectar
mlflow.set_tracking_uri("file:///path/to/mlruns")
client = MlflowClient()

# Obtener experimento
experiment = client.get_experiment_by_name("edge_models_comparison")

# Obtener todos los runs
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.test_accuracy DESC"]
)

# Ver mejor modelo
best = runs[0]
print(f"Mejor modelo: {best.data.params['model_name']}")
print(f"Accuracy: {best.data.metrics['test_accuracy']:.2%}")
```

---

## IMPORTANTE: Monitoreo Durante Entrenamiento

**En otra terminal:**

```bash
# 1. Iniciar MLflow UI
docker-compose --profile mlflow up -d

# 2. Ver logs en tiempo real
docker-compose logs -f edge-experiments

# 3. Abrir MLflow
open http://localhost:5000
```

**Refresca la página de MLflow** cada 5-10 minutos para ver:
- Nuevos runs apareciendo
- Métricas actualizándose en tiempo real
- Progreso de cada modelo

---

El sistema está LISTO para trackear todo automáticamente.

