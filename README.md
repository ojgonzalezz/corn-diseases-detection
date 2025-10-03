# 🌽 Detección de Enfermedades del Maíz - Edge Models

Sistema de Deep Learning para clasificación de enfermedades en hojas de maíz utilizando arquitecturas livianas optimizadas para edge computing.

---

## 📋 Resumen del Proyecto

Pipeline de Deep Learning para diagnóstico automático de enfermedades comunes en hojas de maíz usando **4 arquitecturas edge** entrenadas en **Google Colab con GPU gratuita**.

**Características Principales:**
- 🚀 **Entrenamiento en Google Colab** con GPU T4 gratuita
- 🎯 **4 Arquitecturas Edge** optimizadas para dispositivos móviles
- 📊 **Tracking con MLflow** para comparación de experimentos
- 🧪 **Suite completa de tests** automatizados
- 📱 **Modelos livianos** listos para deployment en edge

---

## 🎯 Clases de Enfermedades

El modelo clasifica 4 categorías:

1. **Blight** (Tizón)
2. **Common_Rust** (Roya Común)
3. **Gray_Leaf_Spot** (Mancha Gris)
4. **Healthy** (Saludable)

---

## 📁 Estructura del Proyecto

```
corn-diseases-detection/
├── data/                          # Dataset (ignorado por git)
│   ├── train/                     # 3,856 imágenes (balanceado)
│   ├── val/                       # 716 imágenes (estratificado)
│   └── test/                      # 722 imágenes (estratificado)
│
├── src/                           # Código fuente
│   ├── adapters/                  # Cargadores de datos
│   ├── builders/                  # Constructores de modelos edge
│   ├── core/                      # Configuración central
│   ├── pipelines/                 # Pipelines ML (preprocess, infer)
│   └── utils/                     # Utilidades
│
├── tests/                         # Suite de tests (10 archivos)
│
├── experimentation/               # Scripts EDA y notebooks exploratorios
│
├── experiments/                   # 🎯 Experimentos edge computing
│   └── edge_models/               # Entrenamiento arquitecturas livianas
│       ├── train_edge_model.py    # Script principal de entrenamiento
│       ├── train_all_models.py    # Orquestador de experimentos
│       ├── compare_models.py      # Comparación de resultados
│       ├── select_best_model.py   # Selección del mejor modelo
│       ├── run_all_experiments.sh # Script de automatización
│       ├── README.md              # Documentación detallada
│       └── MLFLOW_TRACKING.md     # Guía de MLflow
│
├── models/                        # Modelos entrenados (ignorado por git)
│   ├── exported/                  # Modelos finales (.keras)
│   └── mlruns/                    # Tracking MLflow
│
├── colab_edge_models_training.ipynb  # 🚀 Notebook principal para Colab
├── COLAB_SETUP.md                    # Guía de configuración de Colab
├── requirements.txt                  # Dependencias Python (optimizado para Colab)
└── README.md                         # Este archivo
```

---

## 🚀 Inicio Rápido (Google Colab)

### **Paso 1: Preparar Datos en Google Drive**

Sube tu carpeta `data/` a Google Drive con esta estructura:

```
MyDrive/corn-diseases-data/
├── train/
│   ├── Blight/
│   ├── Common_Rust/
│   ├── Gray_Leaf_Spot/
│   └── Healthy/
├── val/
│   └── [mismas clases]
└── test/
    └── [mismas clases]
```

### **Paso 2: Abrir Notebook en Colab**

**Opción A: Desde GitHub (Recomendado)**
1. Ve a: https://colab.research.google.com
2. File > Open notebook > GitHub
3. URL: `https://github.com/ojgonzalezz/corn-diseases-detection`
4. Selecciona: `colab_edge_models_training.ipynb`

**Opción B: Desde archivo local**
1. Descarga `colab_edge_models_training.ipynb`
2. Ve a: https://colab.research.google.com
3. File > Upload notebook

### **Paso 3: Configurar GPU**

1. Runtime > Change runtime type
2. Hardware accelerator: **GPU**
3. GPU type: **T4** (gratis)
4. Save

### **Paso 4: Ejecutar**

1. Runtime > Run all
2. Autoriza acceso a Google Drive cuando se solicite
3. ☕ Espera 2-3 horas

📖 **Guía detallada:** Ver `COLAB_SETUP.md`

---

## 🏗️ Arquitecturas Edge Evaluadas

El proyecto entrena y compara **4 arquitecturas** optimizadas para edge computing:

| Modelo | Parámetros | Tamaño | Características |
|--------|------------|--------|-----------------|
| **MobileNetV3Large** | ~5.4M | ~21MB | Balance óptimo tamaño/precisión |
| **EfficientNetLiteB2** | ~10.1M | ~42MB | Máxima precisión manteniendo eficiencia |
| **MobileViT** | ~6.4M | ~25MB | Vision Transformer móvil + fine-tuning |
| **PMVT** | ~6M | ~24MB | Específico para plantas + fine-tuning |

### **Criterios de Selección:**
- ✅ Precisión global ≥ 85%
- ✅ Recall por clase ≥ 0.80
- ✅ Mejor balance precisión/tamaño
- ✅ Tamaño ≤ 50MB para edge deployment

---

## 📊 Proceso de Entrenamiento

### **1. Entrenamiento Automático**

El notebook de Colab ejecuta automáticamente:

```python
# 1. Entrenar MobileNetV3Large (30 épocas)
# 2. Entrenar EfficientNetLiteB2 (30 épocas)
# 3. Entrenar MobileViT (30 épocas + 10 fine-tuning)
# 4. Entrenar PMVT (30 épocas + 10 fine-tuning)
# 5. Comparar resultados
# 6. Seleccionar mejor modelo
# 7. Generar best_edge_model.json
```

### **2. Tracking con MLflow**

Todas las métricas se registran automáticamente:
- Hiperparámetros
- Accuracy y loss por época
- Recall por clase
- Tamaño del modelo
- Tiempo de entrenamiento

### **3. Salida del Proceso**

**Archivos generados:**
- `experiments/edge_models/best_edge_model.json` - Mejor modelo seleccionado
- `experiments/edge_models/comparison_results.csv` - Comparación completa
- `models/exported/*.keras` - Modelos entrenados
- `models/mlruns/` - Experimentos MLflow

---

## ⏱️ Tiempos de Entrenamiento

| Plataforma | GPU | Tiempo Total |
|------------|-----|--------------|
| **Google Colab** | T4 (16GB) | **2-3 horas** ⚡ |
| CPU Local | - | 20-30 horas 🐌 |

---

## 📦 Archivos de Salida

### **best_edge_model.json**

Archivo principal con el modelo seleccionado:

```json
{
  "selected_model": {
    "name": "MobileNetV3Large",
    "run_id": "abc123...",
    "model_file": "MobileNetV3Large_20251002_selected.keras"
  },
  "performance_metrics": {
    "test_accuracy": 0.8734,
    "min_recall": 0.8245,
    "recall_per_class": {...}
  },
  "model_characteristics": {
    "total_parameters": 5400000,
    "model_size_mb": 21.0,
    "suitable_for_edge": true
  }
}
```

---

## 🧪 Testing

### **Ejecutar Tests Localmente**

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar todos los tests
pytest tests/ -v

# Tests específicos
pytest tests/test_builders.py -v

# Tests sin módulos lentos
pytest tests/ -m "not slow" -v
```

### **Cobertura de Tests**

El proyecto incluye **10 archivos de tests** con **~90% de cobertura**:
- `test_infer.py` - Pipeline de inferencia
- `test_preprocess.py` - Preprocesamiento
- `test_augmentation.py` - Augmentación de datos
- `test_config.py` - Sistema de configuración
- `test_builders.py` - Constructores de modelos
- `test_data_loader.py` - Carga de datos
- `test_logger.py` - Sistema de logging
- `test_paths.py` - Gestión de rutas
- Y más...

---

## 🔧 Configuración

### **Variables de Entorno**

El proyecto usa un archivo `.env` para configuración. Todas las variables tienen valores por defecto en `src/core/.env_example`.

**Variables Principales:**

| Variable | Valor por Defecto | Descripción |
|----------|-------------------|-------------|
| `IMAGE_SIZE` | `(224, 224)` | Dimensiones de entrada |
| `NUM_CLASSES` | `4` | Número de clases |
| `BATCH_SIZE` | `32` | Tamaño del batch |
| `MAX_EPOCHS` | `30` | Épocas máximas |
| `BALANCE_STRATEGY` | `oversample` | Estrategia de balanceo |

Para personalizar, edita `src/core/.env`

---

## 📱 Deployment en Edge

### **Próximos Pasos**

Una vez seleccionado el mejor modelo:

1. **Exportar a TensorFlow Lite**
   ```python
   import tensorflow as tf
   
   # Cargar modelo
   model = tf.keras.models.load_model('models/exported/best_model.keras')
   
   # Convertir a TFLite
   converter = tf.lite.TFLiteConverter.from_keras_model(model)
   tflite_model = converter.convert()
   
   # Guardar
   with open('model.tflite', 'wb') as f:
       f.write(tflite_model)
   ```

2. **Optimización Adicional**
   - Quantization (INT8, FP16)
   - Pruning
   - Knowledge distillation

3. **Deployment**
   - Raspberry Pi
   - Jetson Nano
   - Mobile apps (Android/iOS)
   - Microcontroladores

---


## 🛠️ Desarrollo Local

### **Instalación**

```bash
# Clonar repositorio
git clone https://github.com/ojgonzalezz/corn-diseases-detection.git
cd corn-diseases-detection

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### **Estructura de Datos**

```
data/
├── train/
│   ├── Blight/
│   ├── Common_Rust/
│   ├── Gray_Leaf_Spot/
│   └── Healthy/
├── val/
│   └── [mismas clases]
└── test/
    └── [mismas clases]
```

---

## 📖 Documentación Adicional

- **Guía de Colab:** `COLAB_SETUP.md`
- **Experimentos Edge:** `experiments/edge_models/README.md`
- **MLflow Tracking:** `experiments/edge_models/MLFLOW_TRACKING.md`

---

## 🤝 Contribuir

1. Fork el repositorio
2. Crea una rama (`git checkout -b feature/nueva-funcionalidad`)
3. Haz tus cambios
4. Ejecuta tests: `pytest tests/`
5. Commit (`git commit -m 'feat: nueva funcionalidad'`)
6. Push (`git push origin feature/nueva-funcionalidad`)
7. Abre un Pull Request

---

## 📄 Licencia

Este proyecto está bajo la licencia MIT.

---

## 📞 Soporte

- **Issues:** [GitHub Issues](https://github.com/ojgonzalezz/corn-diseases-detection/issues)
- **Repository:** [ojgonzalezz/corn-diseases-detection](https://github.com/ojgonzalezz/corn-diseases-detection)

---

## ⭐ Ventajas del Proyecto

✅ **Sin Docker** - No necesitas configurar contenedores  
✅ **GPU Gratis** - Usa Google Colab con T4 gratuita  
✅ **Rápido** - 2-3 horas vs 20-30 horas en CPU  
✅ **Simple** - Notebook listo para ejecutar  
✅ **Completo** - Tracking, comparación, selección automática  
✅ **Edge-Ready** - Modelos optimizados para dispositivos móviles  

---

**🚀 Desarrollado con Transfer Learning y Google Colab para máxima accesibilidad**
