# 🏗️ Guía de Configuración de Google Colab

**Configuración paso a paso para ejecutar el proyecto de detección de enfermedades del maíz en Google Colab.**

---

## 📋 Requisitos Previos

### **1. Google Drive**
- Cuenta activa de Google Drive
- Espacio suficiente para ~3GB de datos + modelos

### **2. Dataset**
- Dataset organizado en carpetas por clase:
  ```
  MyDrive/corn-diseases-data/
  ├── train/
  │   ├── Blight/         # ~964 imágenes
  │   ├── Common_Rust/    # ~964 imágenes
  │   ├── Gray_Leaf_Spot/ # ~964 imágenes
  │   └── Healthy/        # ~964 imágenes
  ├── val/                # ~716 imágenes
  └── test/               # ~722 imágenes
  ```

---

## 🚀 Configuración Inicial

### **Paso 1: Abrir Notebook en Colab**

**Opción A: Desde GitHub (Recomendado)**
1. Ve a: https://colab.research.google.com
2. File > Open notebook > GitHub
3. URL: `https://github.com/ojgonzalezz/corn-diseases-detection`
4. Selecciona: `colab_edge_models_training.ipynb`

**Opción B: Desde archivo local**
1. Descarga `colab_edge_models_training.ipynb`
2. Ve a: https://colab.research.google.com
3. File > Upload notebook

### **Paso 2: Configurar GPU**

1. Runtime > Change runtime type
2. **Hardware accelerator**: `GPU`
3. **GPU type**: `T4` (gratis, recomendado)
4. **Runtime shape**: `Standard`
5. Save

### **Paso 3: Verificar Recursos**

Ejecuta la primera celda del notebook:

```python
# Verificar GPU disponible
!nvidia-smi

import tensorflow as tf
print(f"\n✅ TensorFlow version: {tf.__version__}")
print(f"✅ GPU disponible: {tf.config.list_physical_devices('GPU')}")

# Verificar memoria
!free -h
```

**Salida esperada:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.105.17   Driver Version: 525.105.17   CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |                      |
|   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
| N/A   44C    P8     9W /  70W |      0MiB / 15360MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

✅ TensorFlow version: 2.15.0
✅ GPU disponible: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

              total        used        free      shared  buff/cache   available
Mem:           12.7Gi       1.1Gi       9.9Gi       1.2Mi       1.7Gi       11.3Gi
Swap:             0B          0B          0B
```

---

## 📁 Preparación de Datos

### **Paso 1: Subir Dataset a Google Drive**

1. Ve a [Google Drive](https://drive.google.com)
2. Crear carpeta: `corn-diseases-data`
3. Subir carpetas `train/`, `val/`, `test/` con la estructura correcta
4. **IMPORTANTE**: Asegurar que las clases sean exactamente:
   - `Blight`
   - `Common_Rust`
   - `Gray_Leaf_Spot`
   - `Healthy`

### **Paso 2: Verificar Estructura**

Ejecuta esta celda en el notebook:

```python
from google.colab import drive
drive.mount('/content/drive')

print("Verificando datos en Drive...")
!ls -la /content/drive/MyDrive/corn-diseases-data/

# Contar imágenes por clase
import os

def count_images(path):
    counts = {}
    for class_name in os.listdir(path):
        class_path = os.path.join(path, class_name)
        if os.path.isdir(class_path):
            counts[class_name] = len([f for f in os.listdir(class_path)
                                      if f.endswith(('.jpg', '.jpeg', '.png'))])
    return counts

print("\nDistribución de datos:")
for split in ['train', 'val', 'test']:
    path = f'/content/drive/MyDrive/corn-diseases-data/{split}'
    if os.path.exists(path):
        counts = count_images(path)
        total = sum(counts.values())
        print(f"{split.upper()}: {total} imágenes")
        for class_name, count in counts.items():
            print(f"  - {class_name}: {count}")
    else:
        print(f"{split.upper()}: ❌ NO ENCONTRADO")
```

---

## ⚙️ Configuración del Entorno

### **Paso 1: Instalar Dependencias**

El notebook instala automáticamente las dependencias. Si hay problemas:

```bash
# Instalar requirements.txt
!pip install -r requirements.txt

# Verificar instalación
import tensorflow as tf
import keras
import mlflow
import numpy as np
import pandas as pd

print("✅ Todas las dependencias instaladas")
print(f"   TensorFlow: {tf.__version__}")
print(f"   Keras: {keras.__version__}")
print(f"   MLflow: {mlflow.__version__}")
```

### **Paso 2: Configurar Variables de Entorno**

El notebook configura automáticamente las variables de entorno. Si necesitas modificar:

```python
import os

# Configurar TensorFlow para Colab
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Configurar GPU memory growth
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU memory growth habilitado para {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(e)
```

---

## 🚀 Ejecución del Entrenamiento

### **Ejecución Completa**

1. Runtime > Run all
2. Autorizar acceso a Google Drive cuando se solicite
3. Esperar 2-3 horas

### **Ejecución por Partes**

Si prefieres ejecutar paso a paso:

```python
# 1. Verificar GPU y recursos ✓
# 2. Montar Google Drive ✓
# 3. Copiar datos ✓
# 4. Verificar datos ✓
# 5. Configurar entorno ✓

# 6. Entrenar modelos (esto toma 2-3 horas)
!python experiments/edge_models/train_all_models.py

# 7. Comparar resultados
!python experiments/edge_models/compare_models.py

# 8. Seleccionar mejor modelo
!python experiments/edge_models/select_best_model.py
```

---

## 📊 Resultados y Descarga

### **Archivos Generados**

Después del entrenamiento, se generan:

- `experiments/edge_models/best_edge_model.json` - Mejor modelo seleccionado
- `experiments/edge_models/comparison_results.csv` - Comparación completa
- `models/exported/*.keras` - Modelos entrenados
- `models/mlruns/` - Experimentos MLflow

### **Descargar Resultados**

```bash
# Comprimir resultados
!zip -r edge_models_results.zip \
    experiments/edge_models/best_edge_model.json \
    experiments/edge_models/comparison_results.csv \
    models/exported/*.keras \
    models/exported/*.json \
    models/mlruns/

# Descargar a tu máquina local
from google.colab import files
files.download('edge_models_results.zip')
```

---

## 🔧 Solución de Problemas

### **Problema: GPU no disponible**
```python
# Verificar
!nvidia-smi

# Si no hay GPU, cambiar runtime:
# Runtime > Change runtime type > GPU > Save
```

### **Problema: Memoria insuficiente (SIGKILL)**
- Reducir batch_size en el código
- Usar GPU T4 en lugar de A100/V100
- Liberar memoria: `!free -h`

### **Problema: Google Drive no monta**
```python
from google.colab import drive
drive.mount('/content/drive')

# Si falla, autorizar en el enlace que aparece
```

### **Problema: Dependencias no instalan**
```python
# Instalar una por una
!pip install tensorflow==2.15.0
!pip install keras-tuner==1.4.7
!pip install mlflow==3.3.2
```

---

## 📞 Recursos Adicionales

- **Repositorio**: [corn-diseases-detection](https://github.com/ojgonzalezz/corn-diseases-detection)
- **Issues**: Reportar problemas en GitHub
- **Colab**: [Google Colab](https://colab.research.google.com)

---

**💡 Tip: Si el entrenamiento se interrumpe, puedes continuar desde donde quedó ejecutando las celdas faltantes.**
