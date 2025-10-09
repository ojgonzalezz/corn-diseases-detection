# Configuración para Google Colab

Este documento explica cómo ejecutar los scripts de entrenamiento en Google Colab.

## Pasos de Configuración

### 1. Clonar el Repositorio

En la primera celda de tu notebook de Colab:

```python
# Clonar repositorio
!git clone https://github.com/ojgonzalezz/corn-diseases-detection.git
%cd corn-diseases-detection
```

### 2. Verificar GPU

```python
# Verificar que tienes GPU habilitada
import tensorflow as tf
print("GPUs disponibles:", tf.config.list_physical_devices('GPU'))
```

**Importante**: Asegúrate de habilitar GPU en Colab:
- `Runtime` > `Change runtime type` > `Hardware accelerator` > `GPU` (T4 o L4)

### 3. Instalar Dependencias

```python
%cd entrenamiento_modelos
!pip install -q -r requirements.txt
```

### 4. Subir Dataset Procesado

Tienes dos opciones:

**Opción A: Desde Google Drive**

```python
from google.colab import drive
drive.mount('/content/drive')

# Copiar dataset desde tu Drive
!cp -r /content/drive/MyDrive/corn-diseases-detection/data_processed /content/corn-diseases-detection/
```

**Opción B: Subir archivo ZIP**

```python
from google.colab import files
import zipfile

# Subir archivo
uploaded = files.upload()  # Sube data_processed.zip

# Descomprimir
!unzip -q data_processed.zip -d /content/corn-diseases-detection/
```

### 5. Entrenar Modelos

**Entrenar un modelo individual:**

```python
# MobileNetV3
!python train_mobilenetv3.py

# EfficientNet-Lite
!python train_efficientnet.py

# MobileViT
!python train_mobilevit.py

# PMVT
!python train_pmvt.py
```

**Entrenar todos los modelos:**

```python
!python train_all_models.py
```

### 6. Descargar Resultados

```python
# Comprimir resultados
!zip -r resultados_entrenamiento.zip models/ logs/

# Descargar
from google.colab import files
files.download('resultados_entrenamiento.zip')
```

## Estructura de Archivos en Colab

```
/content/corn-diseases-detection/
├── data_processed/           # Dataset procesado (subir manualmente)
│   ├── Blight/
│   ├── Common_Rust/
│   ├── Gray_Leaf_Spot/
│   └── Healthy/
├── entrenamiento_modelos/
│   ├── train_*.py           # Scripts de entrenamiento
│   ├── config.py            # Configuración (detecta Colab automáticamente)
│   ├── utils.py             # Utilidades
│   ├── models/              # Modelos entrenados (generado)
│   ├── logs/                # Logs y visualizaciones (generado)
│   └── mlruns/              # Experimentos MLflow (generado)
```

## Configuración Automática

Los scripts detectan automáticamente si se ejecutan en Google Colab y ajustan:

- **Rutas**: Usa `/content/corn-diseases-detection` como base
- **GPU**: Configuración de memoria dinámica (óptima para Colab)
- **Directorios**: Se crean automáticamente si no existen

## Tiempo Estimado de Entrenamiento

Con GPU de Colab (T4):

- **MobileNetV3**: ~30-40 minutos (con fine-tuning)
- **EfficientNet-Lite**: ~35-45 minutos (con fine-tuning)
- **MobileViT**: ~40-50 minutos
- **PMVT**: ~35-45 minutos

**Total**: ~2.5-3 horas para los 4 modelos

## Monitorear Entrenamiento

Durante el entrenamiento verás:

```
============================================================
ENTRENAMIENTO MOBILENETV3-LARGE
============================================================
GPU configurada con crecimiento dinámico de memoria
GPUs disponibles: 1
GPU en uso: /physical_device:GPU:0

Creando generadores de datos...
Found 10332 images belonging to 4 classes.
Found 2214 images belonging to 4 classes.
Found 2214 images belonging to 4 classes.

Iniciando entrenamiento...
Epoch 1/50
323/323 [==============================] - 45s 120ms/step
...
```

## Ver Resultados con MLflow (Opcional)

En Colab no es práctico usar MLflow UI, pero puedes revisar los logs generados:

```python
# Ver logs JSON
import json
with open('logs/mobilenetv3_training_log.json', 'r') as f:
    log = json.load(f)
    print(f"Test Accuracy: {log['test_results']['test_accuracy']:.4f}")

# Ver log de texto
!cat logs/mobilenetv3_training_log.txt
```

## Guardar en Google Drive

Para no perder los resultados:

```python
from google.colab import drive
drive.mount('/content/drive')

# Copiar resultados a Drive
!cp -r models/ /content/drive/MyDrive/corn-diseases-detection/models_trained/
!cp -r logs/ /content/drive/MyDrive/corn-diseases-detection/logs_trained/
```

## Troubleshooting

### Error: "No such file or directory: data_processed"

Asegúrate de haber subido el dataset procesado correctamente.

```python
# Verificar que existe
!ls -la /content/corn-diseases-detection/data_processed
```

### Error: "Out of memory"

Reduce el batch size en `config.py`:

```python
# Editar temporalmente
import sys
sys.path.insert(0, '/content/corn-diseases-detection/entrenamiento_modelos')
import config
config.BATCH_SIZE = 16  # Reducir de 32 a 16
```

### Sesión de Colab Desconectada

Google Colab puede desconectarse después de ~12 horas de inactividad. Para entrenamientos largos:

1. Usa `train_all_models.py` para entrenar todos de una vez
2. Guarda checkpoints frecuentemente (ya configurado en los scripts)
3. Exporta resultados a Drive periódicamente

## Notas Importantes

1. **GPU Gratuita**: Colab ofrece GPU gratuita pero con límites de tiempo
2. **Persistencia**: Los archivos en `/content/` se eliminan al cerrar la sesión
3. **Dataset**: El dataset procesado (14,760 imágenes, ~400MB) debe subirse cada sesión
4. **Reproducibilidad**: Los scripts usan `RANDOM_SEED=42` para resultados reproducibles
