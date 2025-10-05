# MobileNetV3Large Inference Pipeline

Sistema de inferencia optimizado para detección de enfermedades del maíz usando exclusivamente MobileNetV3Large en dispositivos edge.

## Contenido del Directorio

- `run_pipeline.py`: **Script principal para ejecución automática completa (recomendado)**
- `config.yaml`: Configuración completa de hiperparámetros para MobileNetV3Large
- `convert_to_tflite.py`: Conversión del modelo a formato TensorFlow Lite con optimizaciones
- `validate_model.py`: Validación completa incluyendo accuracy y matriz de confusión
- `inference.py`: Pipeline de inferencia optimizado para producción

## Ejecución en Google Colab (Recomendado)

### Preparación Inicial en Colab

```python
# 1. Instalar todas las dependencias necesarias
!pip install tensorflow tensorflow-model-optimization pyyaml scikit-learn pillow matplotlib seaborn

# 2. Clonar o actualizar el repositorio
!git clone https://github.com/ojgonzalezz/corn-diseases-detection.git 2>/dev/null || (cd corn-diseases-detection && git pull)

# 3. Montar Google Drive para acceder a los datos
from google.colab import drive
drive.mount('/content/drive')
```

### Ejecución Automática Completa

```bash
# Ejecutar todo el pipeline automáticamente con tus datos en Drive
!cd corn-diseases-detection/mobilenetv3_inference && python run_pipeline.py --data-path /content/drive/MyDrive/corn-diseases-data
```

**Qué hace automáticamente:**
1. **Conversión**: Crea modelo MobileNetV3Large optimizado en TensorFlow Lite
2. **Validación**: Evalúa el modelo con datos de test y genera accuracy + matriz de confusión
3. **Inferencia**: Ejecuta demo de inferencia con múltiples muestras
4. **Reportes**: Genera archivos JSON con métricas y gráficos de resultados

**Archivos generados:**
- `models/mobilenetv3_large_optimized.tflite`: Modelo optimizado
- `results/validation_report.json`: Accuracy y métricas completas
- `results/confusion_matrix.png`: Matriz de confusión visual
- `results/inference_demo.json`: Resultados de inferencia

### Parámetros del Script Automático

```bash
python run_pipeline.py [opciones]

Opciones:
  --config CONFIG           Archivo de configuración (default: config.yaml)
  --data-path DATA_PATH     Ruta a los datos (default: ../data)
  --max-samples MAX_SAMPLES Máximo muestras para validación (default: 500)
  --inference-samples SAMPLES Muestras para demo de inferencia (default: 20)
```

### Ejemplos de Uso Personalizado

```bash
# Más muestras para validación más precisa
!cd corn-diseases-detection/mobilenetv3_inference && python run_pipeline.py --data-path /content/drive/MyDrive/corn-diseases-data --max-samples 1000

# Menos muestras para demo rápida
!cd corn-diseases-detection/mobilenetv3_inference && python run_pipeline.py --data-path /content/drive/MyDrive/corn-diseases-data --inference-samples 10
```

## Requisitos del Sistema

- **Python**: 3.8+
- **TensorFlow**: 2.10+
- **GPU**: Recomendado para conversión del modelo
- **Memoria RAM**: Mínimo 8GB
- **Espacio en disco**: 2GB libres

## Instalación

```bash
# Instalar dependencias
pip install tensorflow tensorflow-model-optimization pyyaml scikit-learn pillow matplotlib seaborn

# Verificar instalación
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
```

## Arquitectura del Modelo

### MobileNetV3Large Configurado

```yaml
model:
  name: "MobileNetV3Large"
  input_shape: [224, 224, 3]
  alpha: 1.0
  minimalistic: false
  include_top: true
  weights: "imagenet"
```

### Características Técnicas

- **Activación**: Hard-swish (h-swish)
- **Bloques SE**: Squeeze-and-Excitation con ratio 0.25
- **Estructura**: Inverted residual bottlenecks
- **Preprocesamiento**: Escala [-1, 1] específica para MobileNetV3

## Optimizaciones Implementadas

### Poda (Pruning)

```yaml
pruning:
  enabled: false  # Temporalmente deshabilitado por dependencias
  schedule_type: "PolynomialDecay"
  initial_sparsity: 0.0
  final_sparsity: 0.5
  begin_step: 0
  end_step: 100
  frequency: 10
```

### Cuantización INT8

```yaml
quantization:
  enabled: true
  optimization: ["DEFAULT"]
  representative_dataset_samples: 100
  inference_input_type: "tf.uint8"
  inference_output_type: "tf.uint8"
  supported_ops: ["TFLITE_BUILTINS_INT8"]
```

### Configuración de Inferencia

```yaml
inference:
  batch_size: 1
  num_threads: 1
  use_xnnpack_delegate: false
```

## Instalación

```bash
# Instalar dependencias
pip install tensorflow tensorflow-model-optimization pyyaml scikit-learn pillow matplotlib seaborn
```

## Uso Manual (Avanzado)

Si necesitas ejecutar pasos individuales o personalizar el proceso:

### 1. Conversión a TensorFlow Lite

```bash
python convert_to_tflite.py \
    --config config.yaml \
    --output models/mobilenetv3_large_optimized.tflite \
    --data-path ../data/train
```

### 2. Validación del Modelo

```bash
python validate_model.py \
    --config config.yaml \
    --model models/mobilenetv3_large_optimized.tflite \
    --test-data ../data/test \
    --max-samples 500 \
    --output results/validation_report.json
```

### 3. Inferencia

```bash
# Imagen individual
python inference.py \
    --config config.yaml \
    --model models/mobilenetv3_large_optimized.tflite \
    --image path/to/image.jpg

# Procesamiento por lotes
python inference.py \
    --config config.yaml \
    --model models/mobilenetv3_large_optimized.tflite \
    --batch \
    --num-samples 100
```

## Estructura de Datos Requerida

```
data/
├── train/
│   ├── Blight/
│   ├── Common_Rust/
│   ├── Gray_Leaf_Spot/
│   └── Healthy/
├── val/
└── test/
    ├── Blight/
    ├── Common_Rust/
    ├── Gray_Leaf_Spot/
    └── Healthy/
```

## Clases de Enfermedades

- **Blight**: Mancha bacteriana
- **Common_Rust**: Roya común
- **Gray_Leaf_Spot**: Mancha gris de la hoja
- **Healthy**: Hojas sanas

## Métricas Esperadas

- **Accuracy**: ≥85% en conjunto de test
- **Tamaño del modelo**: 5-7 MB (4-5x reducción)
- **Latencia**: <50ms por imagen en hardware edge
- **Uso de memoria**: <100MB durante inferencia

## Configuración de Producción

### Hardware Soportado

- Raspberry Pi 4/5
- NVIDIA Jetson Nano/Xavier
- Dispositivos móviles Android/iOS
- CPU multi-core con XNNPACK delegate

### Variables de Entorno

```bash
export TF_CPP_MIN_LOG_LEVEL=2  # Reducir logs de TensorFlow
export TF_ENABLE_ONEDNN_OPTS=1 # Optimizaciones adicionales
```

## Notas Técnicas

- El sistema está optimizado exclusivamente para MobileNetV3Large
- Los datos deben provenir únicamente del directorio `/data`
- La cuantización requiere un conjunto representativo de al menos 100 muestras
- Los modelos generados son compatibles con TensorFlow Lite Interpreter
- Se recomienda GPU para la conversión y validación inicial

## Solución de Problemas

### Error de Memoria

```bash
# Reducir batch size en config.yaml
inference:
  batch_size: 1
  num_threads: 1
```

### Modelo no Carga

- Verificar que el archivo .tflite existe
- Comprobar compatibilidad de TensorFlow Lite
- Validar que el modelo fue convertido correctamente

### Baja Precisión

- Verificar calidad de datos de entrenamiento
- Ajustar parámetros de cuantización si es necesario
- Considerar reentrenamiento con más epochs

---

**Nota**: Para la mayoría de los usuarios, se recomienda usar `python run_pipeline.py` para ejecutar todo automáticamente en lugar de los scripts individuales.
