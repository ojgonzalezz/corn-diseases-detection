# Entrenamiento de Modelos - Clasificación de Enfermedades del Maíz

Esta carpeta contiene los scripts para entrenar 4 arquitecturas de redes neuronales optimizadas para dispositivos móviles.

## Modelos Implementados

1. **MobileNetV3-Large**: Red neuronal convolucional eficiente de Google
2. **EfficientNet-Lite (B0)**: Arquitectura escalable y eficiente
3. **MobileViT**: Vision Transformer móvil con bloques de atención
4. **PMVT**: Plant Mobile Vision Transformer (optimizado para enfermedades de plantas)

## Configuración del Dataset

**División de datos:**
- Entrenamiento: 70% (10,332 imágenes)
- Validación: 15% (2,214 imágenes)
- Prueba: 15% (2,214 imágenes)

**Total:** 14,760 imágenes (3,690 por clase, perfectamente balanceadas)

## Hiperparámetros Comunes

```python
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10
REDUCE_LR_PATIENCE = 5
```

**Data Augmentation:**
- Rotación: ±20°
- Desplazamiento horizontal/vertical: ±20%
- Flip horizontal y vertical
- Zoom: ±20%

## Instalación

### Entorno Local

```bash
pip install -r requirements.txt
```

### Google Colab

Ver instrucciones detalladas en [COLAB_SETUP.md](COLAB_SETUP.md)

Resumen rápido:
```python
# 1. Clonar repo
!git clone https://github.com/ojgonzalezz/corn-diseases-detection.git
%cd corn-diseases-detection/entrenamiento_modelos

# 2. Instalar dependencias
!pip install -q -r requirements.txt

# 3. Subir dataset data_processed/ (manualmente o desde Drive)

# 4. Entrenar
!python train_mobilenetv3.py
```

## Uso

### Entrenar un modelo individual

```bash
# MobileNetV3
python train_mobilenetv3.py

# EfficientNet-Lite
python train_efficientnet.py

# MobileViT
python train_mobilevit.py

# PMVT
python train_pmvt.py
```

### Entrenar todos los modelos secuencialmente

```bash
python train_all_models.py
```

## Salidas Generadas

Para cada modelo se genera:

### Archivos .keras
- `models/{modelo}_best.keras` - Mejor modelo durante entrenamiento
- `models/{modelo}_final.keras` - Modelo final

### Logs
- `logs/{modelo}_training_log.json` - Log detallado en JSON
- `logs/{modelo}_training_log.txt` - Log legible en texto
- `logs/{modelo}_training_history.png` - Gráficos de accuracy y loss
- `logs/{modelo}_confusion_matrix.png` - Matriz de confusión

### MLflow
- Todos los experimentos se registran automáticamente en MLflow
- Ubicación: `mlruns/`

## Visualizar Resultados en MLflow

```bash
cd entrenamiento_modelos
mlflow ui --backend-store-uri mlruns/
```

Luego abrir en navegador: http://localhost:5000

## Estructura de Directorios

```
entrenamiento_modelos/
├── config.py                  # Configuración común
├── utils.py                   # Utilidades compartidas
├── train_mobilenetv3.py      # Entrenamiento MobileNetV3
├── train_efficientnet.py     # Entrenamiento EfficientNet
├── train_mobilevit.py        # Entrenamiento MobileViT
├── train_pmvt.py             # Entrenamiento PMVT
├── train_all_models.py       # Entrenar todos
├── requirements.txt          # Dependencias
├── models/                   # Modelos entrenados (.keras)
├── logs/                     # Logs y visualizaciones
└── mlruns/                   # Experimentos MLflow
```

## Requisitos del Sistema

**GPU:**
- GPU con soporte CUDA (recomendado)
- Memoria GPU: Mínimo 8GB recomendado
- Alternativamente, puede ejecutarse en Google Colab con GPU gratuita

**CPU/RAM:**
- RAM: Mínimo 16GB recomendado
- Espacio en disco: ~5GB para modelos y logs

## Información de los Logs

Cada log incluye:
- Hiperparámetros utilizados
- Métricas de entrenamiento (accuracy, loss)
- Métricas de validación
- Métricas de prueba
- Matriz de confusión
- Classification report (precision, recall, F1-score)
- Tiempo de entrenamiento

## Comparación de Modelos

Después de entrenar todos los modelos, puedes comparar:

1. **Test Accuracy**: Precisión en conjunto de prueba
2. **Tiempo de entrenamiento**: Eficiencia computacional
3. **Número de parámetros**: Tamaño del modelo
4. **Matrices de confusión**: Errores por clase

Usa MLflow UI para comparar métricas lado a lado.

## Notas Importantes

- Los modelos usan **transfer learning** con pesos de ImageNet
- Se aplica **fine-tuning** después del entrenamiento inicial (MobileNetV3 y EfficientNet)
- **Early stopping** detiene el entrenamiento si no hay mejora
- **ReduceLROnPlateau** reduce el learning rate automáticamente
- Todos los experimentos son **reproducibles** (RANDOM_SEED=42)

## Próximos Pasos

1. Analizar resultados en MLflow
2. Seleccionar el mejor modelo
3. Optimización de hiperparámetros del mejor modelo
4. Conversión a TensorFlow Lite para móviles
5. Despliegue en aplicación móvil
