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
BATCH_SIZE = 64
EPOCHS = 20
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

### Google Colab (Recomendado)

**Preparación Inicial (una sola vez)**:
1. Habilita GPU: `Runtime` > `Change runtime type` > `Hardware accelerator` > `GPU`
2. Sube `data_processed/` a tu Google Drive en: `Mi unidad/data_processed/`

**🚨 DIAGNÓSTICO ANTES DE EMPEZAR (IMPORTANTE)**:
```python
# Ejecuta este script primero para verificar que todo esté configurado correctamente
!wget -q https://raw.githubusercontent.com/ojgonzalezz/corn-diseases-detection/pipe/entrenamiento_modelos/diagnostic.py
!python diagnostic.py
```

**Ejecución Automática (Opción 1 - Recomendada)**:
```python
!wget -q https://raw.githubusercontent.com/ojgonzalezz/corn-diseases-detection/pipe/entrenamiento_modelos/setup_and_train.py
!python setup_and_train.py
```

El script automático mejorado incluye:
- ✓ Verificación robusta de GPU con timeouts
- ✓ Montaje de Google Drive con reintentos automáticos
- ✓ Verificación del dataset con espera inteligente
- ✓ Timeouts en cada paso para evitar cuelgues
- ✓ Mejor manejo de errores y recuperación
- ✓ Timeout total de 4 horas para todo el proceso

**Ejecución Manual (Opción 2)**:
```python
# 1. Montar Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Clonar repo (rama pipe)
!git clone -b pipe https://github.com/ojgonzalezz/corn-diseases-detection.git
%cd corn-diseases-detection/entrenamiento_modelos

# 3. Instalar dependencias
!pip install -q -r requirements.txt

# 4. Entrenar todos los modelos
!python train_all_models.py
```

Los scripts detectan Colab automáticamente y:
- ✓ Verifican que GPU esté habilitada (obligatorio)
- ✓ Leen dataset desde `Mi unidad/data_processed/`
- ✓ Guardan modelos en `Mi unidad/corn-diseases-detection/models/`
- ✓ Guardan logs en `Mi unidad/corn-diseases-detection/logs/`

**Tiempo estimado**: ~40-60 minutos para los 4 modelos con GPU T4
**Nota**: El script mejorado tiene timeouts para evitar cuelgues infinitos

## Uso

### 🔧 Diagnóstico del Entorno

Antes de empezar, verifica que todo esté configurado correctamente:

```bash
# Script de diagnóstico completo
python diagnostic.py

# Solo listar modelos disponibles
python train_single_model.py --list
```

### Entrenar un modelo individual

```bash
# Usando script específico
python train_mobilenetv3.py
python train_efficientnet.py
python train_mobilevit.py
python train_pmvt.py

# O usando el script unificado (útil para testing)
python train_single_model.py mobilenetv3
python train_single_model.py efficientnet
python train_single_model.py mobilevit
python train_single_model.py pmvt
```

### Entrenar todos los modelos secuencialmente

```bash
# Versión mejorada con timeouts y mejor manejo de errores
python train_all_models.py
```

**Características de la versión mejorada:**
- ⏰ **Timeout por modelo**: 2 horas máximo por modelo
- 🔄 **Continúa si falla**: Si un modelo falla, continúa con el siguiente
- 📊 **Resumen detallado**: Genera archivo de resumen al finalizar
- 🧠 **Liberación de memoria**: Pausas entre modelos para liberar GPU

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

## 🔧 Solución de Problemas

### El script se queda atascado (stuck)

**Síntomas:**
- El script deja de mostrar progreso
- No hay error visible
- Parece "congelado"

**Soluciones:**

1. **Ejecuta el diagnóstico primero:**
   ```bash
   python diagnostic.py
   ```
   Esto te dirá exactamente dónde está el problema.

2. **Verifica los puntos comunes de fallo:**
   - ❌ **Google Drive no montado**: Ejecuta `from google.colab import drive; drive.mount('/content/drive')`
   - ❌ **Dataset no encontrado**: Verifica que `data_processed/` esté en la raíz de tu Drive
   - ❌ **GPU no habilitada**: Ve a `Runtime > Change runtime type > GPU`
   - ❌ **Dependencias faltantes**: Ejecuta `pip install -r requirements.txt`

3. **Si el entrenamiento se queda atascado:**
   - Usa `python train_single_model.py mobilenetv3` para probar un modelo individual
   - Los nuevos scripts tienen timeouts de 2 horas por modelo
   - Si un modelo falla, los demás continúan automáticamente

### Errores Comunes

**"No se detectó GPU"**
```bash
# En Google Colab:
# Runtime > Change runtime type > Hardware accelerator > GPU > Save
# Luego reconecta la sesión
```

**"Dataset no encontrado"**
```
Asegúrate de que la carpeta esté en:
Mi unidad/data_processed/
  ├── Blight/
  ├── Common_Rust/
  ├── Gray_Leaf_Spot/
  └── Healthy/
```

**"Error de memoria GPU"**
- Reduce `BATCH_SIZE` en `config.py`
- Reinicia la sesión de Colab
- Usa `GPU_MEMORY_LIMIT = 4096` en config.py

**"Timeout alcanzado"**
- Los nuevos scripts tienen timeouts seguros
- Si un paso toma demasiado tiempo, revisa tu conexión a internet
- Para Drive lento, el script ahora reintenta automáticamente

### Logs de Depuración

Todos los scripts generan logs detallados. Revisa:
- `logs/` - Logs de entrenamiento por modelo
- `entrenamiento_resumen.txt` - Resumen completo
- MLflow UI para métricas detalladas

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
