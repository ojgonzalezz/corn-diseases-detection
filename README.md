# Detección de Enfermedades del Maíz

**Última actualización:** 2025-10-08 19:00:00

## Descripción del Proyecto

Este proyecto implementa un pipeline completo de análisis exploratorio de datos (EDA) y preprocesamiento para un dataset de imágenes de enfermedades del maíz, con el objetivo de preparar los datos para entrenamiento de modelos de clasificación.

## Clases del Dataset

El dataset contiene cuatro clases de enfermedades del maíz:
- **Blight** (Tizón)
- **Common Rust** (Roya Común)
- **Gray Leaf Spot** (Mancha Gris de la Hoja)
- **Healthy** (Saludable)

---

## 1. Análisis Exploratorio de Datos (EDA) - Dataset Original

**Script:** `EDA/eda_corn_diseases.py`

### Resultados del EDA Original

**Total de imágenes:** 13,381

**Distribución por clase:**
- Common_Rust: 3,690 (27.58%)
- Blight: 3,580 (26.75%)
- Healthy: 3,485 (26.04%)
- Gray_Leaf_Spot: 2,626 (19.62%)

**Balance de clases:** 71.17%

**Propiedades de las imágenes:**
- Dimensiones variables: 55 resoluciones diferentes detectadas
- Resolución más común: 256x256 (mayoría de las imágenes)
- Common_Rust con dimensiones muy variables (promedio 365x341 px)
- Tamaño promedio de archivo: 26.9 KB

**Estadísticas de píxeles por clase:**
- **Healthy:** Brillo 136.45, Canal G dominante (162.34)
- **Blight:** Brillo 112.01, Contraste bajo (37.04)
- **Gray_Leaf_Spot:** Brillo 106.77, Contraste medio (46.60)
- **Common_Rust:** Brillo 86.91, Contraste alto (63.95)

**Problemas identificados:**
1. Desbalance moderado entre clases (Gray_Leaf_Spot con menos imágenes)
2. Resoluciones no uniformes (55 diferentes)
3. Alta variabilidad en brillo entre clases

**Archivos generados:**
- `EDA/eda_corn_diseases.png` - 12 visualizaciones del dataset
- `EDA/samples_corn_diseases.png` - Muestras de cada clase
- `EDA/eda_report.txt` - Reporte completo con estadísticas

---

## 2. Preprocesamiento del Dataset

**Script:** `preprocessing/preprocess_corn_diseases.py`

### Transformaciones Aplicadas

1. **Normalización de dimensiones:** Todas las imágenes redimensionadas a 256x256 px
2. **Normalización de brillo:** Target de 120 con rango de ajuste (0.8, 1.2)
3. **Balanceo de clases:** Data augmentation para igualar todas las clases a 3,690 imágenes
4. **Técnicas de augmentation:**
   - Rotación (0°, 90°, 180°, 270°)
   - Flip horizontal y vertical
   - Ajustes aleatorios de brillo (0.9-1.1)
   - Ajustes aleatorios de contraste (0.9-1.1)

### Resultados del Preprocesamiento

**Dataset procesado ubicado en:** `data_processed/`

**Antes y después:**

| Clase | Original | Procesado | Incremento |
|-------|----------|-----------|------------|
| Blight | 3,580 | 3,690 | +110 |
| Common_Rust | 3,690 | 3,690 | 0 |
| Gray_Leaf_Spot | 2,626 | 3,690 | +1,064 |
| Healthy | 3,485 | 3,690 | +205 |
| **TOTAL** | **13,381** | **14,760** | **+1,379 (10.3%)** |

**Mejoras logradas:**
- Balance de clases: 71.17% → 100.00% (perfectamente balanceado)
- Uniformidad de dimensiones: 55 resoluciones → 1 resolución (256x256)
- Normalización de brillo: Desviación del target reducida a 7.39

**Archivos generados:**
- `preprocessing/preprocessing_results.png` - 8 visualizaciones comparativas
- `preprocessing/samples_comparison.png` - Comparación antes/después
- `preprocessing/preprocessing_report.txt` - Reporte detallado

---

## 3. Análisis Exploratorio de Datos (EDA) - Dataset Procesado

**Script:** `EDA/eda_processed_data.py`

### Resultados del EDA del Dataset Procesado

**Total de imágenes:** 14,760

**Distribución por clase:**
- Todas las clases: 3,690 imágenes (25.00% cada una)

**Balance de clases:** 100.00% (perfecto)

**Propiedades de las imágenes:**
- Dimensiones: 256x256 px (100% uniforme, varianza = 0.0)
- Tamaño promedio de archivo: 26.4 KB
- Modo de color: 100% RGB

**Estadísticas de píxeles normalizadas:**

| Clase | Brillo | Contraste |
|-------|--------|-----------|
| Healthy | 120.36 ± 13.31 | 37.32 ± 10.22 |
| Blight | 118.44 ± 5.07 | 39.50 ± 7.91 |
| Gray_Leaf_Spot | 116.38 ± 7.89 | 51.32 ± 13.43 |
| Common_Rust | 99.95 ± 17.57 | 74.29 ± 11.59 |

**Brillo global:** 113.78 ± 14.46 (desviación del target: 6.22)

**Verificación de calidad:**
- Balance de clases: PERFECTO (100.00%)
- Uniformidad de dimensiones: PERFECTA (256x256)
- Normalización de brillo: EXCELENTE (desviación < 10)

**Estado:** Dataset procesado listo para entrenamiento de modelos

**Archivos generados:**
- `eda_processed/eda_processed_data.png` - 12 visualizaciones del dataset procesado
- `eda_processed/samples_processed_data.png` - Muestras procesadas
- `eda_processed/eda_processed_report.txt` - Reporte completo con comparación

---

## Estructura del Proyecto

```
corn-diseases-detection/
├── data/                           # Dataset original (no incluido en repo)
├── data_processed/                 # Dataset procesado (no incluido en repo)
├── EDA/
│   ├── eda_corn_diseases.py       # Script EDA dataset original
│   ├── eda_processed_data.py      # Script EDA dataset procesado
│   ├── eda_corn_diseases.png      # Visualizaciones originales
│   ├── samples_corn_diseases.png  # Muestras originales
│   └── eda_report.txt             # Reporte dataset original
├── preprocessing/
│   ├── preprocess_corn_diseases.py     # Script de preprocesamiento
│   ├── preprocessing_results.png       # Visualizaciones de preprocesamiento
│   ├── samples_comparison.png          # Comparación antes/después
│   └── preprocessing_report.txt        # Reporte de preprocesamiento
└── eda_processed/
    ├── eda_processed_data.png          # Visualizaciones dataset procesado
    ├── samples_processed_data.png      # Muestras procesadas
    └── eda_processed_report.txt        # Reporte dataset procesado
```

---

## Uso de los Scripts

### 1. Ejecutar EDA del dataset original
```bash
python EDA/eda_corn_diseases.py
```

### 2. Ejecutar preprocesamiento
```bash
python preprocessing/preprocess_corn_diseases.py
```
Genera el dataset procesado en `data_processed/`

### 3. Ejecutar EDA del dataset procesado
```bash
python EDA/eda_processed_data.py
```

---

## Dependencias

```
numpy
pandas
matplotlib
seaborn
pillow
tqdm
```

Instalación:
```bash
pip install numpy pandas matplotlib seaborn pillow tqdm
```

---

## 4. Entrenamiento de Modelos

**Directorio:** `entrenamiento_modelos/`

### Modelos Implementados
- MobileNetV3-Large
- EfficientNet-Lite (B0)
- MobileViT
- PMVT (Plant Mobile Vision Transformer)

### Hiperparámetros
- Épocas: 20
- Batch size: 64
- Learning rate: 0.001
- Early stopping: 10 épocas
- Tiempo estimado: ~40-60 minutos (4 modelos con GPU T4)

### Ejecución en Google Colab

**Preparación (una sola vez):**
1. Habilita GPU: `Runtime` > `Change runtime type` > `Hardware accelerator` > `GPU`
2. Sube `data_processed/` a Google Drive en: `Mi unidad/data_processed/`

**Opción 1 - Script Automático (Recomendado):**
```python
!wget -q https://raw.githubusercontent.com/ojgonzalezz/corn-diseases-detection/pipe/entrenamiento_modelos/setup_and_train.py
!python setup_and_train.py
```

**Opción 2 - Ejecución Manual:**
```python
# 1. Montar Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Clonar repositorio (rama pipe)
!git clone -b pipe https://github.com/ojgonzalezz/corn-diseases-detection.git
%cd corn-diseases-detection/entrenamiento_modelos

# 3. Instalar dependencias
!pip install -q -r requirements.txt

# 4. Entrenar todos los modelos
!python train_all_models.py
```

**Salidas (guardadas automáticamente en Drive):**
- Modelos: `Mi unidad/corn-diseases-detection/models/`
- Logs: `Mi unidad/corn-diseases-detection/logs/`
- MLflow: `Mi unidad/corn-diseases-detection/mlruns/`

Ver más detalles en: `entrenamiento_modelos/README.md`

---

## Autor

**afelipfo**
- Email: afelipeflorezo@gmail.com

---

## Notas Importantes

- Los datasets (`data/` y `data_processed/`) no están incluidos en el repositorio por su tamaño
- Todos los scripts generan reportes detallados y visualizaciones automáticamente
- El dataset procesado está completamente balanceado y normalizado, listo para entrenamiento
