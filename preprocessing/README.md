# Preprocesamiento de Dataset - Corn Diseases

Este directorio contiene el script y resultados del proceso de preprocesamiento aplicado al dataset original de enfermedades del maiz.

## Contenido

### Scripts

- **preprocess_corn_diseases.py**: Script principal de preprocesamiento y normalizacion

### Resultados

- **preprocessing_report.txt**: Reporte detallado del proceso de transformacion
- **preprocessing_results.png**: Graficos comparativos de metricas antes y despues
- **samples_comparison.png**: Comparacion visual de imagenes originales vs procesadas

## Objetivo

Transformar el dataset original en un conjunto de datos optimizado para entrenamiento de modelos de clasificacion, aplicando normalizacion dimensional, balanceo de clases y estandarizacion de brillo.

## Transformaciones Aplicadas

### 1. Normalizacion de Dimensiones

**Objetivo**: Uniformizar el tamano de todas las imagenes

- **Target**: 256x256 px
- **Metodo**: Redimensionamiento con interpolacion bilineal
- **Resultado**: 100% de imagenes con dimensiones uniformes

**Antes**:
- 55 resoluciones diferentes
- Common_Rust con dimensiones variables (promedio 365x341 px)

**Despues**:
- 1 resolucion uniforme (256x256 px)
- Varianza de dimensiones: 0.0000

### 2. Normalizacion de Brillo

**Objetivo**: Reducir variabilidad de iluminacion entre imagenes

- **Target**: Brillo promedio de 120
- **Rango permitido**: Factor de ajuste entre 0.8 y 1.2
- **Metodo**: Ajuste proporcional de intensidad de pixeles

**Resultados**:
- Brillo global: 112.61 (desviacion: 15.32)
- Desviacion del target: 7.39
- Reduccion de variabilidad: 45.2% respecto al original

### 3. Balanceo de Clases

**Objetivo**: Eliminar sesgo por desbalance de clases

- **Target**: 3,690 imagenes por clase
- **Metodo**: Data augmentation para clases minoritarias

**Transformaciones de Augmentation**:
- Rotaciones aleatorias: -20 a +20 grados
- Flips horizontales y verticales
- Ajustes de brillo: -10 a +10
- Ajustes de contraste: -10 a +10

**Resultados por Clase**:

| Clase | Original | Procesado | Diferencia |
|-------|----------|-----------|------------|
| Blight | 3,580 | 3,690 | +110 |
| Common_Rust | 3,690 | 3,690 | 0 |
| Gray_Leaf_Spot | 2,626 | 3,690 | +1,064 |
| Healthy | 3,485 | 3,690 | +205 |

**Total**: 13,381 → 14,760 imagenes (+10.3%)

**Balance**: 71.17% → 100.00%

## Estadisticas del Dataset Procesado

### Caracteristicas Generales

- Total de imagenes: 14,760
- Imagenes por clase: 3,690 (25% cada una)
- Balance de clases: 100.00%
- Dimensiones: 256x256 px (uniformes)
- Tamano promedio: ~26.4 KB

### Estadisticas de Brillo por Clase

| Clase | Brillo (media ± std) | Target Deviation |
|-------|---------------------|------------------|
| Blight | 118.52 ± 6.43 | -1.48 |
| Common_Rust | 95.66 ± 16.90 | -24.34 |
| Gray_Leaf_Spot | 116.07 ± 6.40 | -3.93 |
| Healthy | 120.17 ± 13.48 | +0.17 |

### Estadisticas de Contraste por Clase

| Clase | Contraste (media ± std) |
|-------|------------------------|
| Blight | 39.27 ± 7.18 |
| Common_Rust | 75.78 ± 8.04 |
| Gray_Leaf_Spot | 51.71 ± 14.15 |
| Healthy | 37.69 ± 10.48 |

### Estadisticas RGB por Clase

**Blight**: R=121.60, G=129.63, B=104.32

**Common_Rust**: R=100.55, G=108.29, B=78.15

**Gray_Leaf_Spot**: R=121.24, G=128.25, B=98.72

**Healthy**: R=112.00, G=144.34, B=104.19

## Validaciones de Calidad

El dataset procesado paso todas las validaciones:

- [OK] Todas las imagenes tienen dimensiones uniformes: 256x256 px
- [OK] Balance de clases: 100.00%
- [OK] Brillo normalizado: desviacion=7.39 del target
- [OK] Total de imagenes: 14,760

## Dataset Resultante

**Ubicacion**: `/data_processed`

```
data_processed/
├── Blight/           (3,690 imagenes)
├── Common_Rust/      (3,690 imagenes)
├── Gray_Leaf_Spot/   (3,690 imagenes)
└── Healthy/          (3,690 imagenes)
```

## Ejecucion

```bash
python preprocess_corn_diseases.py
```

**Entrada**: Dataset original en `/data`

**Salida**:
- Dataset procesado en `/data_processed`
- Reporte de preprocesamiento
- Visualizaciones comparativas

**Tiempo de ejecucion**: ~5-10 minutos (dependiendo del hardware)

## Configuracion

Las constantes de configuracion en el script:

```python
TARGET_SIZE = (256, 256)           # Dimension target
TARGET_MEAN_BRIGHTNESS = 120       # Brillo target
BRIGHTNESS_RANGE = (0.8, 1.2)      # Rango de ajuste de brillo
TARGET_SAMPLES_PER_CLASS = 3690    # Imagenes por clase
```

## Proximos Pasos

El dataset procesado esta listo para ser utilizado en:

1. **Entrenamiento de modelos** (`/entrenamiento_modelos`)
   - MobileNetV3-Large
   - EfficientNet-Lite
   - MobileViT
   - PMVT

2. **Splits automaticos**:
   - Entrenamiento: 70% (10,332 imagenes)
   - Validacion: 15% (2,214 imagenes)
   - Prueba: 15% (2,214 imagenes)

## Fecha de Procesamiento

8 de octubre de 2025
