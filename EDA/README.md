# Analisis Exploratorio de Datos (EDA) - Dataset Original

Este directorio contiene el analisis exploratorio inicial del dataset de enfermedades del maiz.

## Contenido

### Scripts

- **eda_corn_diseases.py**: Script principal de analisis exploratorio del dataset original
- **eda_processed_data.py**: Script de analisis del dataset procesado (referencia)

### Resultados

- **eda_report.txt**: Reporte completo con estadisticas del dataset original
- **eda_corn_diseases.png**: Visualizaciones de distribucion de clases, dimensiones y estadisticas de pixeles
- **samples_corn_diseases.png**: Muestras aleatorias de imagenes de cada clase

## Dataset Original Analizado

**Ubicacion**: `/data`

### Estadisticas Generales

- Total de imagenes: 13,381
- Numero de clases: 4 (Blight, Common_Rust, Gray_Leaf_Spot, Healthy)
- Tamano promedio: 26.9 KB

### Distribucion de Clases

| Clase | Imagenes | Porcentaje |
|-------|----------|------------|
| Common_Rust | 3,690 | 27.58% |
| Blight | 3,580 | 26.75% |
| Healthy | 3,485 | 26.04% |
| Gray_Leaf_Spot | 2,626 | 19.62% |

**Balance**: 71.17% (desbalance moderado)

### Caracteristicas de Dimensiones

- **Blight, Gray_Leaf_Spot, Healthy**: 256x256 px (uniformes)
- **Common_Rust**: Dimensiones variables (promedio 365x341 px)
- Total de resoluciones diferentes: 55
- Ancho promedio: 283.3 px (desviacion: 214.2)
- Alto promedio: 277.5 px (desviacion: 187.3)
- Ratio de aspecto promedio: 1.012

### Caracteristicas de Pixeles

**Estadisticas Globales**:
- Brillo promedio: 110.54 (desviacion: 26.36)
- Contraste promedio: 47.09 (desviacion: 13.54)
- Canal R: 111.50, Canal G: 125.07, Canal B: 95.04

**Por Clase**:
- **Healthy**: Mayor brillo (136.45), mas verde (G=162.34)
- **Common_Rust**: Menor brillo (86.91), mayor contraste (63.95)
- **Blight**: Brillo medio (112.01), contraste bajo (37.04)
- **Gray_Leaf_Spot**: Similar a Blight con mayor variabilidad

## Hallazgos Principales

1. **Desbalance de clases moderado**: Gray_Leaf_Spot tiene 28.8% menos imagenes que la clase mayoritaria
2. **Dimensiones no uniformes**: Common_Rust requiere redimensionamiento
3. **Alta variabilidad de brillo**: Diferencias significativas entre clases (rango: 86.91 - 136.45)
4. **Caracteristicas distintivas**: Cada clase tiene patrones de color y contraste unicos

## Recomendaciones Aplicadas

1. Aplicar data augmentation para balancear clases
2. Normalizar dimensiones a 256x256 px
3. Normalizar brillo para reducir variabilidad

Estas recomendaciones fueron implementadas en la fase de preprocesamiento.

## Ejecucion

```bash
python eda_corn_diseases.py
```

El script genera:
- Reporte de texto con estadisticas detalladas
- Visualizaciones de distribucion y caracteristicas
- Muestras representativas de cada clase

## Fecha de Analisis

8 de octubre de 2025
