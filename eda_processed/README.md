# Analisis Exploratorio de Datos (EDA) - Dataset Procesado

Este directorio contiene el analisis exploratorio del dataset despues de aplicar preprocesamiento y normalizacion.

## Contenido

### Resultados

- **eda_processed_report.txt**: Reporte completo con estadisticas del dataset procesado
- **eda_processed_data.png**: Visualizaciones de distribucion de clases, dimensiones y estadisticas de pixeles
- **samples_processed_data.png**: Muestras aleatorias de imagenes procesadas de cada clase

## Dataset Procesado Analizado

**Ubicacion**: `/data_processed`

### Estadisticas Generales

- Total de imagenes: 14,760
- Numero de clases: 4 (Blight, Common_Rust, Gray_Leaf_Spot, Healthy)
- Tamano promedio: 26.4 KB
- Incremento respecto al original: +1,379 imagenes (10.3%)

### Distribucion de Clases

| Clase | Imagenes | Porcentaje |
|-------|----------|------------|
| Blight | 3,690 | 25.00% |
| Common_Rust | 3,690 | 25.00% |
| Gray_Leaf_Spot | 3,690 | 25.00% |
| Healthy | 3,690 | 25.00% |

**Balance**: 100.00% (perfectamente balanceado)

### Caracteristicas de Dimensiones

- **Uniformidad**: PERFECTA
- Todas las imagenes: 256x256 px
- Ancho promedio: 256.0 px (desviacion: 0.0)
- Alto promedio: 256.0 px (desviacion: 0.0)
- Ratio de aspecto: 1.000 (uniforme)

### Caracteristicas de Pixeles

**Estadisticas Globales**:
- Brillo promedio: 113.78 (desviacion: 14.46)
- Contraste promedio: 50.61 (desviacion: 18.32)
- Canal R: 115.26, Canal G: 128.76, Canal B: 97.33
- Desviacion del brillo target (120): 6.22

**Por Clase**:

#### Blight
- Brillo: 118.44 (desviacion: 5.07)
- Contraste: 39.50 (desviacion: 7.91)
- RGB: R=121.59, G=129.17, B=104.56

#### Common_Rust
- Brillo: 99.95 (desviacion: 17.57)
- Contraste: 74.29 (desviacion: 11.59)
- RGB: R=105.49, G=113.86, B=80.51

#### Gray_Leaf_Spot
- Brillo: 116.38 (desviacion: 7.89)
- Contraste: 51.32 (desviacion: 13.43)
- RGB: R=121.73, G=127.78, B=99.61

#### Healthy
- Brillo: 120.36 (desviacion: 13.31)
- Contraste: 37.32 (desviacion: 10.22)
- RGB: R=112.22, G=144.24, B=104.63

## Mejoras Logradas

### Comparacion con Dataset Original

| Metrica | Original | Procesado | Mejora |
|---------|----------|-----------|--------|
| Balance de clases | 71.17% | 100.00% | +28.83% |
| Uniformidad dimensional | 55 resoluciones | 1 resolucion | 100% |
| Variabilidad de brillo | 26.36 | 14.46 | -45.2% |
| Total imagenes | 13,381 | 14,760 | +10.3% |

### Validaciones de Calidad

- [OK] Balance de clases: 100.00%
- [OK] Uniformidad de dimensiones: PERFECTA
- [OK] Normalizacion de brillo: desviacion=6.22 del target (120)
- [LISTO] Dataset listo para entrenamiento

## Caracteristicas del Preprocesamiento Aplicado

1. **Normalizacion de dimensiones**: Todas las imagenes a 256x256 px
2. **Normalizacion de brillo**: Target=120, rango=(0.8, 1.2)
3. **Balanceo de clases**: 3,690 imagenes por clase mediante data augmentation
4. **Data augmentation**: Rotaciones, flips, ajustes de brillo y contraste

## Estado del Dataset

**LISTO PARA ENTRENAMIENTO**

El dataset procesado cumple con todos los requisitos de calidad:
- Clases perfectamente balanceadas
- Dimensiones uniformes
- Brillo normalizado
- Total de 14,760 imagenes de alta calidad

## Uso en Entrenamiento

Este dataset es utilizado por los scripts de entrenamiento en `/entrenamiento_modelos`:
- MobileNetV3-Large
- EfficientNet-Lite
- MobileViT
- PMVT (Plant Mobile Vision Transformer)

Con split: 70% entrenamiento, 15% validacion, 15% prueba

## Fecha de Analisis

8 de octubre de 2025
