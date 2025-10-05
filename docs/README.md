# Detección de Enfermedades del Maíz - Modelos Edge

Sistema de deep learning para diagnóstico automático de enfermedades comunes en hojas de maíz usando 4 arquitecturas optimizadas para edge entrenadas en Google Colab con GPU gratuita.

## Resumen

Implementa pipeline completo para clasificación automática de enfermedades usando arquitecturas ligeras optimizadas para dispositivos edge.

## Clases de Enfermedad

Clasifica 4 categorías:
- Blight (Barrenador del maíz)
- Common_Rust (Roya común)
- Gray_Leaf_Spot (Mancha gris de la hoja)
- Healthy (Hojas sanas)

## Estructura del Proyecto

```
corn-diseases-detection/
├── data/                    # Dataset (ignorado por git)
├── src/                     # Código fuente
│   ├── adapters/           # Carga de datos
│   ├── builders/           # Constructores de modelos edge
│   ├── core/               # Configuración central
│   ├── pipelines/          # Pipelines ML
│   └── utils/              # Utilidades
├── tests/                  # Suite de pruebas (10 archivos)
├── EDA/                    # Scripts y notebooks de EDA
├── experiments/            # Experimentos edge computing
│   └── edge_models/        # Entrenamiento arquitecturas ligeras
├── notebooks/colab_edge_models_training.ipynb  # Notebook principal Colab
└── README.md
```

## Inicio Rápido

1. Subir carpeta `data/` a Google Drive en `MyDrive/corn-diseases-data/`
2. Abrir `notebooks/colab_edge_models_training.ipynb` en Google Colab
3. Configurar runtime a GPU (T4)
4. Ejecutar todas las celdas
5. Esperar 2-3 horas para completar entrenamiento para cada iteración.

Ver documentación en carpeta `docs/` para instrucciones detalladas.
