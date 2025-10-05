# MobileNetV3Large Inference Pipeline

Sistema de inferencia optimizado para detección de enfermedades del maíz usando MobileNetV3Large en dispositivos edge.

## Contenido

- `run_pipeline.py`: Script principal de ejecución automática
- `config.yaml`: Configuración de hiperparámetros
- `convert_to_tflite.py`: Construcción, entrenamiento y conversión a TensorFlow Lite (auto-detecta estructura de datos)
- `validate_model.py`: Validación y métricas
- `inference.py`: Pipeline de inferencia
- `requirements.txt`: Archivo centralizado de dependencias

## Instalación de Dependencias

```bash
# Instalar todas las dependencias requeridas
pip install -r requirements.txt
```

## Ejecución en Google Colab

```python
# Instalar dependencias desde el archivo centralizado
!pip install -r requirements.txt

!rm -rf corn-diseases-detection
!git clone https://github.com/ojgonzalezz/corn-diseases-detection.git

from google.colab import drive
drive.mount('/content/drive')

!cd corn-diseases-detection/mobilenetv3_inference && python run_pipeline.py --data-path /content/drive/MyDrive/corn-diseases-data
```
