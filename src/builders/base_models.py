#####################################################################################
# ----------------------------- Base Models (backnones) -----------------------------
#####################################################################################

#########################
# ---- Depdendencies ----
#########################

import tensorflow as tf
from typing import Tuple, Optional
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.models import Model
from src.core.config import config

#################
# ---- VGG16 ----
#################

def load_vgg16(input_shape: Optional[Tuple[int, int, int]] = None, weights: str = 'imagenet') -> Model:
    """
    Carga el modelo VGG16 pre-entrenado sin la capa superior.

    Args:
        input_shape: Dimensiones de entrada (altura, ancho, canales).
                    Si es None, usa IMAGE_SIZE de la configuración + 3 canales.
        weights: Pesos preentrenados a usar ('imagenet', None, o ruta a archivo).

    Returns:
        Modelo VGG16 sin la capa de clasificación.

    Example:
        >>> model = load_vgg16()  # Usa configuración por defecto
        >>> print(model.output_shape)
    """
    if input_shape is None:
        img_size = config.data.image_size
        input_shape = (*img_size, 3)

    return VGG16(include_top=False, weights=weights, input_shape=input_shape)

#################
# ---- YOLO ----
#################

def load_yolo(input_shape: Optional[Tuple[int, int, int]] = None) -> Optional[object]:
    """
    Carga un modelo YOLO para clasificación de imágenes.

    IMPORTANTE: Esta función usa YOLOv8-cls (clasificación), no detección de objetos.
    YOLO puede usarse tanto para detección como para clasificación.

    Args:
        input_shape (tuple, optional): Shape de entrada (altura, ancho, canales).
                                      Si es None, usa IMAGE_SIZE de la configuración.
                                      Para YOLO, típicamente (640, 640, 3) pero se puede ajustar.

    Returns:
        model: Modelo YOLO adaptado para clasificación, o None si falla.

    Raises:
        ImportError: Si ultralytics no está instalado.

    Note:
        Requiere: pip install ultralytics
        Documentación: https://docs.ultralytics.com/tasks/classify/

    Example:
        >>> model = load_yolo()  # Usa configuración por defecto
        >>> # Entrenar: model.train(data='path/to/dataset', epochs=100)
    """
    if input_shape is None:
        img_size = config.data.image_size
        input_shape = (*img_size, 3)
    try:
        from ultralytics import YOLO
        import warnings

        # YOLOv8n-cls es la versión de clasificación (no detección)
        # Modelos disponibles: yolov8n-cls, yolov8s-cls, yolov8m-cls, yolov8l-cls, yolov8x-cls
        model = YOLO('yolov8n-cls.pt')  # Nano model for classification

        print("[INFO] Modelo YOLOv8-cls cargado exitosamente")
        print(f"[INFO] Input shape configurado: {input_shape}")
        print("[NOTA] YOLO usa su propio formato de entrenamiento. Ver: https://docs.ultralytics.com/tasks/classify/")

        # Advertencia sobre compatibilidad
        warnings.warn(
            "YOLO usa un pipeline de entrenamiento diferente a Keras. "
            "Considera usar VGG16 o ResNet50 para este proyecto, ya que están mejor integrados con Keras Tuner.",
            UserWarning
        )

        return model

    except ImportError as e:
        print("[ERROR] ultralytics no está instalado.")
        print("[SOLUCIÓN] Ejecuta: pip install ultralytics")
        print(f"[DETALLE] {e}")
        return None

    except Exception as e:
        print(f"[ERROR] Error al cargar el modelo YOLOv8: {e}")
        print("[SOLUCIÓN] Verifica tu conexión a internet (descarga el modelo automáticamente)")
        return None


####################
# ---- ResNet50 ----
####################

def load_resnet50(input_shape: Optional[Tuple[int, int, int]] = None, weights: str = 'imagenet') -> Model:
    """
    Carga un modelo ResNet50 pre-entrenado en ImageNet.

    El modelo se carga sin la capa superior de clasificación para ser
    utilizado como backbone para fine-tuning.

    Args:
        input_shape: Shape de entrada de las imágenes (alto, ancho, canales).
                    Si es None, usa IMAGE_SIZE de la configuración + 3 canales.
        weights: Pesos preentrenados a usar ('imagenet', None, o ruta a archivo).

    Returns:
        Modelo ResNet50 sin la capa de clasificación final.

    Example:
        >>> model = load_resnet50()  # Usa configuración por defecto
        >>> print(model.output_shape)
    """
    if input_shape is None:
        img_size = config.data.image_size
        input_shape = (*img_size, 3)

    return ResNet50(include_top=False, weights=weights, input_shape=input_shape)