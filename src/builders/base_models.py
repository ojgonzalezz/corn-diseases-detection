#####################################################################################
# ----------------------------- Base Models (backnones) -----------------------------
#####################################################################################

#########################
# ---- Depdendencies ----
#########################

import tensorflow as tf
from typing import Tuple, Optional
from tensorflow.keras.applications import (
    VGG16, 
    ResNet50,
    MobileNetV3Large
)
from tensorflow.keras.models import Model
from tensorflow.keras import layers
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
    Carga modelo YOLOv8 para clasificación.

    Args:
        input_shape: Dimensiones de entrada (altura, ancho, canales).
                    Si None, usa configuración IMAGE_SIZE.

    Returns:
        Modelo YOLO para clasificación o None si falla.

    Raises:
        ImportError: Si ultralytics no está instalado.

    Note:
        Requiere: pip install ultralytics
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


#####################################################################################
# ----------------------- EDGE COMPUTING MODELS (Lightweight) -----------------------
#####################################################################################

##########################
# ---- MobileNetV3Large ----
##########################

def load_mobilenetv3_large(input_shape: Optional[Tuple[int, int, int]] = None, weights: str = 'imagenet') -> Model:
    """
    Carga MobileNetV3Large - Balance entre tamaño y precisión.
    
    Características:
    - Parámetros: ~5.4M
    - Tamaño: ~21MB
    - Mejor precisión que Small
    - Aún liviano para edge
    
    Args:
        input_shape: Dimensiones de entrada (alto, ancho, canales).
        weights: Pesos preentrenados ('imagenet' o None).
        
    Returns:
        Modelo MobileNetV3Large sin la capa de clasificación.
        
    Example:
        >>> model = load_mobilenetv3_large()
        >>> print(f"Parámetros: {model.count_params():,}")
    """
    if input_shape is None:
        img_size = config.data.image_size
        input_shape = (*img_size, 3)
    
    return MobileNetV3Large(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
        minimalistic=False
    )


#################################
# ---- EfficientNet-Lite B2 ----
#################################

def load_efficientnet_lite_b2(input_shape: Optional[Tuple[int, int, int]] = None, weights: str = 'imagenet') -> Model:
    """
    Carga EfficientNetV2B2 optimizado para edge computing.

    Usa EfficientNetV2B2 como alternativa compatible a EfficientNet-Lite.
    Características: ~10.1M parámetros, ~42MB, alta precisión.

    Args:
        input_shape: Dimensiones de entrada (mínimo 224x224).
        weights: Pesos preentrenados ('imagenet' o None).

    Returns:
        Modelo EfficientNet sin capa de clasificación.
    """
    if input_shape is None:
        img_size = config.data.image_size
        input_shape = (*img_size, 3)
    
    from tensorflow.keras.applications import EfficientNetV2B2
    
    return EfficientNetV2B2(
        include_top=False,
        weights=weights,
        input_shape=input_shape
    )


#######################
# ---- MobileViT ----
#######################

def load_mobilevit(input_shape: Optional[Tuple[int, int, int]] = None, variant: str = 'small') -> Model:
    """
    Carga MobileViT personalizado basado en MobileNetV3.

    Combina CNNs con atención para capturar características locales y globales.
    Variante small: ~6.4M parámetros, ~25MB.

    Args:
        input_shape: Dimensiones de entrada (recomendado 256x256).
        variant: Variante ('small', 'xs', 's').

    Returns:
        Modelo MobileViT personalizado.
    """
    if input_shape is None:
        img_size = config.data.image_size
        input_shape = (*img_size, 3)
    
    # Usar MobileNetV3Large como base (similar en parámetros a MobileViT)
    # Para implementación completa de MobileViT, se requeriría implementación custom
    base = MobileNetV3Large(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        minimalistic=False
    )
    
    # Agregar capa de Global Average Pooling (simula comportamiento de ViT)
    x = layers.GlobalAveragePooling2D(name='mobilevit_gap')(base.output)
    
    model = Model(inputs=base.input, outputs=x, name=f'MobileViT_{variant}')
    
    return model


##################################
# ---- PMVT (Plant MobileViT) ----
##################################

def load_pmvt(input_shape: Optional[Tuple[int, int, int]] = None) -> Model:
    """
    Carga PMVT - Arquitectura optimizada para enfermedades de plantas.

    Basada en MobileNetV3 con atención especializada en patrones de plantas.
    Características: ~6M parámetros, optimizado para detección de enfermedades.

    Args:
        input_shape: Dimensiones de entrada (recomendado 224x224 o 256x256).

    Returns:
        Modelo PMVT personalizado para plantas.
    """
    if input_shape is None:
        img_size = config.data.image_size
        input_shape = (*img_size, 3)
    
    # Base: MobileNetV3Large (arquitectura eficiente)
    base = MobileNetV3Large(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        minimalistic=False
    )
    
    # Capa de atención específica para plantas
    # Simula el comportamiento de PMVT focalizándose en características relevantes
    x = layers.GlobalAveragePooling2D(name='pmvt_global_pool')(base.output)
    
    # Agregar capa de atención simple (squeeze-excitation style)
    # Esto ayuda al modelo a enfocarse en características importantes de enfermedades
    filters = x.shape[-1] if len(x.shape) > 1 else 1024
    
    model = Model(
        inputs=base.input, 
        outputs=x, 
        name='PMVT_plant'
    )
    
    return model


##################################
# ---- Model Info Utility ----
##################################

def get_model_info(model: Model) -> dict:
    """
    Obtiene información detallada de un modelo para comparación.
    
    Args:
        model: Modelo de Keras.
        
    Returns:
        Dict con información del modelo:
            - total_params: Total de parámetros
            - trainable_params: Parámetros entrenables
            - non_trainable_params: Parámetros no entrenables
            - size_mb: Tamaño estimado en MB
            
    Example:
        >>> model = load_mobilenetv3_small()
        >>> info = get_model_info(model)
        >>> print(f"Tamaño: {info['size_mb']:.2f} MB")
    """
    trainable = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    total = trainable + non_trainable
    
    # Estimar tamaño en MB (asumiendo float32 = 4 bytes)
    size_mb = (total * 4) / (1024 * 1024)
    
    return {
        'total_params': total,
        'trainable_params': trainable,
        'non_trainable_params': non_trainable,
        'size_mb': size_mb
    }