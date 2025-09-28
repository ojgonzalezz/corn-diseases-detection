#####################################################################################
# ----------------------------- Base Models (backnones) -----------------------------
#####################################################################################

#########################
# ---- Depdendencies ----
#########################

import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50

#################
# ---- VGG16 ----
#################

def load_vgg16(input_shape=(224, 224, 3)):
    return VGG16(include_top=False, weights='imagenet', input_shape=input_shape)

#################
# ---- YOLO ----
#################

def load_yolo(input_shape=(640, 640, 3)):
    """
    Carga un modelo YOLO.
    
    Nota: YOLO es un modelo de detección de objetos, no de clasificación.
    No está disponible en tensorflow.keras.applications. 
    Esta es una implementación de ejemplo que podría requerir la instalación
    de paquetes adicionales como `ultralytics`.
    """
    try:
        from ultralytics import YOLO
        # Puedes cargar un modelo pre-entrenado de YOLOv8
        model = YOLO('yolov8n.pt') 
        # Convertir a un formato de Keras o TensorFlow si es necesario
        # Esto es un placeholder, la conversión no es trivial.
        print("Modelo YOLOv8 cargado. ¡Recuerda que esto es un detector de objetos!")
        return model.model 
    except ImportError:
        print("Error: El modelo YOLO (ultralytics) no se pudo importar. Asegúrate de instalarlo con: pip install ultralytics")
        return None
    except Exception as e:
        print(f"Error al cargar el modelo YOLOv8: {e}")
        return None


####################
# ---- ResNet50 ----
####################

def load_resnet50(input_shape=(224, 224, 3)):
    """
    Carga un modelo ResNet50 pre-entrenado en ImageNet.
    
    El modelo se carga sin la capa superior de clasificación para ser
    utilizado como backbone para fine-tuning.
    """
    return ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)