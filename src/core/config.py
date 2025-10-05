"""
Simplified configuration management.

This module provides configuration for the entire project without external dependencies.
"""
from typing import List, Tuple, Optional, Dict, Any
import os
from pydantic import BaseModel, Field, ValidationError, field_validator, ValidationInfo


class DataConfig(BaseModel):
    """Configuration related to data and images."""

    # Image parameters
    image_size: Tuple[int, int] = Field((224, 224), description="Tamaño de las imágenes (ancho, alto)")
    num_classes: int = Field(4, description="Número de clases de enfermedades", gt=0)
    class_names: List[str] = Field(['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy'], description="Nombres de las clases")

    # Data split parameters
    split_ratios: Tuple[float, float, float] = Field((0.7, 0.15, 0.15), description="Proporciones de división (train, val, test)")

    # Augmentation parameters
    max_added_balance: int = Field(50, description="Máximo de imágenes a añadir para balanceo", ge=0)
    im_sim_threshold: float = Field(
        default_factory=lambda: float(os.getenv('IM_SIM_THRESHOLD', '0.95')),
        description="Umbral de similitud de imágenes para detección de duplicados",
        ge=0.0,
        le=1.0,
    )

    # 🚀 PRIORIDAD CRÍTICA - Data Augmentation Agresiva
    augmentation_config: Dict[str, Any] = Field(
        default_factory=lambda: {
            'random_flip': True,  # horizontal y vertical
            'random_rotation': True,  # rotaciones de 90° (0°, 90°, 180°, 270°)
            'random_zoom': (0.8, 1.2),
            'random_shear': 0.2,
            'color_jitter': {
                'brightness': 0.3,
                'contrast': 0.3,
                'saturation': 0.3,
                'hue': 0.1
            },
            'gaussian_noise': 0.05,
            'random_erasing': 0.2,  # probabilidad
            'cutmix': True,  # alpha=1.0
            'mixup': True,   # alpha=0.2
        },
        description="Configuración de data augmentation agresiva",
    )

    # Parámetros adicionales
    datasets_consideration: List[str] = Field(default_factory=lambda: ["no-augmentation", "augmented"], description="Consideraciones de los datasets (ej. si se aplica aumento)")
    data_raw_subdirs: List[str] = Field(default_factory=lambda: ["data", "raw"], description="Subdirectorios para datos crudos")
    embedding_model: str = Field("ResNet50", description="Modelo de embedding para detección de similitud")
    embedding_weights: Optional[str] = Field("imagenet", description="Pesos preentrenados para el modelo de embedding")
    embedding_include_top: bool = Field(False, description="Incluir la capa top en el modelo de embedding")
    embedding_pooling: Optional[str] = Field("avg", description="Tipo de pooling para el modelo de embedding")

    @field_validator('image_size')
    @classmethod
    def image_size_must_be_positive(cls, v):
        if not all(dim > 0 for dim in v):
            raise ValueError("image_size dimensions must be positive")
        return v

    @field_validator('split_ratios')
    @classmethod
    def split_ratios_sum_to_one(cls, v):
        if abs(sum(v) - 1.0) > 1e-6:
            raise ValueError('split_ratios must sum to 1.0')
        return v

    @field_validator('class_names')
    @classmethod
    def class_names_match_num_classes(cls, v, info: ValidationInfo):
        if 'num_classes' in info.data and len(v) != info.data['num_classes']:
            raise ValueError('Length of class_names must match num_classes')
        return v


class TrainingConfig(BaseModel):
    """Configuración relacionada con entrenamiento de modelos."""

    # Hiperparámetros de entrenamiento
    batch_size: int = Field(default_factory=lambda: int(os.getenv('BATCH_SIZE', '32')), description="Tamaño del batch de entrenamiento", gt=0)
    max_epochs: int = Field(default_factory=lambda: int(os.getenv('MAX_EPOCHS', '30')), description="Número máximo de épocas de entrenamiento", gt=0)

    # Parámetros de Keras Tuner
    max_trials: int = Field(default_factory=lambda: int(os.getenv('MAX_TRIALS', '10')), description="Número máximo de pruebas para Keras Tuner", gt=0)
    tuner_epochs: int = Field(default_factory=lambda: int(os.getenv('TUNER_EPOCHS', '10')), description="Número de épocas por prueba de Keras Tuner", gt=0)
    factor: int = Field(default_factory=lambda: int(os.getenv('FACTOR', '3')), description="Factor de reducción para Keras Tuner (en reducciones de learning rate)", gt=0)

    # Estrategia de balanceo
    balance_strategy: str = Field(default_factory=lambda: os.getenv('BALANCE_STRATEGY', 'oversample'), description="Estrategia de balanceo de clases (oversample, downsample)")


# Instancias globales para compatibilidad
data = DataConfig()
training = TrainingConfig()


class Config(BaseModel):
    """Clase config compatible con el código existente."""
    data: DataConfig
    training: TrainingConfig

    def to_dict(self) -> Dict[str, Any]:
        """Serializa la configuración a un diccionario."""
        return {
            "data": self.data.model_dump(),
            "training": self.training.model_dump(),
        }


# Instancia global config para compatibilidad
config = Config(data=data, training=training)


# Funciones de conveniencia para retrocompatibilidad
def get_image_size() -> Tuple[int, int]:
    """Obtiene el tamaño de imagen configurado."""
    return config.data.image_size


def get_num_classes() -> int:
    """Obtiene el número de clases."""
    return config.data.num_classes


def get_class_names() -> List[str]:
    """Obtiene los nombres de las clases."""
    return config.data.class_names


def get_batch_size() -> int:
    """Obtiene el tamaño de batch."""
    return config.training.batch_size
