"""
Simplified configuration management.

This module provides configuration for the entire project without external dependencies.
"""
from typing import List, Tuple, Optional, Dict, Any
import os


class DataConfig:
    """Configuration related to data and images."""

    def __init__(self):
        # Image parameters
        self.image_size: Tuple[int, int] = (224, 224)
        self.num_classes: int = 4
        self.class_names: List[str] = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']

        # Data split parameters
        self.split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)

        # Augmentation parameters
        self.max_added_balance: int = 50
        self.im_sim_threshold: float = float(os.getenv('IM_SIM_THRESHOLD', '0.95'))

        # 🚀 PRIORIDAD CRÍTICA - Data Augmentation Agresiva
        self.augmentation_config: Dict[str, Any] = {
            'random_flip': True,  # horizontal y vertical
            'random_rotation': 30,  # grados
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
        }

        # Parámetros adicionales
        self.datasets_consideration: List[str] = ["no-augmentation", "augmented"]
        self.data_raw_subdirs: List[str] = ["data", "raw"]
        self.embedding_model: str = "ResNet50"
        self.embedding_weights: Optional[str] = "imagenet"
        self.embedding_include_top: bool = False
        self.embedding_pooling: Optional[str] = "avg"

    @property
    def similarity_threshold(self) -> float:
        """Retorna el umbral de similitud."""
        return self.im_sim_threshold


class TrainingConfig:
    """Configuración relacionada con entrenamiento de modelos."""

    def __init__(self):
        # Hiperparámetros de entrenamiento
        self.batch_size: int = int(os.getenv('BATCH_SIZE', '16'))
        self.max_epochs: int = int(os.getenv('MAX_EPOCHS', '30'))

        # Parámetros de Keras Tuner
        self.max_trials: int = int(os.getenv('MAX_TRIALS', '10'))
        self.tuner_epochs: int = int(os.getenv('TUNER_EPOCHS', '10'))
        self.factor: int = int(os.getenv('FACTOR', '3'))

        # Estrategia de balanceo
        self.balance_strategy: str = os.getenv('BALANCE_STRATEGY', 'oversample')


# Instancias globales para compatibilidad
data = DataConfig()
training = TrainingConfig()


# Funciones de conveniencia para retrocompatibilidad
def get_image_size() -> Tuple[int, int]:
    """Obtiene el tamaño de imagen configurado."""
    return data.image_size


def get_num_classes() -> int:
    """Obtiene el número de clases."""
    return data.num_classes


def get_class_names() -> List[str]:
    """Obtiene los nombres de las clases."""
    return data.class_names


def get_batch_size() -> int:
    """Obtiene el tamaño de batch."""
    return training.batch_size
