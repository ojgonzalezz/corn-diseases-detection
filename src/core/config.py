"""
Central configuration management using Pydantic.

This module provides type validation and robust configuration
for the entire project, replacing manual environment variable handling.
"""
from typing import List, Tuple, Literal, Optional
from pathlib import Path
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataConfig(BaseSettings):
    """Configuration related to data and images."""

    model_config = SettingsConfigDict(
        env_file='src/core/.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )

    # Image parameters
    image_size: Tuple[int, int] = Field(
        default=(224, 224),
        description="Image size (width, height)"
    )

    num_classes: int = Field(
        default=4,
        ge=2,
        description="Number of classes to classify"
    )

    class_names: List[str] = Field(
        default=['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy'],
        description="Class names"
    )

    # Data split parameters
    split_ratios: Tuple[float, float, float] = Field(
        default=(0.7, 0.15, 0.15),
        description="Split ratios (train, val, test)"
    )

    # Augmentation parameters
    max_added_balance: int = Field(
        default=50,
        ge=0,
        description="Maximum number of images to add when balancing"
    )

    im_sim_threshold: float = Field(
        default=0.95,
        validation_alias='IM_SIM_THRESHOLD',  # Nombre correcto
        ge=0.0,
        le=1.0,
        description="Umbral de similitud para detección de imágenes duplicadas"
    )

    # Compatibilidad con typo legacy (deprecado)
    im_sim_treshold: Optional[float] = Field(
        default=None,
        validation_alias='IM_SIM_TRESHOLD',  # Typo legacy
        ge=0.0,
        le=1.0,
        description="[DEPRECADO] Usar IM_SIM_THRESHOLD en su lugar"
    )

    @property
    def similarity_threshold(self) -> float:
        """
        Retorna el umbral de similitud, priorizando el nombre correcto.
        Mantiene compatibilidad con el typo legacy.
        """
        # Si se usó el typo, retornarlo (pero debería migrar)
        if self.im_sim_treshold is not None:
            import warnings
            warnings.warn(
                "IM_SIM_TRESHOLD está deprecado. Usa IM_SIM_THRESHOLD en su lugar.",
                DeprecationWarning,
                stacklevel=2
            )
            return self.im_sim_treshold
        return self.im_sim_threshold

    datasets_consideration: List[str] = Field(
        default=["no-augmentation", "augmented"],
        description="Tipos de datasets a considerar"
    )

    # Parámetros de rutas de datos
    data_raw_subdirs: List[str] = Field(
        default=["data", "raw"],
        description="Subdirectorios para datos raw (ej: ['data', 'raw'])"
    )

    # Parámetros del modelo de embedding para de-augmentación
    embedding_model: Literal["ResNet50", "VGG16", "MobileNetV2"] = Field(
        default="ResNet50",
        description="Modelo preentrenado para generar embeddings en de-augmentación"
    )

    embedding_weights: Literal["imagenet", None] = Field(
        default="imagenet",
        description="Pesos preentrenados para el modelo de embedding"
    )

    embedding_include_top: bool = Field(
        default=False,
        description="Incluir capa de clasificación en modelo de embedding"
    )

    embedding_pooling: Literal["avg", "max", None] = Field(
        default="avg",
        description="Tipo de pooling para modelo de embedding"
    )

    @field_validator('class_names')
    @classmethod
    def validate_class_names(cls, v: List[str], info) -> List[str]:
        """Valida que los nombres de clase coincidan con num_classes."""
        num_classes = info.data.get('num_classes', 4)
        if len(v) != num_classes:
            raise ValueError(
                f"El número de class_names ({len(v)}) debe coincidir "
                f"con num_classes ({num_classes})"
            )
        return v

    @field_validator('split_ratios')
    @classmethod
    def validate_split_ratios(cls, v: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Valida que los ratios sumen aproximadamente 1.0."""
        if abs(sum(v) - 1.0) > 0.01:
            raise ValueError(
                f"Los split_ratios deben sumar 1.0, suma actual: {sum(v):.4f}"
            )
        for ratio in v:
            if not (0.0 < ratio < 1.0):
                raise ValueError(
                    f"Cada ratio debe estar entre 0 y 1, se encontró: {ratio}"
                )
        return v


class TrainingConfig(BaseSettings):
    """Configuración relacionada con entrenamiento de modelos."""

    model_config = SettingsConfigDict(
        env_file='src/core/.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )

    # Hiperparámetros de entrenamiento
    batch_size: int = Field(
        default=32,
        ge=1,
        description="Tamaño del batch"
    )

    max_epochs: int = Field(
        default=20,
        ge=1,
        description="Número máximo de épocas"
    )

    # Parámetros de Keras Tuner
    max_trials: int = Field(
        default=10,
        ge=1,
        description="Número máximo de trials para Keras Tuner"
    )

    tuner_epochs: int = Field(
        default=10,
        ge=1,
        description="Épocas por trial en Keras Tuner"
    )

    factor: int = Field(
        default=3,
        ge=2,
        description="Factor de reducción para Hyperband"
    )

    # Estrategia de balanceo
    balance_strategy: Literal["oversample", "downsample", "none"] = Field(
        default="oversample",
        description="Estrategia de balanceo de clases"
    )

    # Arquitectura del modelo
    backbone: Literal["VGG16", "ResNet50", "YOLO"] = Field(
        default="VGG16",
        description="Arquitectura base del modelo"
    )


class ProjectConfig(BaseSettings):
    """Configuración general del proyecto."""

    model_config = SettingsConfigDict(
        env_file='src/core/.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )

    # Información del proyecto
    project_name: str = Field(
        default="corn-diseases-detection",
        description="Nombre del proyecto"
    )

    version: str = Field(
        default="0.1.0",
        description="Versión del proyecto"
    )

    # Configuración de logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Nivel de logging"
    )

    log_to_file: bool = Field(
        default=False,
        description="Si se debe guardar logs en archivo"
    )

    # Configuración de MLflow
    mlflow_experiment_name: str = Field(
        default="image_classification_experiment",
        description="Nombre del experimento en MLflow"
    )

    # Semilla para reproducibilidad
    random_seed: int = Field(
        default=42,
        ge=0,
        description="Semilla para reproducibilidad"
    )


class Config:
    """
    Clase principal de configuración que agrupa todas las sub-configuraciones.

    Esta clase proporciona acceso unificado a toda la configuración del proyecto,
    con validación automática y valores por defecto razonables.

    Example:
        >>> from src.core.config import config
        >>> print(config.data.image_size)
        (224, 224)
        >>> print(config.training.batch_size)
        32
        >>> print(config.project.project_name)
        'corn-diseases-detection'
    """

    def __init__(self):
        """Inicializa todas las sub-configuraciones."""
        self._data = None
        self._training = None
        self._project = None

    @property
    def data(self) -> DataConfig:
        """Configuración de datos."""
        if self._data is None:
            self._data = DataConfig()
        return self._data

    @property
    def training(self) -> TrainingConfig:
        """Configuración de entrenamiento."""
        if self._training is None:
            self._training = TrainingConfig()
        return self._training

    @property
    def project(self) -> ProjectConfig:
        """Configuración del proyecto."""
        if self._project is None:
            self._project = ProjectConfig()
        return self._project

    def reload(self):
        """Recarga toda la configuración desde archivos."""
        self._data = None
        self._training = None
        self._project = None

    def to_dict(self) -> dict:
        """
        Exporta toda la configuración como diccionario.

        Returns:
            dict: Configuración completa.
        """
        return {
            'data': self.data.model_dump(),
            'training': self.training.model_dump(),
            'project': self.project.model_dump()
        }

    def __repr__(self) -> str:
        """Representación legible de la configuración."""
        return (
            f"Config(\n"
            f"  data={self.data.model_dump()},\n"
            f"  training={self.training.model_dump()},\n"
            f"  project={self.project.model_dump()}\n"
            f")"
        )


# Instancia global para uso en todo el proyecto
config = Config()


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
