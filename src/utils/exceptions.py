"""
Excepciones personalizadas para el proyecto con mensajes informativos y sugerencias de recuperación.

Este módulo define todas las excepciones específicas del dominio, proporcionando
contexto claro y pasos sugeridos para resolver cada tipo de error.
"""


class CornDiseaseDetectionError(Exception):
    """
    Excepción base para todas las excepciones del proyecto.

    Todas las excepciones personalizadas deben heredar de esta clase.
    """

    def __init__(self, message: str, recovery_hint: str = None):
        """
        Inicializa la excepción con mensaje y sugerencia opcional.

        Args:
            message: Descripción del error.
            recovery_hint: Sugerencia para resolver el error.
        """
        self.message = message
        self.recovery_hint = recovery_hint
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Formatea el mensaje de error con la sugerencia."""
        if self.recovery_hint:
            return f"{self.message}\n\nSugerencia: {self.recovery_hint}"
        return self.message


# ==================== Excepciones de Datos ====================


class DataError(CornDiseaseDetectionError):
    """Excepción base para errores relacionados con datos."""
    pass


class DatasetNotFoundError(DataError):
    """Se levanta cuando no se encuentra el dataset."""

    def __init__(self, path: str):
        message = f"No se encontró el dataset en la ruta: {path}"
        recovery_hint = (
            "Verifica que:\n"
            "  1. La ruta es correcta\n"
            "  2. El directorio 'data/raw' existe\n"
            "  3. Las carpetas de clases contienen imágenes\n"
            "  4. Ejecutaste el script de descarga de datos"
        )
        super().__init__(message, recovery_hint)


class EmptyDatasetError(DataError):
    """Se levanta cuando el dataset está vacío o no tiene suficientes muestras."""

    def __init__(self, class_name: str = None, count: int = 0):
        if class_name:
            message = f"La clase '{class_name}' tiene solo {count} imágenes"
            recovery_hint = (
                f"La clase '{class_name}' necesita más muestras para entrenar.\n"
                "Opciones:\n"
                "  1. Agregar más imágenes a esta clase\n"
                "  2. Aplicar data augmentation\n"
                "  3. Remover la clase si no es crítica"
            )
        else:
            message = f"El dataset está vacío o tiene muy pocas muestras ({count})"
            recovery_hint = (
                "Verifica que:\n"
                "  1. Las imágenes estén en el formato correcto (jpg, png)\n"
                "  2. Los nombres de carpetas coincidan con CLASS_NAMES en .env\n"
                "  3. Las imágenes no estén corruptas"
            )
        super().__init__(message, recovery_hint)


class InvalidImageError(DataError):
    """Se levanta cuando una imagen no se puede cargar o es inválida."""

    def __init__(self, image_path: str, reason: str = "desconocido"):
        message = f"No se pudo cargar la imagen: {image_path}\nRazón: {reason}"
        recovery_hint = (
            "Verifica que:\n"
            "  1. El archivo es una imagen válida (jpg, png)\n"
            "  2. El archivo no está corrupto\n"
            "  3. Tienes permisos de lectura\n"
            "  4. La ruta no contiene caracteres especiales problemáticos"
        )
        super().__init__(message, recovery_hint)


class DataSplitError(DataError):
    """Se levanta cuando hay un problema con la división de datos."""

    def __init__(self, reason: str):
        message = f"Error al dividir el dataset: {reason}"
        recovery_hint = (
            "Verifica que:\n"
            "  1. SPLIT_RATIOS en .env suma 1.0\n"
            "  2. Cada clase tiene suficientes muestras\n"
            "  3. Los ratios son razonables (ej: 0.7, 0.15, 0.15)"
        )
        super().__init__(message, recovery_hint)


# ==================== Excepciones de Modelos ====================


class ModelError(CornDiseaseDetectionError):
    """Excepción base para errores relacionados con modelos."""
    pass


class NoModelToLoadError(ModelError):
    """Se levanta cuando no se encuentra el archivo del modelo."""

    def __init__(self, model_path: str):
        message = f"No se encontró el modelo en: {model_path}"
        recovery_hint = (
            "Opciones:\n"
            "  1. Entrenar un modelo ejecutando: python -m src.pipelines.train\n"
            "  2. Verificar la ruta del modelo en la configuración\n"
            "  3. Verificar que existe el directorio 'models/exported/'\n"
            "  4. Descargar un modelo pre-entrenado si está disponible"
        )
        super().__init__(message, recovery_hint)


class ModelLoadError(ModelError):
    """Se levanta cuando falla la carga del modelo."""

    def __init__(self, model_path: str, reason: str):
        message = f"Error al cargar el modelo desde {model_path}: {reason}"
        recovery_hint = (
            "Posibles causas:\n"
            "  1. El archivo del modelo está corrupto\n"
            "  2. Versión incompatible de TensorFlow/Keras\n"
            "  3. El modelo fue entrenado con una arquitectura diferente\n"
            "  4. Falta de memoria para cargar el modelo\n\n"
            "Intenta:\n"
            "  - Re-entrenar el modelo\n"
            "  - Verificar la versión de TensorFlow (requiere 2.10.0)"
        )
        super().__init__(message, recovery_hint)


class InvalidBackboneError(ModelError):
    """Se levanta cuando se especifica una arquitectura inválida."""

    def __init__(self, backbone: str, valid_options: list):
        message = f"Backbone inválido: '{backbone}'"
        recovery_hint = (
            f"Backbones válidos: {', '.join(valid_options)}\n\n"
            "Especifica uno de los backbones soportados:\n"
            f"  {chr(10).join(f'  - {opt}' for opt in valid_options)}"
        )
        super().__init__(message, recovery_hint)


# ==================== Excepciones de Configuración ====================


class ConfigError(CornDiseaseDetectionError):
    """Excepción base para errores de configuración."""
    pass


class MissingConfigError(ConfigError):
    """Se levanta cuando falta una variable de configuración requerida."""

    def __init__(self, config_name: str, config_file: str = "src/core/.env"):
        message = f"Falta la variable de configuración requerida: {config_name}"
        recovery_hint = (
            f"Agrega la variable al archivo {config_file}:\n\n"
            f"  {config_name}=<valor>\n\n"
            "Consulta el archivo de ejemplo o la documentación para valores válidos."
        )
        super().__init__(message, recovery_hint)


class InvalidConfigError(ConfigError):
    """Se levanta cuando una variable de configuración tiene un valor inválido."""

    def __init__(self, config_name: str, value: str, expected: str):
        message = f"Valor inválido para {config_name}: '{value}'"
        recovery_hint = (
            f"Valor esperado: {expected}\n\n"
            "Corrige el valor en src/core/.env y vuelve a intentar."
        )
        super().__init__(message, recovery_hint)


class NoLabelsError(ConfigError):
    """Se levanta cuando no se pueden cargar las etiquetas de clases."""

    def __init__(self):
        message = "No se pudieron cargar las etiquetas de clases desde la configuración"
        recovery_hint = (
            "Verifica que CLASS_NAMES esté definido en src/core/.env:\n\n"
            "  CLASS_NAMES=['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']\n\n"
            "El formato debe ser una lista válida de Python."
        )
        super().__init__(message, recovery_hint)


# ==================== Excepciones de GPU/Hardware ====================


class HardwareError(CornDiseaseDetectionError):
    """Excepción base para errores de hardware."""
    pass


class GPUNotAvailableError(HardwareError):
    """Se levanta cuando se requiere GPU pero no está disponible."""

    def __init__(self):
        message = "GPU no disponible pero es requerida para esta operación"
        recovery_hint = (
            "Opciones:\n"
            "  1. Instalar drivers CUDA apropiados\n"
            "  2. Instalar TensorFlow con soporte GPU:\n"
            "     pip install tensorflow-gpu==2.10.0\n"
            "  3. Ejecutar en CPU (más lento) modificando la configuración\n"
            "  4. Usar Google Colab o similar para acceso a GPU"
        )
        super().__init__(message, recovery_hint)


class InsufficientMemoryError(HardwareError):
    """Se levanta cuando no hay suficiente memoria."""

    def __init__(self, required_mb: int = None):
        if required_mb:
            message = f"Memoria insuficiente. Se requieren al menos {required_mb}MB"
        else:
            message = "Memoria insuficiente para completar la operación"

        recovery_hint = (
            "Intenta:\n"
            "  1. Reducir el tamaño del batch (BATCH_SIZE en .env)\n"
            "  2. Reducir el tamaño de imagen (IMAGE_SIZE en .env)\n"
            "  3. Cerrar otras aplicaciones\n"
            "  4. Usar un modelo más pequeño (ej: MobileNet en lugar de VGG16)\n"
            "  5. Procesar los datos en chunks más pequeños"
        )
        super().__init__(message, recovery_hint)


# ==================== Excepciones de Entrenamiento ====================


class TrainingError(CornDiseaseDetectionError):
    """Excepción base para errores durante el entrenamiento."""
    pass


class TrainingDivergenceError(TrainingError):
    """Se levanta cuando el entrenamiento diverge (loss -> NaN o Inf)."""

    def __init__(self):
        message = "El entrenamiento ha divergido (loss = NaN o Inf)"
        recovery_hint = (
            "Causas comunes y soluciones:\n"
            "  1. Learning rate muy alto → Reducir learning rate\n"
            "  2. Datos no normalizados → Verificar preprocesamiento\n"
            "  3. Batch size muy pequeño → Aumentar batch size\n"
            "  4. Inicialización incorrecta → Cambiar initializer\n"
            "  5. Datos corruptos → Validar dataset\n\n"
            "Intenta reducir el learning rate en 10x y vuelve a entrenar."
        )
        super().__init__(message, recovery_hint)


class NoImprovementError(TrainingError):
    """Se levanta cuando el modelo no mejora después de muchas épocas."""

    def __init__(self, epochs_without_improvement: int):
        message = (
            f"No hay mejora en {epochs_without_improvement} épocas. "
            "El entrenamiento puede estar estancado."
        )
        recovery_hint = (
            "Posibles soluciones:\n"
            "  1. Ajustar learning rate (probar valores más altos/bajos)\n"
            "  2. Cambiar la arquitectura del modelo\n"
            "  3. Agregar más datos de entrenamiento\n"
            "  4. Aplicar diferentes técnicas de augmentation\n"
            "  5. Verificar que las etiquetas son correctas"
        )
        super().__init__(message, recovery_hint)


# ==================== Excepciones de Inferencia ====================


class InferenceError(CornDiseaseDetectionError):
    """Excepción base para errores durante la inferencia."""
    pass


class InvalidInputError(InferenceError):
    """Se levanta cuando la entrada para inferencia es inválida."""

    def __init__(self, reason: str):
        message = f"Entrada inválida para inferencia: {reason}"
        recovery_hint = (
            "Verifica que:\n"
            "  1. La imagen está en formato RGB\n"
            "  2. El tamaño de la imagen es compatible\n"
            "  3. Los valores de píxeles están en el rango correcto [0, 255]\n"
            "  4. El archivo es una imagen válida (jpg, png)"
        )
        super().__init__(message, recovery_hint)


class PredictionError(InferenceError):
    """Se levanta cuando falla la predicción."""

    def __init__(self, reason: str):
        message = f"Error durante la predicción: {reason}"
        recovery_hint = (
            "Intenta:\n"
            "  1. Verificar que el modelo está correctamente cargado\n"
            "  2. Asegurar que la entrada tiene el formato correcto\n"
            "  3. Re-cargar el modelo\n"
            "  4. Verificar compatibilidad de versiones de TensorFlow"
        )
        super().__init__(message, recovery_hint)
