"""
Sistema de logging centralizado para el proyecto.

Este módulo proporciona configuración estandarizada de logging para todo el proyecto,
incluyendo formato consistente, niveles apropiados y manejo de archivos.
"""
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


# Colores para terminal (ANSI)
class LogColors:
    """Códigos de color ANSI para mejorar legibilidad en terminal."""
    RESET = '\033[0m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    GRAY = '\033[90m'


class ColoredFormatter(logging.Formatter):
    """
    Formatter personalizado que agrega colores a los logs en terminal.
    """

    COLORS = {
        logging.DEBUG: LogColors.GRAY,
        logging.INFO: LogColors.BLUE,
        logging.WARNING: LogColors.YELLOW,
        logging.ERROR: LogColors.RED,
        logging.CRITICAL: LogColors.MAGENTA,
    }

    def format(self, record):
        """Aplica color según el nivel de log."""
        log_color = self.COLORS.get(record.levelno, LogColors.RESET)
        record.levelname_colored = f"{log_color}{record.levelname}{LogColors.RESET}"
        return super().format(record)


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    use_colors: bool = True
) -> logging.Logger:
    """
    Configura y retorna un logger con formato estandarizado.

    Args:
        name: Nombre del logger (típicamente __name__ del módulo).
        level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Ruta opcional para guardar logs en archivo.
        use_colors: Si True, usa colores en la salida de consola.

    Returns:
        Logger configurado.

    Example:
        >>> logger = setup_logger(__name__)
        >>> logger.info("Iniciando entrenamiento...")
        >>> logger.warning("GPU no disponible, usando CPU")
        >>> logger.error("Error al cargar datos")
    """
    logger = logging.getLogger(name)

    # Evitar duplicar handlers si ya existe
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    # Formato para consola
    if use_colors:
        console_format = (
            f"{LogColors.GRAY}[%(asctime)s]{LogColors.RESET} "
            f"%(levelname_colored)s "
            f"{LogColors.CYAN}[%(name)s]{LogColors.RESET} "
            f"%(message)s"
        )
        console_formatter = ColoredFormatter(
            console_format,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        console_format = '[%(asctime)s] %(levelname)s [%(name)s] %(message)s'
        console_formatter = logging.Formatter(
            console_format,
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    # Handler para consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Handler para archivo (opcional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Formato más detallado para archivos
        file_format = (
            '[%(asctime)s] %(levelname)-8s [%(name)s:%(funcName)s:%(lineno)d] '
            '%(message)s'
        )
        file_formatter = logging.Formatter(
            file_format,
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str, log_to_file: bool = False) -> logging.Logger:
    """
    Función de conveniencia para obtener un logger configurado.

    Args:
        name: Nombre del logger (usar __name__).
        log_to_file: Si True, también guarda logs en archivo.

    Returns:
        Logger configurado.

    Example:
        >>> from src.utils.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Procesando dataset...")
    """
    log_file = None
    if log_to_file:
        from src.utils.paths import paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = paths.root / 'logs' / f'{name.replace(".", "_")}_{timestamp}.log'

    return setup_logger(name, log_file=log_file)


# Logger global para el proyecto
project_logger = get_logger('corn_diseases_detection')


def log_print(message: str, level: str = "INFO", logger: Optional[logging.Logger] = None):
    """
    Función de compatibilidad que reemplaza print() con logging estructurado.

    Detecta automáticamente el nivel basado en prefijos en el mensaje como:
    [INFO], [ERROR], [ADVERTENCIA], [OK], etc.

    Args:
        message: Mensaje a registrar.
        level: Nivel por defecto si no se detecta en el mensaje.
        logger: Logger opcional. Si no se proporciona, usa el logger global.

    Example:
        >>> log_print("[INFO] Procesando datos...")
        >>> log_print("[ERROR] Fallo al cargar modelo")
        >>> log_print("[OK] Completado exitosamente")
    """
    if logger is None:
        logger = project_logger

    # Detectar nivel del mensaje basado en prefijos comunes
    message_upper = message.upper()

    # Mapeo de prefijos a niveles
    level_mapping = {
        '[ERROR]': 'ERROR',
        '[ADVERTENCIA]': 'WARNING',
        '[WARNING]': 'WARNING',
        '[INFO]': 'INFO',
        '[OK]': 'INFO',
        '[SUCCESS]': 'INFO',
        '[DEBUG]': 'DEBUG',
        '[CRITICAL]': 'CRITICAL',
        '[GPU]': 'INFO',
        '[CARGA]': 'INFO',
        '[BUSQUEDA]': 'INFO',
        '[PROCESO]': 'INFO',
        '[CONFIG]': 'INFO',
        '[EVAL]': 'INFO',
        '[BALANCE]': 'INFO',
        '[GRAFICO]': 'INFO',
        '[GUARDADO]': 'INFO',
        '[ARCHIVO]': 'INFO',
        '[INICIO]': 'INFO',
        '[CALCULO]': 'INFO',
        '[ELIMINAR]': 'WARNING',
        '[OBJETIVO]': 'INFO',
        '[RESULTADO]': 'INFO',
        '[NOTA]': 'INFO',
    }

    detected_level = level
    for prefix, lv in level_mapping.items():
        if message_upper.startswith(prefix):
            detected_level = lv
            break

    # Limpiar prefijos redundantes para evitar duplicación
    clean_message = message
    for prefix in level_mapping.keys():
        if message.startswith(prefix):
            clean_message = message[len(prefix):].strip()
            break

    # Registrar según el nivel detectado
    log_method = getattr(logger, detected_level.lower(), logger.info)
    log_method(clean_message)


def log_section(logger: logging.Logger, title: str, char: str = '=', width: int = 70):
    """
    Imprime una sección visual en los logs.

    Args:
        logger: Logger a usar.
        title: Título de la sección.
        char: Carácter para la línea decorativa.
        width: Ancho total de la línea.

    Example:
        >>> log_section(logger, "Inicio de Entrenamiento")
        ======================================================================
        ==================== Inicio de Entrenamiento ====================
        ======================================================================
    """
    logger.info(char * width)
    padding = (width - len(title) - 2) // 2
    centered_title = f"{char * padding} {title} {char * padding}"
    logger.info(centered_title)
    logger.info(char * width)


def log_dict(logger: logging.Logger, data: dict, title: Optional[str] = None):
    """
    Imprime un diccionario de manera legible en los logs.

    Args:
        logger: Logger a usar.
        data: Diccionario a imprimir.
        title: Título opcional.

    Example:
        >>> config = {'lr': 0.001, 'batch_size': 32, 'epochs': 10}
        >>> log_dict(logger, config, "Configuración de Entrenamiento")
    """
    if title:
        logger.info(f"\n{title}:")

    max_key_length = max(len(str(k)) for k in data.keys()) if data else 0

    for key, value in data.items():
        logger.info(f"  {str(key):<{max_key_length}} : {value}")


def log_model_summary(logger: logging.Logger, model, title: str = "Resumen del Modelo"):
    """
    Imprime un resumen del modelo en los logs.

    Args:
        logger: Logger a usar.
        model: Modelo de Keras/TensorFlow.
        title: Título de la sección.
    """
    log_section(logger, title)

    try:
        # Capturar el summary en un string
        import io
        stream = io.StringIO()
        model.summary(print_fn=lambda x: stream.write(x + '\n'))
        summary_str = stream.getvalue()
        stream.close()

        for line in summary_str.split('\n'):
            if line.strip():
                logger.info(line)
    except Exception as e:
        logger.warning(f"No se pudo imprimir el resumen del modelo: {e}")
