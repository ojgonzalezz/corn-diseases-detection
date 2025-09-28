#####################################################################################
# ---------------------------------- Error Handling ---------------------------------
#####################################################################################

class EnvironmentLoadFailed(Exception):
    """Excepción generada cuando las variables de entorno no se cargan correctamente."""
    def __init__(self, message: str):
        super().__init__(message)


class NoModelToLoadError(Exception):
    """
    Excepción generada cuando no se encuentra un modelo para cargar.
    Esto puede ocurrir si el archivo del modelo no existe en la ruta esperada.
    """
    def __init__(self, message: str):
        super().__init__(message)


class NoLabelsError(Exception):
    """
    Excepción generada cuando no se encuentran etiquetas (nombres de clases).
    Esto es crucial para la predicción, ya que sin etiquetas,
    la salida del modelo no puede ser interpretada.
    """
    def __init__(self, message: str):
        super().__init__(message)