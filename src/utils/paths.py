"""
Utilidades centralizadas para el manejo de rutas del proyecto.
"""
import os
from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """
    Obtiene la ruta raíz del proyecto de manera confiable.

    Esta función busca el directorio raíz navegando hacia arriba desde
    el archivo actual hasta encontrar el directorio que contiene 'src/'.

    Returns:
        Path: Ruta absoluta al directorio raíz del proyecto.

    Raises:
        RuntimeError: Si no se puede determinar la raíz del proyecto.
    """
    try:
        # Intentar obtener desde __file__ (cuando se ejecuta desde un módulo)
        current_file = Path(__file__).resolve()
        # Desde src/utils/paths.py -> subir 2 niveles
        project_root = current_file.parent.parent.parent
    except NameError:
        # Fallback si se ejecuta desde un script interactivo
        project_root = Path(os.getcwd())

    # Validar que encontramos la raíz correcta (debe tener directorio 'src')
    if not (project_root / 'src').exists():
        raise RuntimeError(
            f"No se pudo determinar la raíz del proyecto. "
            f"Ruta calculada: {project_root}. "
            f"Asegúrate de que el directorio 'src/' existe."
        )

    return project_root


class ProjectPaths:
    """
    Clase para gestionar las rutas estándar del proyecto.

    Proporciona acceso centralizado a todas las rutas importantes,
    asegurando consistencia en todo el proyecto.
    """

    def __init__(self):
        """Inicializa las rutas del proyecto."""
        self._root = get_project_root()

    @property
    def root(self) -> Path:
        """Directorio raíz del proyecto."""
        return self._root

    @property
    def src(self) -> Path:
        """Directorio de código fuente."""
        return self._root / 'src'

    @property
    def data(self) -> Path:
        """Directorio principal de datos."""
        return self._root / 'data'

    @property
    def data_raw(self) -> Path:
        """Directorio de datos crudos."""
        return self.data / 'raw'

    @property
    def data_processed(self) -> Path:
        """Directorio de datos procesados."""
        return self.data / 'processed'

    @property
    def models(self) -> Path:
        """Directorio principal de modelos."""
        return self._root / 'models'

    @property
    def models_exported(self) -> Path:
        """Directorio de modelos exportados."""
        return self.models / 'exported'

    @property
    def models_tuner(self) -> Path:
        """Directorio de checkpoints del tuner."""
        return self.models / 'tuner_checkpoints'

    @property
    def mlruns(self) -> Path:
        """Directorio de experimentos MLflow."""
        return self.models / 'mlruns'

    @property
    def tests(self) -> Path:
        """Directorio de tests."""
        return self._root / 'tests'

    def ensure_dir(self, path: Path) -> Path:
        """
        Asegura que un directorio existe, creándolo si es necesario.

        Args:
            path: Ruta del directorio a verificar/crear.

        Returns:
            Path: La misma ruta proporcionada.
        """
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_model_path(self, model_name: str) -> Path:
        """
        Obtiene la ruta completa para un modelo específico.

        Args:
            model_name: Nombre del archivo del modelo.

        Returns:
            Path: Ruta completa al modelo.
        """
        return self.models_exported / model_name

    def relative_to_root(self, path: Path) -> Path:
        """
        Convierte una ruta absoluta en relativa al root del proyecto.

        Args:
            path: Ruta absoluta.

        Returns:
            Path: Ruta relativa al root.
        """
        try:
            return path.relative_to(self.root)
        except ValueError:
            # Si path no es relativo a root, retornar tal cual
            return path


# Instancia global para uso conveniente
paths = ProjectPaths()


# Funciones de conveniencia para compatibilidad
def get_data_dir() -> Path:
    """Obtiene el directorio de datos raw."""
    return paths.data_raw


def get_models_dir() -> Path:
    """Obtiene el directorio de modelos exportados."""
    return paths.models_exported


def get_mlruns_dir() -> Path:
    """Obtiene el directorio de MLflow."""
    return paths.mlruns
