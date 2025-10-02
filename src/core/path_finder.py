#####################################################################################
# --------------------------- Project Data Paths Adapter ----------------------------
#####################################################################################

#########################
# ---- Depdendencies ----
#########################

import os
from pathlib import Path
from typing import Dict, Union

################################
# ---- ProjectPaths adapter ----
################################
class ProjectPaths:
    def __init__(self, data_subpath=("data",)):
        """
        Inicializa el buscador de rutas del proyecto.

        Args:
            data_subpath (tuple): Ruta relativa desde la raíz del proyecto
                                  hacia la carpeta de datos.
        """
        self.project_root = self._find_project_root()
        self.data_path = self.project_root.joinpath(*data_subpath)

        if not self.data_path.exists():
            raise FileNotFoundError(f"❌ No existe la ruta esperada: {self.data_path}")

        # Construir automáticamente la estructura JSON de rutas
        self.rutas_data = self._build_structure(self.data_path)

    def _find_project_root(self) -> Path:
        """
        Busca la raíz del proyecto subiendo en los directorios hasta
        encontrar una carpeta llamada 'data'.
        """
        directory = Path().resolve()
        for parent in directory.parents:
            if (parent / "data").exists():
                return parent
        raise FileNotFoundError("No se encontró la carpeta 'data' en los directorios superiores.")

    def _build_structure(self, path: Path) -> Dict[str, Union[dict, str]]:
        """
        Construye recursivamente un diccionario que representa
        la estructura de carpetas bajo 'path'.
        - Si la carpeta contiene subcarpetas: crea un diccionario anidado.
        - Si no tiene subcarpetas pero sí archivos: asigna la ruta absoluta.
        - Si está vacía: devuelve {}.
        """
        entries = list(path.iterdir())
        subdirs = [e for e in entries if e.is_dir()]
        files = [e for e in entries if e.is_file()]

        if subdirs:
            return {subdir.name: self._build_structure(subdir) for subdir in subdirs}
        elif files:
            return str(path.resolve())
        else:
            return {}

    def get_data_path(self) -> Path:
        """
        Devuelve la ruta absoluta de la carpeta de datos.
        """
        return self.data_path

    def get_structure(self) -> Dict[str, Union[dict, str]]:
        """
        Devuelve el diccionario con la estructura de subcarpetas.
        """
        return self.rutas_data

    def summary(self):
        """
        Imprime información sobre la carpeta de datos y sus subcarpetas.
        """
        print("📂 Estructura detectada en data:")
        self._print_dict(self.rutas_data)

    def _print_dict(self, d: Dict, indent: int = 0):
        """
        Pretty print recursivo para la estructura de carpetas.
        """
        prefix = " " * indent
        if isinstance(d, dict):
            for k, v in d.items():
                print(f"{prefix}- {k}:")
                self._print_dict(v, indent + 4)
        else:
            print(f"{prefix}{d}")