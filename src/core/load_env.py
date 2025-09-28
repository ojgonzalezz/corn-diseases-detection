#####################################################################################
# ----------------------------- Train Protocol Adapter ------------------------------
#####################################################################################

#########################
# ---- Depdendencies ----
#########################

import os
from pathlib import Path
from dotenv import load_dotenv, dotenv_values
from typing import Dict, Union, Optional

########################################
# ---- Train Env Variables adapter ----
########################################

class EnvLoader:
    def __init__(self, env_file: Optional[str] = None):
        # Por defecto: src/core/.env (archivo junto a este módulo)
        if env_file is None:
            env_file = os.path.join(os.path.dirname(__file__), ".env")

        # Hacemos absoluta la ruta del .env
        env_file = os.path.abspath(env_file)
        if not os.path.exists(env_file):
            raise FileNotFoundError(f"No se encontró el archivo .env en {env_file}")

        self.env_file = env_file
        self.env_dir = os.path.dirname(self.env_file)  # src/core
        self.project_root = os.path.abspath(
            os.path.join(self.env_dir, os.pardir, os.pardir)
        )

        # Cargar las variables en el entorno
        load_dotenv(self.env_file)
        # También las guardamos en un dict para acceso directo
        self._env_vars = dotenv_values(self.env_file)

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        raw = os.getenv(key, default)
        return raw

    def get_all(self) -> Dict[str, Optional[str]]:
        """
        Devuelve todas las variables definidas en el archivo .env
        en forma de diccionario.
        """
        return {k: v  for k, v in self._env_vars.items()}

