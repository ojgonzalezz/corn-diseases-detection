"""
Script de prueba rápida para un solo modelo edge.

Este script prueba la implementación eficiente de carga de datos
con un solo modelo y parámetros reducidos para debugging.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger, log_section

logger = get_logger(__name__)


def test_single_model():
    """
    Prueba un solo modelo con parámetros reducidos.
    """
    log_section(logger, "PRUEBA DE MODELO ÚNICO - PMVT")

    # Importar aquí para evitar problemas de dependencias
    from experiments.edge_models.train_edge_model import train_edge_model

    logger.info("Probando PMVT con parámetros reducidos para verificar funcionamiento...")

    try:
        # Parámetros reducidos para prueba rápida
        results = train_edge_model(
            model_name='PMVT',
            learning_rate=0.001,
            dropout_rate=0.3,
            epochs=2,  # Solo 2 épocas para prueba
            batch_size=8,
            fine_tune=True,
            fine_tune_epochs=1  # Solo 1 época de fine-tuning
        )

        logger.info("OK Prueba completada exitosamente!")
        logger.info(f"Resultados: {results}")

        return True

    except Exception as e:
        logger.error(f"ERROR Error en la prueba: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == '__main__':
    success = test_single_model()
    sys.exit(0 if success else 1)

