"""
Script para entrenar todas las arquitecturas edge y compararlas.

Este script ejecuta experimentos con todas las arquitecturas edge:
- MobileNetV3 (Small y Large)
- EfficientNet-Lite (B0, B1, B2)
- MobileViT
- PMVT

Registra todos los experimentos en MLflow para comparación.
"""

import sys
from pathlib import Path
import subprocess

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger, log_section

logger = get_logger(__name__)


# Configuración de experimentos (4 arquitecturas seleccionadas)
EXPERIMENTS = [
    # MobileNetV3Large - Balance tamaño/precisión
    {
        'name': 'MobileNetV3Large',
        'lr': 0.001,
        'dropout': 0.3,
        'epochs': 30,
        'batch_size': 32,
        'fine_tune': False,
    },
    
    # EfficientNet-Lite B2 - Máxima eficiencia
    {
        'name': 'EfficientNetLiteB2',
        'lr': 0.0008,
        'dropout': 0.25,
        'epochs': 30,
        'batch_size': 32,
        'fine_tune': False,
    },
    
    # MobileViT - Vision Transformer móvil
    {
        'name': 'MobileViT',
        'lr': 0.001,
        'dropout': 0.3,
        'epochs': 35,
        'batch_size': 32,
        'fine_tune': True,
        'fine_tune_epochs': 10,
    },
    
    # PMVT - Optimizado para plantas
    {
        'name': 'PMVT',
        'lr': 0.001,
        'dropout': 0.3,
        'epochs': 35,
        'batch_size': 32,
        'fine_tune': True,
        'fine_tune_epochs': 10,
    },
]


def train_all_models():
    """
    Entrena todas las arquitecturas edge secuencialmente.
    """
    log_section(logger, "ENTRENAMIENTO DE TODAS LAS ARQUITECTURAS EDGE")
    
    logger.info(f"Total de experimentos: {len(EXPERIMENTS)}")
    logger.info("")
    
    results = []
    
    for i, exp in enumerate(EXPERIMENTS, 1):
        log_section(logger, f"EXPERIMENTO {i}/{len(EXPERIMENTS)}: {exp['name']}")
        
        # Construir comando
        cmd = [
            'python',
            'experiments/edge_models/train_edge_model.py',
            '--model', exp['name'],
            '--lr', str(exp['lr']),
            '--dropout', str(exp['dropout']),
            '--epochs', str(exp['epochs']),
            '--batch-size', str(exp['batch_size']),
        ]
        
        if exp.get('fine_tune', False):
            cmd.append('--fine-tune')
            cmd.extend(['--fine-tune-epochs', str(exp.get('fine_tune_epochs', 10))])
        
        logger.info(f"Comando: {' '.join(cmd)}")
        logger.info("")
        
        # Ejecutar
        try:
            result = subprocess.run(cmd, check=True, capture_output=False)
            logger.info(f" {exp['name']} completado exitosamente")
            results.append({'model': exp['name'], 'status': 'success'})
        except subprocess.CalledProcessError as e:
            logger.error(f" {exp['name']} falló: {e}")
            results.append({'model': exp['name'], 'status': 'failed', 'error': str(e)})
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("")
    
    # Resumen final
    log_section(logger, "RESUMEN DE EXPERIMENTOS")
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    logger.info(f" Exitosos: {len(successful)}/{len(results)}")
    logger.info(f" Fallidos: {len(failed)}/{len(results)}")
    
    if successful:
        logger.info("")
        logger.info("Modelos entrenados exitosamente:")
        for r in successful:
            logger.info(f"   {r['model']}")
    
    if failed:
        logger.info("")
        logger.info("Modelos con errores:")
        for r in failed:
            logger.info(f"   {r['model']}")
    
    logger.info("")
    logger.info("Próximo paso:")
    logger.info("  python experiments/edge_models/compare_models.py")
    
    return results


if __name__ == '__main__':
    results = train_all_models()
    
    # Exit code basado en resultados
    failed = [r for r in results if r['status'] == 'failed']
    sys.exit(1 if failed else 0)

