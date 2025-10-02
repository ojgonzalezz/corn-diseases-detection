"""
Script para comparar todos los modelos edge entrenados.

Lee los experimentos de MLflow y genera un análisis comparativo
mostrando métricas de rendimiento y eficiencia.
"""

import sys
import json
from pathlib import Path
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.paths import paths
from src.utils.logger import get_logger, log_section

logger = get_logger(__name__)


def get_all_experiments():
    """
    Obtiene todos los experimentos de MLflow del experimento edge_models_comparison.
    
    Returns:
        DataFrame con todos los experimentos y sus métricas.
    """
    # Configurar MLflow
    mlflow.set_tracking_uri(f"file:///{paths.mlruns.as_posix()}")
    client = MlflowClient()
    
    # Obtener experimento
    try:
        experiment = client.get_experiment_by_name("edge_models_comparison")
        if experiment is None:
            logger.error("No se encontró el experimento 'edge_models_comparison'")
            logger.info("Ejecuta primero: python experiments/edge_models/train_all_models.py")
            return None
        
        experiment_id = experiment.experiment_id
    except Exception as e:
        logger.error(f"Error al obtener experimento: {e}")
        return None
    
    # Obtener todos los runs
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["metrics.test_accuracy DESC"]
    )
    
    if not runs:
        logger.warning("No se encontraron runs en el experimento")
        return None
    
    # Extraer datos
    data = []
    for run in runs:
        metrics = run.data.metrics
        params = run.data.params
        
        data.append({
            'model_name': params.get('model_name', 'Unknown'),
            'test_accuracy': metrics.get('test_accuracy', 0.0),
            'test_loss': metrics.get('test_loss', 0.0),
            'min_recall': metrics.get('min_recall', 0.0),
            'backbone_params': int(params.get('backbone_params', 0)),
            'backbone_size_mb': float(params.get('backbone_size_mb', 0.0)),
            'learning_rate': float(params.get('learning_rate', 0.0)),
            'dropout_rate': float(params.get('dropout_rate', 0.0)),
            'epochs': int(params.get('epochs', 0)),
            'fine_tune': params.get('fine_tune', 'False') == 'True',
            'meets_requirements': metrics.get('meets_requirements', 0.0) == 1.0,
            'run_id': run.info.run_id,
            'start_time': run.info.start_time
        })
    
    df = pd.DataFrame(data)
    return df


def print_comparison_table(df: pd.DataFrame):
    """
    Imprime tabla comparativa de modelos.
    """
    log_section(logger, "COMPARACIÓN DE MODELOS EDGE")
    
    # Ordenar por accuracy descendente
    df_sorted = df.sort_values('test_accuracy', ascending=False)
    
    logger.info("")
    logger.info("Tabla Comparativa:")
    logger.info("=" * 120)
    logger.info(f"{'Modelo':<20} {'Accuracy':>10} {'Min Recall':>12} {'Params':>12} {'Size (MB)':>12} {'Cumple':>10}")
    logger.info("=" * 120)
    
    for _, row in df_sorted.iterrows():
        status = " SÍ" if row['meets_requirements'] else " NO"
        logger.info(
            f"{row['model_name']:<20} "
            f"{row['test_accuracy']:>9.2%} "
            f"{row['min_recall']:>11.2%} "
            f"{row['backbone_params']:>11,} "
            f"{row['backbone_size_mb']:>11.1f} "
            f"{status:>10}"
        )
    
    logger.info("=" * 120)
    logger.info("")


def print_best_models(df: pd.DataFrame):
    """
    Identifica y muestra los mejores modelos según diferentes criterios.
    """
    log_section(logger, "MEJORES MODELOS POR CRITERIO")
    
    # Filtrar solo los que cumplen requisitos
    df_qualified = df[df['meets_requirements'] == True]
    
    if df_qualified.empty:
        logger.warning("  Ningún modelo cumple los requisitos mínimos (Acc ≥ 85%, Recall ≥ 0.80)")
        logger.info("")
        logger.info("Mejores resultados sin filtro:")
        df_qualified = df
    
    # Mejor precisión
    best_accuracy = df_qualified.nlargest(1, 'test_accuracy').iloc[0]
    logger.info(f" Mejor Precisión: {best_accuracy['model_name']}")
    logger.info(f"   Accuracy: {best_accuracy['test_accuracy']:.2%}")
    logger.info(f"   Tamaño: {best_accuracy['backbone_size_mb']:.1f} MB")
    logger.info("")
    
    # Modelo más liviano
    lightest = df_qualified.nsmallest(1, 'backbone_size_mb').iloc[0]
    logger.info(f" Más Liviano: {lightest['model_name']}")
    logger.info(f"   Tamaño: {lightest['backbone_size_mb']:.1f} MB")
    logger.info(f"   Accuracy: {lightest['test_accuracy']:.2%}")
    logger.info("")
    
    # Mejor balance (score = accuracy / log(size_mb))
    df_qualified_copy = df_qualified.copy()
    df_qualified_copy['efficiency_score'] = (
        df_qualified_copy['test_accuracy'] / 
        (df_qualified_copy['backbone_size_mb'] / 10.0)
    )
    best_balance = df_qualified_copy.nlargest(1, 'efficiency_score').iloc[0]
    
    logger.info(f"  Mejor Balance (Precisión/Tamaño): {best_balance['model_name']}")
    logger.info(f"   Accuracy: {best_balance['test_accuracy']:.2%}")
    logger.info(f"   Tamaño: {best_balance['backbone_size_mb']:.1f} MB")
    logger.info(f"   Score eficiencia: {best_balance['efficiency_score']:.4f}")
    logger.info("")


def print_recommendations(df: pd.DataFrame):
    """
    Genera recomendaciones basadas en los resultados.
    """
    log_section(logger, "RECOMENDACIONES PARA EDGE COMPUTING")
    
    df_qualified = df[df['meets_requirements'] == True]
    
    if df_qualified.empty:
        logger.warning("  Ningún modelo cumple los requisitos mínimos.")
        logger.info("")
        logger.info("Recomendaciones:")
        logger.info("  1. Aumentar épocas de entrenamiento")
        logger.info("  2. Aplicar fine-tuning (--fine-tune)")
        logger.info("  3. Ajustar learning rate")
        logger.info("  4. Aumentar data augmentation")
        return
    
    # Calcular efficiency score
    df_qualified_copy = df_qualified.copy()
    df_qualified_copy['efficiency_score'] = (
        df_qualified_copy['test_accuracy'] / 
        (df_qualified_copy['backbone_size_mb'] / 10.0)
    )
    
    best_model = df_qualified_copy.nlargest(1, 'efficiency_score').iloc[0]
    
    logger.info(" MODELO RECOMENDADO PARA EDGE COMPUTING:")
    logger.info("")
    logger.info(f"   Modelo: {best_model['model_name']}")
    logger.info(f"    Accuracy: {best_model['test_accuracy']:.2%}")
    logger.info(f"    Recall mínimo: {best_model['min_recall']:.2%}")
    logger.info(f"    Tamaño: {best_model['backbone_size_mb']:.1f} MB")
    logger.info(f"    Parámetros: {best_model['backbone_params']:,}")
    logger.info("")
    logger.info("Razones:")
    logger.info(f"  • Cumple requisitos mínimos (Acc ≥ 85%, Recall ≥ 0.80)")
    logger.info(f"  • Mejor balance precisión/tamaño")
    logger.info(f"  • Adecuado para dispositivos edge")
    logger.info("")
    logger.info("Próximo paso:")
    logger.info("  python experiments/edge_models/select_best_model.py")


if __name__ == '__main__':
    log_section(logger, "ANÁLISIS COMPARATIVO DE MODELOS EDGE")
    
    # Obtener experimentos
    logger.info("Cargando experimentos de MLflow...")
    df = get_all_experiments()
    
    if df is None or df.empty:
        logger.error("No se pudieron cargar experimentos")
        sys.exit(1)
    
    logger.info(f"Experimentos encontrados: {len(df)}")
    logger.info("")
    
    # Mostrar comparación
    print_comparison_table(df)
    
    # Mostrar mejores modelos
    print_best_models(df)
    
    # Recomendaciones
    print_recommendations(df)
    
    # Guardar CSV
    output_path = paths.root / 'experiments' / 'edge_models' / 'comparison_results.csv'
    df.to_csv(output_path, index=False)
    logger.info(f"Resultados guardados en: {output_path}")

