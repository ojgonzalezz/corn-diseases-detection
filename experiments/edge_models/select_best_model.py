"""
Script para seleccionar el mejor modelo edge y generar archivo de salida.

Este script analiza todos los modelos entrenados, selecciona el mejor
según criterios de edge computing, y genera un archivo JSON con toda
la información necesaria para la siguiente fase del proyecto.

Criterios de selección:
1. Cumple requisitos mínimos (Acc ≥ 85%, Recall ≥ 0.80)
2. Mejor balance precisión/tamaño (efficiency score)
3. Tamaño ≤ 50MB para deployment en edge
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.paths import paths
from src.utils.logger import get_logger, log_section, log_dict
from src.core.config import config

logger = get_logger(__name__)


def load_experiments():
    """Carga experimentos de MLflow."""
    mlflow.set_tracking_uri(f"file:///{paths.mlruns.as_posix()}")
    client = MlflowClient()
    
    experiment = client.get_experiment_by_name("edge_models_comparison")
    if experiment is None:
        raise ValueError("No se encontró el experimento 'edge_models_comparison'")
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.test_accuracy DESC"]
    )
    
    data = []
    for run in runs:
        metrics = run.data.metrics
        params = run.data.params
        
        data.append({
            'model_name': params.get('model_name', 'Unknown'),
            'test_accuracy': metrics.get('test_accuracy', 0.0),
            'test_loss': metrics.get('test_loss', 0.0),
            'min_recall': metrics.get('min_recall', 0.0),
            'recall_Blight': metrics.get('recall_Blight', 0.0),
            'recall_Common_Rust': metrics.get('recall_Common_Rust', 0.0),
            'recall_Gray_Leaf_Spot': metrics.get('recall_Gray_Leaf_Spot', 0.0),
            'recall_Healthy': metrics.get('recall_Healthy', 0.0),
            'backbone_params': int(params.get('backbone_params', 0)),
            'backbone_size_mb': float(params.get('backbone_size_mb', 0.0)),
            'learning_rate': float(params.get('learning_rate', 0.0)),
            'dropout_rate': float(params.get('dropout_rate', 0.0)),
            'epochs': int(params.get('epochs', 0)),
            'batch_size': int(params.get('batch_size', 0)),
            'fine_tune': params.get('fine_tune', 'False') == 'True',
            'meets_requirements': metrics.get('meets_requirements', 0.0) == 1.0,
            'run_id': run.info.run_id,
            'artifact_uri': run.info.artifact_uri
        })
    
    return pd.DataFrame(data)


def calculate_efficiency_score(accuracy: float, size_mb: float, min_recall: float) -> float:
    """
    Calcula score de eficiencia para edge computing.
    
    Formula: (accuracy * min_recall) / log(size_mb + 1)
    
    Prioriza:
    - Alta precisión
    - Alto recall (importante para enfermedades)
    - Bajo tamaño
    """
    import numpy as np
    return (accuracy * min_recall) / np.log(size_mb + 1)


def select_best_model(df: pd.DataFrame):
    """
    Selecciona el mejor modelo según criterios de edge computing.
    
    Returns:
        Dict con información del mejor modelo.
    """
    # Filtrar modelos que cumplen requisitos
    df_qualified = df[df['meets_requirements'] == True].copy()
    
    if df_qualified.empty:
        logger.warning("⚠️  Ningún modelo cumple los requisitos mínimos")
        logger.info("Usando el mejor modelo disponible sin filtro de requisitos")
        df_qualified = df.copy()
        best_available = True
    else:
        best_available = False
    
    # Calcular efficiency score
    df_qualified['efficiency_score'] = df_qualified.apply(
        lambda row: calculate_efficiency_score(
            row['test_accuracy'],
            row['backbone_size_mb'],
            row['min_recall']
        ),
        axis=1
    )
    
    # Seleccionar el mejor
    best_model = df_qualified.nlargest(1, 'efficiency_score').iloc[0]
    
    return best_model.to_dict(), best_available


def generate_output_file(best_model: dict, all_models: pd.DataFrame, best_available: bool = False):
    """
    Genera archivo JSON de salida con el modelo seleccionado.
    
    Este archivo será usado en la siguiente fase del proyecto.
    """
    # Construir output
    output = {
        'selection_info': {
            'timestamp': datetime.now().isoformat(),
            'total_models_evaluated': len(all_models),
            'models_meeting_requirements': int(all_models['meets_requirements'].sum()),
            'selection_criteria': 'Mejor balance precisión/tamaño cumpliendo requisitos mínimos',
            'minimum_requirements': {
                'global_accuracy': '≥ 85%',
                'recall_per_class': '≥ 0.80'
            },
            'best_available_only': best_available
        },
        
        'selected_model': {
            'name': best_model['model_name'],
            'run_id': best_model['run_id'],
            'artifact_uri': best_model['artifact_uri'],
            'model_file': f"{best_model['model_name']}_{datetime.now().strftime('%Y%m%d')}_selected.keras"
        },
        
        'performance_metrics': {
            'test_accuracy': round(best_model['test_accuracy'], 4),
            'test_loss': round(best_model['test_loss'], 4),
            'min_recall': round(best_model['min_recall'], 4),
            'recall_per_class': {
                'Blight': round(best_model.get('recall_Blight', 0.0), 4),
                'Common_Rust': round(best_model.get('recall_Common_Rust', 0.0), 4),
                'Gray_Leaf_Spot': round(best_model.get('recall_Gray_Leaf_Spot', 0.0), 4),
                'Healthy': round(best_model.get('recall_Healthy', 0.0), 4)
            },
            'meets_minimum_requirements': bool(best_model['meets_requirements'])
        },
        
        'model_characteristics': {
            'total_parameters': int(best_model['backbone_params']),
            'model_size_mb': round(best_model['backbone_size_mb'], 2),
            'efficiency_score': round(best_model.get('efficiency_score', 0.0), 4),
            'suitable_for_edge': best_model['backbone_size_mb'] <= 50
        },
        
        'training_configuration': {
            'learning_rate': best_model['learning_rate'],
            'dropout_rate': best_model['dropout_rate'],
            'epochs_trained': int(best_model['epochs']),
            'batch_size': int(best_model['batch_size']),
            'fine_tuning_applied': bool(best_model['fine_tune']),
            'image_size': list(config.data.image_size),
            'num_classes': config.data.num_classes,
            'class_names': config.data.class_names
        },
        
        'all_models_comparison': []
    }
    
    # Agregar comparación de todos los modelos
    for _, row in all_models.sort_values('test_accuracy', ascending=False).iterrows():
        output['all_models_comparison'].append({
            'model_name': row['model_name'],
            'test_accuracy': round(row['test_accuracy'], 4),
            'min_recall': round(row['min_recall'], 4),
            'size_mb': round(row['backbone_size_mb'], 2),
            'parameters': int(row['backbone_params']),
            'meets_requirements': bool(row['meets_requirements'])
        })
    
    return output


def save_output(output: dict):
    """
    Guarda el archivo de salida JSON.
    """
    output_path = paths.root / 'experiments' / 'edge_models' / 'best_edge_model.json'
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Archivo de salida guardado en: {output_path}")
    
    return output_path


if __name__ == '__main__':
    log_section(logger, "SELECCIÓN DEL MEJOR MODELO EDGE")
    
    # Cargar experimentos
    logger.info("Cargando experimentos de MLflow...")
    df = load_experiments()
    
    if df is None or df.empty:
        logger.error("No se encontraron experimentos")
        logger.info("Ejecuta primero: python experiments/edge_models/train_all_models.py")
        sys.exit(1)
    
    logger.info(f"Experimentos encontrados: {len(df)}")
    logger.info("")
    
    # Seleccionar mejor modelo
    best_model, best_available = select_best_model(df)
    
    # Generar output
    output = generate_output_file(best_model, df, best_available)
    
    # Mostrar selección
    log_section(logger, "MODELO SELECCIONADO")
    
    if best_available:
        logger.warning("⚠️  NOTA: Ningún modelo cumplió los requisitos mínimos")
        logger.warning("Se seleccionó el mejor modelo disponible")
        logger.info("")
    
    logger.info(f"🏆 Modelo: {output['selected_model']['name']}")
    logger.info("")
    logger.info("📊 Métricas:")
    logger.info(f"   Accuracy: {output['performance_metrics']['test_accuracy']:.2%}")
    logger.info(f"   Min Recall: {output['performance_metrics']['min_recall']:.2%}")
    logger.info(f"   Loss: {output['performance_metrics']['test_loss']:.4f}")
    logger.info("")
    logger.info("💾 Características:")
    logger.info(f"   Tamaño: {output['model_characteristics']['model_size_mb']} MB")
    logger.info(f"   Parámetros: {output['model_characteristics']['total_parameters']:,}")
    logger.info(f"   Edge-ready: {'✅ Sí' if output['model_characteristics']['suitable_for_edge'] else '❌ No'}")
    logger.info("")
    
    # Guardar archivo
    output_path = save_output(output)
    
    log_section(logger, "ARCHIVO DE SALIDA GENERADO")
    logger.info("")
    logger.info(f"📁 Ubicación: {output_path}")
    logger.info("")
    logger.info("Este archivo contiene:")
    logger.info("  ✓ Modelo seleccionado y métricas")
    logger.info("  ✓ Configuración de entrenamiento")
    logger.info("  ✓ Comparación de todos los modelos")
    logger.info("  ✓ Información para deployment")
    logger.info("")
    logger.info("Úsalo en la siguiente fase del proyecto para:")
    logger.info("  • Exportar modelo a TFLite")
    logger.info("  • Deployment en dispositivos edge")
    logger.info("  • Optimización adicional")
    logger.info("")
    logger.info("🎉 Proceso de selección completado exitosamente")

