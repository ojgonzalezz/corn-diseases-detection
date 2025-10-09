"""
Script para entrenar todos los modelos secuencialmente
"""

import os
import sys
from datetime import datetime

# Importar scripts de entrenamiento
from train_mobilenetv3 import train_mobilenetv3
from train_efficientnet import train_efficientnet
from train_mobilevit import train_mobilevit
from train_pmvt import train_pmvt

def main():
    """Entrenar todos los modelos"""

    print("\n" + "="*60)
    print("ENTRENAMIENTO DE TODOS LOS MODELOS")
    print("="*60)
    print(f"Fecha de inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    modelos = [
        ("MobileNetV3-Large", train_mobilenetv3),
        ("EfficientNet-Lite", train_efficientnet),
        ("MobileViT", train_mobilevit),
        ("PMVT", train_pmvt)
    ]

    resultados = []

    for i, (nombre, train_func) in enumerate(modelos, 1):
        print(f"\n\n{'='*60}")
        print(f"MODELO {i}/4: {nombre}")
        print(f"{'='*60}\n")

        try:
            start_time = datetime.now()
            train_func()
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            resultados.append({
                'modelo': nombre,
                'estado': 'COMPLETADO',
                'duracion': duration
            })

            print(f"\n{nombre} - Entrenamiento completado en {duration/60:.2f} minutos")

        except Exception as e:
            print(f"\nERROR entrenando {nombre}: {str(e)}")
            resultados.append({
                'modelo': nombre,
                'estado': 'ERROR',
                'error': str(e)
            })

    # Resumen final
    print("\n" + "="*60)
    print("RESUMEN DE ENTRENAMIENTOS")
    print("="*60)

    for resultado in resultados:
        if resultado['estado'] == 'COMPLETADO':
            print(f"[OK] {resultado['modelo']}: {resultado['duracion']/60:.2f} min")
        else:
            print(f"[ERROR] {resultado['modelo']}: {resultado.get('error', 'Unknown')}")

    print("\n" + "="*60)
    print("TODOS LOS ENTRENAMIENTOS FINALIZADOS")
    print(f"Fecha de finalizacion: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    print("\nPara ver los resultados en MLflow UI, ejecuta:")
    print("  mlflow ui --backend-store-uri mlruns/")

if __name__ == "__main__":
    main()
