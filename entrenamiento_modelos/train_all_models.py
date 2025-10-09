"""
Script para entrenar todos los modelos secuencialmente
Versión mejorada con mejor manejo de errores y timeouts
"""

import os
import sys
import time
import signal
from datetime import datetime

# Importar scripts de entrenamiento
from train_mobilenetv3 import train_mobilenetv3
from train_efficientnet import train_efficientnet
from train_mobilevit import train_mobilevit
from train_pmvt import train_pmvt

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Timeout alcanzado")

def train_with_timeout(train_func, model_name, timeout_hours=2):
    """Entrenar modelo con timeout"""
    # Configurar timeout (en segundos)
    timeout_seconds = timeout_hours * 3600
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)

    try:
        start_time = datetime.now()
        print(f"Iniciando entrenamiento de {model_name}...")
        print(f"Timeout configurado: {timeout_hours} horas")

        train_func()

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Cancelar timeout
        signal.alarm(0)

        return {
            'estado': 'COMPLETADO',
            'duracion': duration,
            'mensaje': f"Completado en {duration/60:.2f} minutos"
        }

    except TimeoutError:
        print(f"\n⚠️  Timeout alcanzado para {model_name} después de {timeout_hours} horas")
        return {
            'estado': 'TIMEOUT',
            'duracion': timeout_seconds,
            'mensaje': f"Timeout después de {timeout_hours} horas"
        }

    except Exception as e:
        # Cancelar timeout
        signal.alarm(0)
        print(f"\n✗ Error entrenando {model_name}: {str(e)}")
        return {
            'estado': 'ERROR',
            'error': str(e),
            'mensaje': f"Error: {str(e)}"
        }

def main():
    """Entrenar todos los modelos"""

    print("\n" + "="*60)
    print("ENTRENAMIENTO DE TODOS LOS MODELOS")
    print("="*60)
    print(f"Fecha de inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    print("NOTA: Cada modelo tiene un timeout de 2 horas")
    print("Si un modelo se queda atascado, se saltará al siguiente")

    modelos = [
        ("MobileNetV3-Large", train_mobilenetv3),
        ("EfficientNet-Lite", train_efficientnet),
        ("MobileViT", train_mobilevit),
        ("PMVT", train_pmvt)
    ]

    resultados = []
    global_start = time.time()

    for i, (nombre, train_func) in enumerate(modelos, 1):
        print(f"\n\n{'='*60}")
        print(f"MODELO {i}/4: {nombre}")
        print(f"{'='*60}\n")

        # Entrenar con timeout
        resultado = train_with_timeout(train_func, nombre, timeout_hours=2)

        resultados.append({
            'modelo': nombre,
            **resultado
        })

        # Pequeña pausa entre modelos para liberar memoria
        if i < len(modelos):
            print(f"\nPausa de 30 segundos antes del siguiente modelo...")
            time.sleep(30)

    # Resumen final
    global_duration = time.time() - global_start

    print("\n" + "="*60)
    print("RESUMEN DE ENTRENAMIENTOS")
    print("="*60)
    print(f"Tiempo total de entrenamiento: {global_duration/60:.2f} minutos")
    completed = 0
    for resultado in resultados:
        status_icon = {
            'COMPLETADO': '[✓]',
            'TIMEOUT': '[⏰]',
            'ERROR': '[✗]'
        }.get(resultado['estado'], '[?]')

        print(f"{status_icon} {resultado['modelo']}: {resultado['mensaje']}")

        if resultado['estado'] == 'COMPLETADO':
            completed += 1

    print(f"\nModelos completados: {completed}/{len(modelos)}")

    print("\n" + "="*60)
    print("PROCESO FINALIZADO")
    print(f"Fecha de finalización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    if completed > 0:
        print("\nPara ver los resultados en MLflow UI, ejecuta:")
        print("  mlflow ui --backend-store-uri mlruns/")
        print("\nLos modelos entrenados se guardaron en:")
        print("  Mi unidad/corn-diseases-detection/models/")

    # Guardar resumen en archivo
    try:
        summary_file = "entrenamiento_resumen.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("RESUMEN DE ENTRENAMIENTO\n")
            f.write("="*60 + "\n\n")
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Tiempo total: {global_duration/60:.2f} minutos\n")
            f.write(f"Modelos completados: {completed}/{len(modelos)}\n\n")

            for resultado in resultados:
                f.write(f"{resultado['modelo']}: {resultado['mensaje']}\n")

            f.write(f"\n{'='*60}\n")

        print(f"\nResumen guardado en: {summary_file}")
    except Exception as e:
        print(f"Error guardando resumen: {e}")

if __name__ == "__main__":
    main()
