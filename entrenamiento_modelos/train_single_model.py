"""
Script para entrenar un modelo específico
Útil para diagnosticar problemas o entrenar modelos individualmente
"""

import sys
import argparse
from pathlib import Path

# Importar scripts de entrenamiento
from train_mobilenetv3 import train_mobilenetv3
from train_efficientnet import train_efficientnet
from train_mobilevit import train_mobilevit
from train_pmvt import train_pmvt

def main():
    # Modelos disponibles
    modelos_disponibles = {
        'mobilenetv3': ('MobileNetV3-Large', train_mobilenetv3),
        'efficientnet': ('EfficientNet-Lite', train_efficientnet),
        'mobilevit': ('MobileViT', train_mobilevit),
        'pmvt': ('PMVT', train_pmvt)
    }

    # Parser de argumentos
    parser = argparse.ArgumentParser(description='Entrenar un modelo específico')
    parser.add_argument('modelo', choices=modelos_disponibles.keys(),
                       help='Modelo a entrenar')
    parser.add_argument('--list', action='store_true',
                       help='Listar modelos disponibles')

    args = parser.parse_args()

    if args.list:
        print("Modelos disponibles:")
        for key, (name, _) in modelos_disponibles.items():
            print(f"  {key}: {name}")
        return

    # Obtener modelo seleccionado
    nombre_modelo, funcion_entrenamiento = modelos_disponibles[args.modelo]

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║     ENTRENAMIENTO INDIVIDUAL - {nombre_modelo}
╚══════════════════════════════════════════════════════════════╝
    """)

    try:
        print(f"Iniciando entrenamiento de {nombre_modelo}...")
        funcion_entrenamiento()
        print(f"\n✓ Entrenamiento de {nombre_modelo} completado exitosamente")

    except KeyboardInterrupt:
        print(f"\n⚠️  Entrenamiento de {nombre_modelo} interrumpido por el usuario")
        sys.exit(1)

    except Exception as e:
        print(f"\n✗ Error durante el entrenamiento de {nombre_modelo}:")
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
