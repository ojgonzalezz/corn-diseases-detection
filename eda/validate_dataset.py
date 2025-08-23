import pathlib
from PIL import Image
from tqdm import tqdm

# --- CONFIGURACIÓN ---
DATA_ROOT = pathlib.Path('data')
# ---------------------

def validate_dataset(root_path: pathlib.Path):
    """
    Recorre un directorio de datos, cuenta las imágenes por clase y verifica
    su integridad de forma robusta.
    """
    if not root_path.exists():
        print(f" Error: El directorio '{root_path}' no fue encontrado.")
        return

    print(f" Iniciando validación en: '{root_path}'\n")

    class_counts = {}
    corrupted_files = []
    
    # Obtener las carpetas de clase
    class_dirs = sorted([d for d in root_path.iterdir() if d.is_dir()])
    
    if not class_dirs:
        print(" No se encontraron subdirectorios (clases) en la ruta especificada.")
        return

    print(f" Se encontraron {len(class_dirs)} directorios de clases.")
    print("-" * 30)

    # Iterar sobre cada directorio de clase
    for class_dir in class_dirs:
        class_name = class_dir.name
        # Búsqueda flexible de extensiones de imagen (jpg, jpeg, png) sin importar mayúsculas/minúsculas
        image_paths = [p for p in class_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        
        class_counts[class_name] = 0
        
        # Usar tqdm para una barra de progreso limpia
        progress_bar = tqdm(image_paths, desc=f"Verificando '{class_name}'", unit='img', leave=False)
        
        for image_path in progress_bar:
            try:
                with Image.open(image_path) as img:
                    img.load()
                class_counts[class_name] += 1
            except (IOError, SyntaxError, OSError) as e:
                corrupted_files.append(str(image_path))

    # --- INFORME FINAL ---
    print("\n" + "=" * 35)
    print(" REPORTE DE VALIDACIÓN FINAL ")
    print("=" * 35)

    total_valid_images = sum(class_counts.values())

    print("\n Conteo de imágenes válidas por clase:")
    for class_name, count in class_counts.items():
        # Calcula el porcentaje para un contexto adicional
        percentage = (count / total_valid_images) * 100 if total_valid_images > 0 else 0
        print(f"  - {class_name:<15} | {count:>4} imágenes ({percentage:.1f}%)")

    print("-" * 35)
    print(f" Total de imágenes válidas: {total_valid_images}")
    print("-" * 35)

    if not corrupted_files:
        print("\n ¡Excelente! No se encontraron archivos corruptos.")
    else:
        print(f"\n ¡Atención! Se encontraron {len(corrupted_files)} archivos corruptos:")
        for file_path in corrupted_files:
            print(f"  - {file_path}")

if __name__ == "__main__":
    validate_dataset(DATA_ROOT)