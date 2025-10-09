# Changelog - Compatibilidad Google Colab

## Cambios Realizados para Compatibilidad con Google Colab

### Archivos Modificados

#### 1. `config.py`

**Cambios**:
- Añadida detección automática de entorno Google Colab
- Ajuste automático de rutas base según el entorno:
  - Colab: `/content/corn-diseases-detection`
  - Local: Ruta relativa del proyecto
- Cambio de `GPU_MEMORY_LIMIT` de `10240` a `None` para gestión automática

**Código añadido**:
```python
# Detectar si estamos en Google Colab
try:
    import google.colab
    IN_COLAB = True
    BASE_DIR = Path('/content/corn-diseases-detection')
except ImportError:
    IN_COLAB = False
    BASE_DIR = Path(__file__).parent.parent
```

#### 2. `utils.py`

**Cambios**:
- Actualizada función `setup_gpu()` para soportar configuración dinámica
- Añadido soporte para crecimiento dinámico de memoria GPU (óptimo para Colab)
- Mejorada compatibilidad con diferentes entornos de GPU

**Funcionalidad**:
- Si `memory_limit=None`: Usa crecimiento dinámico (recomendado para Colab)
- Si `memory_limit=valor`: Configura límite específico (entornos locales)

#### 3. `README.md`

**Cambios**:
- Añadida sección de instalación para Google Colab
- Actualizada sección de requisitos del sistema
- Referencias a documentación específica de Colab

### Archivos Nuevos

#### 4. `COLAB_SETUP.md` (NUEVO)

Documentación completa para ejecutar en Google Colab:
- Pasos de configuración paso a paso
- Instrucciones para clonar repositorio
- Métodos para subir dataset (Drive o ZIP)
- Comandos de ejecución
- Troubleshooting común
- Estimaciones de tiempo
- Instrucciones para guardar resultados

### Scripts de Entrenamiento (Sin cambios)

Los siguientes archivos **NO requieren modificación**:
- `train_mobilenetv3.py`
- `train_efficientnet.py`
- `train_mobilevit.py`
- `train_pmvt.py`
- `train_all_models.py`

**Razón**: Todos usan `from config import *` que hereda automáticamente la configuración según el entorno.

## Compatibilidad

### Totalmente Compatible

- **Google Colab**: Detección y configuración automática
- **Jupyter Notebook**: Funciona como entorno local
- **Script Python**: Ejecución directa en terminal
- **Windows/Linux/Mac**: Rutas compatibles con pathlib

### Características Específicas de Colab

1. **Detección automática**: No requiere cambios en el código
2. **GPU optimizada**: Configuración de memoria dinámica
3. **Rutas correctas**: Base en `/content/`
4. **Sin configuración manual**: Todo automático

## Uso Recomendado

### En Google Colab:
```python
!git clone https://github.com/ojgonzalezz/corn-diseases-detection.git
%cd corn-diseases-detection/entrenamiento_modelos
!pip install -q -r requirements.txt
# Subir data_processed/
!python train_mobilenetv3.py
```

### En Entorno Local:
```bash
cd entrenamiento_modelos
pip install -r requirements.txt
python train_mobilenetv3.py
```

Ambos métodos usan exactamente el mismo código sin modificaciones.
