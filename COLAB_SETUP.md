# 🚀 Guía de Ejecución en Google Colab

## 📋 Preparación (5 minutos)

### 1. Preparar Datos en Google Drive

```
MyDrive/
└── corn-diseases-data/
    ├── train/
    │   ├── Blight/
    │   ├── Common_Rust/
    │   ├── Gray_Leaf_Spot/
    │   └── Healthy/
    ├── val/
    │   └── [mismas clases]
    └── test/
        └── [mismas clases]
```

**Cómo subir:**
1. Abre Google Drive: https://drive.google.com
2. Crea carpeta: `corn-diseases-data`
3. Sube tus carpetas `train/`, `val/`, `test/`

---

### 2. Abrir Notebook en Colab

**Opción A: Desde archivo local**
1. Ve a: https://colab.research.google.com
2. File > Upload notebook
3. Sube: `colab_edge_models_training.ipynb`

**Opción B: Desde GitHub**
1. Ve a: https://colab.research.google.com
2. File > Open notebook > GitHub
3. URL: `https://github.com/ojgonzalezz/corn-diseases-detection`
4. Selecciona: `colab_edge_models_training.ipynb`

---

### 3. Configurar GPU

1. Runtime > Change runtime type
2. Hardware accelerator: **GPU**
3. GPU type: **T4** (gratis)
4. Save

---

## ▶️ Ejecución (2-3 horas)

### Ejecutar Todo Automáticamente

1. Runtime > Run all
2. Autorizar acceso a Google Drive cuando se solicite
3. Esperar a que termine

### O Ejecutar Paso a Paso

Ejecuta cada celda en orden (Shift + Enter):

1. ✅ Verificar GPU
2. ✅ Clonar repositorio
3. ✅ Instalar dependencias
4. ✅ Montar Google Drive
5. ✅ Copiar datos
6. ✅ Verificar datos
7. ✅ Configurar entorno
8. ✅ **Entrenar modelos** (⏱️ 2-3 horas)
9. ✅ Comparar resultados
10. ✅ Seleccionar mejor modelo
11. ✅ Ver resultados
12. ✅ Descargar archivos

---

## 📊 Monitoreo Durante Entrenamiento

### Ver Progreso

```python
# En una nueva celda:
!tail -f /tmp/training.log  # Si hay logs
```

### Ver Uso de GPU

```python
# En una nueva celda:
!watch -n 1 nvidia-smi
```

### Ver Memoria

```python
# En una nueva celda:
!free -h
```

---

## 💾 Descargar Resultados

### Archivos Generados

El notebook genera automáticamente:
- `edge_models_results.zip` - Todo comprimido
- También se guarda en Google Drive

### Cómo Descargar

**Opción 1: Desde Colab**
1. Click en icono de carpeta (Files) a la izquierda
2. Busca `edge_models_results.zip`
3. Click derecho > Download

**Opción 2: Desde Google Drive**
1. Abre Google Drive
2. Busca `edge_models_results.zip` en MyDrive
3. Descarga

---

## 📁 Contenido de Resultados

```
edge_models_results.zip
├── experiments/edge_models/
│   ├── best_edge_model.json       # 🏆 Mejor modelo
│   └── comparison_results.csv     # 📊 Comparación
├── models/exported/
│   ├── MobileNetV3Large_*.keras
│   ├── EfficientNetLiteB2_*.keras
│   ├── MobileViT_*.keras
│   ├── PMVT_*.keras
│   └── *_metadata.json
└── models/mlruns/                 # Experimentos MLflow
```

---

## 🔧 Solución de Problemas

### Error: "No module named 'src'"

```python
# Ejecuta en una celda:
import sys
sys.path.insert(0, '/content/corn-diseases-detection')
```

### Error: "Data not found"

Verifica la ruta en Google Drive:
```python
# Ejecuta en una celda:
!ls -la /content/drive/MyDrive/corn-diseases-data/
```

Ajusta la ruta en la celda 4 si es necesario.

### Error: "Out of memory"

Reduce batch_size en `train_all_models.py`:
```python
# Edita antes de ejecutar:
'batch_size': 16,  # En lugar de 32
```

### Sesión Desconectada

Colab desconecta después de 12 horas o inactividad.

**Prevenir:**
```javascript
// Ejecuta en consola del navegador (F12):
function KeepAlive() {
    console.log("Keeping alive...");
    document.querySelector("colab-connect-button").click();
}
setInterval(KeepAlive, 60000);
```

---

## ⏱️ Tiempos Estimados

| Tarea | Tiempo (GPU T4) | Tiempo (CPU) |
|-------|----------------|--------------|
| Setup inicial | 5 min | 5 min |
| MobileNetV3Large | 30-40 min | 4-6 horas |
| EfficientNetLiteB2 | 40-50 min | 5-7 horas |
| MobileViT + FT | 50-60 min | 6-8 horas |
| PMVT + FT | 50-60 min | 6-8 horas |
| **TOTAL** | **2-3 horas** | **20-30 horas** |

---

## 💡 Tips

### Guardar Checkpoints

Modifica `train_edge_model.py` para guardar checkpoints:
```python
callbacks.append(
    tf.keras.callbacks.ModelCheckpoint(
        'checkpoint.keras',
        save_best_only=True
    )
)
```

### Entrenar Solo 1 Modelo (Prueba)

```python
# En lugar de train_all_models.py:
!python experiments/edge_models/train_edge_model.py \
    --model MobileNetV3Large \
    --epochs 5  # Prueba rápida
```

### Ver Logs en Tiempo Real

```python
# Modifica train_all_models.py:
# Línea 103: capture_output=False -> capture_output=True
```

---

## 🎯 Siguiente Paso

Una vez descargados los resultados:

```bash
# En tu Mac local:
cd /Users/felipe/Downloads/corn-diseases-detection

# Extraer resultados
unzip edge_models_results.zip

# Ver mejor modelo
cat experiments/edge_models/best_edge_model.json

# Iniciar MLflow local
docker-compose --profile mlflow up -d
open http://localhost:5000
```

---

## 📞 Soporte

Si tienes problemas:
1. Revisa los logs en Colab
2. Verifica que la GPU esté activa: `!nvidia-smi`
3. Asegúrate de tener espacio en Drive (>5GB)

---

**¡Listo para entrenar! 🚀**
