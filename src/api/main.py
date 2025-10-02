"""
API de inferencia para el modelo de detección de enfermedades del maíz.

Este módulo proporciona endpoints REST para realizar predicciones sobre imágenes
de hojas de maíz utilizando el modelo entrenado.
"""
import io
from typing import Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.pipelines.infer import predict
from src.core.config import config
from src.utils.logger import get_logger

# Configurar logger
logger = get_logger(__name__)

# Crear instancia de FastAPI
app = FastAPI(
    title="Corn Diseases Detection API",
    description="API para clasificación de enfermedades en hojas de maíz usando Deep Learning",
    version=config.project.version,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> Dict[str, str]:
    """
    Endpoint raíz que proporciona información básica de la API.

    Returns:
        Dict con mensaje de bienvenida y versión.
    """
    return {
        "message": "Corn Diseases Detection API",
        "version": config.project.version,
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint para verificar el estado del servicio.

    Returns:
        Dict con el estado del servicio.
    """
    return {
        "status": "healthy",
        "service": "corn-diseases-detection",
        "version": config.project.version
    }


@app.get("/info")
async def get_model_info() -> Dict[str, Any]:
    """
    Endpoint que proporciona información sobre el modelo cargado.

    Returns:
        Dict con información del modelo y clases.
    """
    return {
        "model": {
            "backbone": config.training.backbone,
            "image_size": config.data.image_size,
            "num_classes": config.data.num_classes,
        },
        "classes": config.data.class_names,
        "version": config.project.version
    }


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)) -> JSONResponse:
    """
    Endpoint principal de predicción que clasifica una imagen de hoja de maíz.

    Args:
        file: Archivo de imagen (jpg, png) enviado como multipart/form-data.

    Returns:
        JSONResponse con la predicción y probabilidades.

    Raises:
        HTTPException: Si hay un error en la predicción o el archivo no es válido.

    Example:
        ```bash
        curl -X POST "http://localhost:8000/predict" \
             -F "file=@imagen_hoja.jpg"
        ```
    """
    # Validar tipo de archivo
    if not file.content_type.startswith("image/"):
        logger.warning(f"Tipo de archivo inválido: {file.content_type}")
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de archivo no soportado: {file.content_type}. "
                   f"Solo se aceptan imágenes (jpg, png)."
        )

    try:
        # Leer contenido del archivo
        contents = await file.read()
        logger.info(f"Procesando imagen: {file.filename} ({len(contents)} bytes)")

        # Realizar predicción
        result = predict(contents)

        # Verificar si hubo error en la predicción
        if "error" in result:
            logger.error(f"Error en predicción: {result['error']}")
            raise HTTPException(
                status_code=500,
                detail=result.get("message", "Error durante la predicción")
            )

        # Log de resultado exitoso
        predicted_label = result.get("predicted_label", "unknown")
        confidence = result.get("confidence", 0.0)
        logger.info(f"Predicción exitosa: {predicted_label} (confianza: {confidence:.2%})")

        # Retornar resultado
        return JSONResponse(
            content={
                "success": True,
                "filename": file.filename,
                "prediction": result
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error inesperado en predicción: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error interno del servidor: {str(e)}"
        )


@app.post("/batch-predict")
async def batch_predict(files: list[UploadFile] = File(...)) -> JSONResponse:
    """
    Endpoint para predicción por lotes (múltiples imágenes).

    Args:
        files: Lista de archivos de imagen enviados como multipart/form-data.

    Returns:
        JSONResponse con las predicciones para cada imagen.

    Raises:
        HTTPException: Si hay un error en el procesamiento.

    Example:
        ```bash
        curl -X POST "http://localhost:8000/batch-predict" \
             -F "files=@imagen1.jpg" \
             -F "files=@imagen2.jpg"
        ```
    """
    if len(files) == 0:
        raise HTTPException(
            status_code=400,
            detail="No se proporcionaron archivos"
        )

    if len(files) > 50:
        raise HTTPException(
            status_code=400,
            detail="Máximo 50 imágenes por lote"
        )

    results = []
    errors = []

    for idx, file in enumerate(files):
        try:
            # Validar tipo de archivo
            if not file.content_type.startswith("image/"):
                errors.append({
                    "filename": file.filename,
                    "error": f"Tipo de archivo no soportado: {file.content_type}"
                })
                continue

            # Leer y predecir
            contents = await file.read()
            result = predict(contents)

            if "error" in result:
                errors.append({
                    "filename": file.filename,
                    "error": result.get("message", "Error desconocido")
                })
            else:
                results.append({
                    "filename": file.filename,
                    "prediction": result
                })

        except Exception as e:
            errors.append({
                "filename": file.filename,
                "error": str(e)
            })

    logger.info(f"Batch predict: {len(results)} exitosas, {len(errors)} fallidas")

    return JSONResponse(
        content={
            "success": len(errors) == 0,
            "total": len(files),
            "successful": len(results),
            "failed": len(errors),
            "results": results,
            "errors": errors if errors else None
        }
    )


# Punto de entrada para ejecución directa
if __name__ == "__main__":
    logger.info("Iniciando servidor de inferencia...")
    logger.info(f"Documentación disponible en: http://localhost:8000/docs")

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # En producción, usar False
        log_level="info"
    )


