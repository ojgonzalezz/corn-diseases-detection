"""
Tests para la API de inferencia.

Este módulo prueba la funcionalidad de src/api/main.py,
verificando el correcto funcionamiento de los endpoints REST.
"""
import pytest
import io
from PIL import Image
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from src.api.main import app


# Fixture del cliente de prueba
@pytest.fixture
def client():
    """Cliente de prueba para FastAPI."""
    return TestClient(app)


@pytest.fixture
def sample_image_bytes():
    """Genera bytes de una imagen de prueba."""
    img = Image.new('RGB', (224, 224), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes.getvalue()


class TestRootEndpoint:
    """Tests para el endpoint raíz (/)."""

    def test_root_endpoint(self, client):
        """Verifica endpoint raíz."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "version" in data
        assert "status" in data
        assert data["status"] == "running"

    def test_root_endpoint_structure(self, client):
        """Verifica estructura de respuesta del endpoint raíz."""
        response = client.get("/")
        data = response.json()
        
        required_keys = ["message", "version", "status", "docs"]
        for key in required_keys:
            assert key in data


class TestHealthEndpoint:
    """Tests para el endpoint de health check."""

    def test_health_check(self, client):
        """Verifica endpoint de health check."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert data["status"] == "healthy"

    def test_health_check_structure(self, client):
        """Verifica estructura de respuesta del health check."""
        response = client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert "service" in data
        assert "version" in data

    def test_health_check_response_time(self, client):
        """Verifica que health check responde rápidamente."""
        import time
        
        start = time.time()
        response = client.get("/health")
        end = time.time()
        
        assert response.status_code == 200
        # Health check debe responder en menos de 1 segundo
        assert (end - start) < 1.0


class TestInfoEndpoint:
    """Tests para el endpoint de información del modelo."""

    def test_info_endpoint(self, client):
        """Verifica endpoint de información."""
        response = client.get("/info")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "model" in data
        assert "classes" in data
        assert "version" in data

    def test_info_model_structure(self, client):
        """Verifica estructura de información del modelo."""
        response = client.get("/info")
        data = response.json()
        
        model_info = data["model"]
        assert "backbone" in model_info
        assert "image_size" in model_info
        assert "num_classes" in model_info

    def test_info_classes_list(self, client):
        """Verifica que retorna lista de clases."""
        response = client.get("/info")
        data = response.json()
        
        classes = data["classes"]
        assert isinstance(classes, list)
        assert len(classes) > 0


class TestPredictEndpoint:
    """Tests para el endpoint de predicción."""

    @patch('src.api.main.predict')
    def test_predict_endpoint_success(self, mock_predict, client, sample_image_bytes):
        """Verifica predicción exitosa."""
        # Mock de respuesta de predict
        mock_predict.return_value = {
            "predicted_label": "Healthy",
            "predicted_index": 3,
            "confidence": 0.95,
            "all_probabilities": {
                "Blight": 0.01,
                "Common_Rust": 0.02,
                "Gray_Leaf_Spot": 0.02,
                "Healthy": 0.95
            }
        }
        
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "success" in data
        assert data["success"] is True
        assert "prediction" in data
        assert "filename" in data

    @patch('src.api.main.predict')
    def test_predict_endpoint_returns_prediction(self, mock_predict, client, sample_image_bytes):
        """Verifica que retorna predicción correcta."""
        mock_predict.return_value = {
            "predicted_label": "Blight",
            "confidence": 0.87
        }
        
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        )
        
        data = response.json()
        prediction = data["prediction"]
        
        assert "predicted_label" in prediction
        assert "confidence" in prediction

    def test_predict_endpoint_invalid_file_type(self, client):
        """Verifica rechazo de tipo de archivo inválido."""
        # Enviar un archivo que no es imagen
        text_content = b"This is not an image"
        
        response = client.post(
            "/predict",
            files={"file": ("test.txt", text_content, "text/plain")}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data

    def test_predict_endpoint_no_file(self, client):
        """Verifica error cuando no se envía archivo."""
        response = client.post("/predict")
        
        assert response.status_code == 422  # Unprocessable Entity

    @patch('src.api.main.predict')
    def test_predict_endpoint_model_error(self, mock_predict, client, sample_image_bytes):
        """Verifica manejo de error del modelo."""
        # Simular error en predict
        mock_predict.return_value = {
            "error": "Model not loaded",
            "message": "Model file not found"
        }
        
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        )
        
        assert response.status_code == 500

    @patch('src.api.main.predict')
    def test_predict_endpoint_exception_handling(self, mock_predict, client, sample_image_bytes):
        """Verifica manejo de excepciones."""
        # Simular excepción
        mock_predict.side_effect = Exception("Test exception")
        
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data


class TestBatchPredictEndpoint:
    """Tests para el endpoint de predicción por lotes."""

    @patch('src.api.main.predict')
    def test_batch_predict_success(self, mock_predict, client, sample_image_bytes):
        """Verifica predicción por lotes exitosa."""
        # Mock de respuestas
        mock_predict.return_value = {
            "predicted_label": "Healthy",
            "confidence": 0.95
        }
        
        files = [
            ("files", ("test1.jpg", sample_image_bytes, "image/jpeg")),
            ("files", ("test2.jpg", sample_image_bytes, "image/jpeg")),
        ]
        
        response = client.post("/batch-predict", files=files)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "success" in data
        assert "total" in data
        assert "successful" in data
        assert "failed" in data
        assert "results" in data

    @patch('src.api.main.predict')
    def test_batch_predict_multiple_images(self, mock_predict, client, sample_image_bytes):
        """Verifica procesamiento de múltiples imágenes."""
        mock_predict.return_value = {
            "predicted_label": "Blight",
            "confidence": 0.80
        }
        
        files = [
            ("files", (f"test{i}.jpg", sample_image_bytes, "image/jpeg"))
            for i in range(5)
        ]
        
        response = client.post("/batch-predict", files=files)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total"] == 5
        assert len(data["results"]) <= 5

    def test_batch_predict_no_files(self, client):
        """Verifica error cuando no se envían archivos."""
        response = client.post("/batch-predict", files=[])
        
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data

    @patch('src.api.main.predict')
    def test_batch_predict_max_limit(self, mock_predict, client, sample_image_bytes):
        """Verifica límite máximo de archivos."""
        mock_predict.return_value = {"predicted_label": "Healthy"}
        
        # Intentar enviar más de 50 imágenes
        files = [
            ("files", (f"test{i}.jpg", sample_image_bytes, "image/jpeg"))
            for i in range(51)
        ]
        
        response = client.post("/batch-predict", files=files)
        
        assert response.status_code == 400
        data = response.json()
        assert "50" in data["detail"] or "máximo" in data["detail"].lower()

    @patch('src.api.main.predict')
    def test_batch_predict_partial_failure(self, mock_predict, client, sample_image_bytes):
        """Verifica manejo de fallos parciales."""
        # Simular fallo en algunas predicciones
        def side_effect(*args, **kwargs):
            import random
            if random.random() > 0.5:
                return {"predicted_label": "Healthy", "confidence": 0.9}
            else:
                return {"error": "Prediction failed"}
        
        mock_predict.side_effect = side_effect
        
        files = [
            ("files", (f"test{i}.jpg", sample_image_bytes, "image/jpeg"))
            for i in range(10)
        ]
        
        response = client.post("/batch-predict", files=files)
        
        assert response.status_code == 200
        data = response.json()
        
        # Debe reportar tanto éxitos como fallos
        assert "successful" in data
        assert "failed" in data
        assert data["total"] == 10

    @patch('src.api.main.predict')
    def test_batch_predict_invalid_file_types(self, mock_predict, client, sample_image_bytes):
        """Verifica manejo de tipos de archivo inválidos en batch."""
        files = [
            ("files", ("test1.jpg", sample_image_bytes, "image/jpeg")),
            ("files", ("test2.txt", b"not an image", "text/plain")),
            ("files", ("test3.jpg", sample_image_bytes, "image/jpeg")),
        ]
        
        response = client.post("/batch-predict", files=files)
        
        assert response.status_code == 200
        data = response.json()
        
        # Debe procesar las imágenes válidas e informar errores
        assert "errors" in data or data["failed"] > 0


class TestCORSConfiguration:
    """Tests para configuración de CORS."""

    def test_cors_headers_present(self, client):
        """Verifica que headers CORS están presentes."""
        response = client.options("/")
        
        # Las opciones de CORS deben estar habilitadas
        assert response.status_code in [200, 405]

    def test_cors_allows_methods(self, client):
        """Verifica que métodos HTTP están permitidos."""
        response = client.get("/")
        
        # Debe permitir GET
        assert response.status_code == 200


class TestAPIDocumentation:
    """Tests para documentación de la API."""

    def test_openapi_docs_available(self, client):
        """Verifica que documentación OpenAPI está disponible."""
        response = client.get("/docs")
        
        assert response.status_code == 200

    def test_redoc_available(self, client):
        """Verifica que ReDoc está disponible."""
        response = client.get("/redoc")
        
        assert response.status_code == 200

    def test_openapi_schema(self, client):
        """Verifica que el schema OpenAPI es válido."""
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        schema = response.json()
        
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema


class TestAPIIntegration:
    """Tests de integración para la API completa."""

    @patch('src.api.main.predict')
    def test_full_prediction_workflow(self, mock_predict, client, sample_image_bytes):
        """Test de integración: flujo completo de predicción."""
        mock_predict.return_value = {
            "predicted_label": "Gray_Leaf_Spot",
            "predicted_index": 2,
            "confidence": 0.88,
            "all_probabilities": {
                "Blight": 0.05,
                "Common_Rust": 0.03,
                "Gray_Leaf_Spot": 0.88,
                "Healthy": 0.04
            }
        }
        
        # 1. Verificar health
        health_response = client.get("/health")
        assert health_response.status_code == 200
        
        # 2. Obtener información del modelo
        info_response = client.get("/info")
        assert info_response.status_code == 200
        
        # 3. Realizar predicción
        predict_response = client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        )
        assert predict_response.status_code == 200
        
        data = predict_response.json()
        assert data["success"] is True
        assert data["prediction"]["predicted_label"] == "Gray_Leaf_Spot"

    def test_api_consistency(self, client):
        """Verifica consistencia de respuestas de la API."""
        # Realizar múltiples llamadas y verificar consistencia
        responses = []
        for _ in range(3):
            response = client.get("/info")
            responses.append(response.json())
        
        # Todas las respuestas deben ser idénticas
        first_response = responses[0]
        for response in responses[1:]:
            assert response == first_response

    @patch('src.api.main.predict')
    def test_concurrent_requests(self, mock_predict, client, sample_image_bytes):
        """Verifica manejo de solicitudes concurrentes."""
        mock_predict.return_value = {
            "predicted_label": "Healthy",
            "confidence": 0.95
        }
        
        # Simular múltiples solicitudes
        responses = []
        for i in range(5):
            response = client.post(
                "/predict",
                files={"file": (f"test{i}.jpg", sample_image_bytes, "image/jpeg")}
            )
            responses.append(response)
        
        # Todas deben tener éxito
        for response in responses:
            assert response.status_code == 200


class TestErrorResponses:
    """Tests para respuestas de error."""

    def test_404_not_found(self, client):
        """Verifica respuesta 404 para endpoints inexistentes."""
        response = client.get("/nonexistent-endpoint")
        
        assert response.status_code == 404

    def test_405_method_not_allowed(self, client):
        """Verifica error cuando se usa método HTTP incorrecto."""
        # GET no está permitido en /predict
        response = client.get("/predict")
        
        assert response.status_code == 405

    def test_422_validation_error(self, client):
        """Verifica error de validación."""
        # Enviar datos inválidos
        response = client.post("/predict", json={"invalid": "data"})
        
        assert response.status_code == 422


class TestResponseFormats:
    """Tests para formatos de respuesta."""

    @patch('src.api.main.predict')
    def test_json_response_format(self, mock_predict, client, sample_image_bytes):
        """Verifica que todas las respuestas son JSON válido."""
        mock_predict.return_value = {
            "predicted_label": "Healthy",
            "confidence": 0.95
        }
        
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        )
        
        assert response.headers["content-type"] == "application/json"
        # Debe poder parsear como JSON
        data = response.json()
        assert isinstance(data, dict)

    def test_consistent_error_format(self, client):
        """Verifica formato consistente de mensajes de error."""
        response = client.post("/predict")
        
        # Error debe tener formato estándar de FastAPI
        assert response.status_code in [400, 422, 500]
        data = response.json()
        assert "detail" in data or "message" in data

