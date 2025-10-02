"""
Tests para el módulo de inferencia.
"""
import io
import pytest
import numpy as np
from PIL import Image
from pathlib import Path

# Importar después de definir el mock para evitar cargar el modelo real
import sys
from unittest.mock import Mock, patch, MagicMock


class TestImagePreprocessing:
    """Tests para el preprocesamiento de imágenes."""

    @pytest.fixture
    def sample_image_bytes(self):
        """Genera bytes de imagen de prueba."""
        # Crear imagen RGB de prueba
        img = Image.new('RGB', (224, 224), color='red')
        byte_arr = io.BytesIO()
        img.save(byte_arr, format='PNG')
        return byte_arr.getvalue()

    def test_preprocess_image_shape(self, sample_image_bytes):
        """Verifica que preprocess_image retorna el shape correcto."""
        from src.pipelines.infer import preprocess_image

        result = preprocess_image(sample_image_bytes)

        assert result.shape == (1, 224, 224, 3)
        assert result.dtype == np.float32 or result.dtype == np.float64

    def test_preprocess_image_normalization(self, sample_image_bytes):
        """Verifica que los valores están normalizados [0, 1]."""
        from src.pipelines.infer import preprocess_image

        result = preprocess_image(sample_image_bytes)

        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_preprocess_different_sizes(self):
        """Verifica que imágenes de diferentes tamaños se redimensionan."""
        from src.pipelines.infer import preprocess_image

        # Crear imagen de tamaño diferente
        img = Image.new('RGB', (512, 512), color='blue')
        byte_arr = io.BytesIO()
        img.save(byte_arr, format='PNG')
        image_bytes = byte_arr.getvalue()

        result = preprocess_image(image_bytes)

        # Debe redimensionarse a IMG_SIZE (224, 224)
        assert result.shape == (1, 224, 224, 3)

    def test_preprocess_grayscale_conversion(self):
        """Verifica conversión de escala de grises a RGB."""
        from src.pipelines.infer import preprocess_image

        # Crear imagen en escala de grises
        img = Image.new('L', (224, 224), color=128)
        byte_arr = io.BytesIO()
        img.save(byte_arr, format='PNG')
        image_bytes = byte_arr.getvalue()

        result = preprocess_image(image_bytes)

        # Debe convertirse a RGB (3 canales)
        assert result.shape == (1, 224, 224, 3)


class TestPredictFunction:
    """Tests para la función de predicción."""

    @pytest.fixture
    def sample_image_bytes(self):
        """Genera bytes de imagen de prueba."""
        img = Image.new('RGB', (224, 224), color='green')
        byte_arr = io.BytesIO()
        img.save(byte_arr, format='PNG')
        return byte_arr.getvalue()

    @pytest.fixture
    def mock_model(self):
        """Mock de modelo de TensorFlow."""
        model = MagicMock()
        # Simular predicción con 4 clases
        model.predict.return_value = np.array([[0.1, 0.6, 0.2, 0.1]])
        return model

    def test_predict_no_model_loaded(self, sample_image_bytes):
        """Verifica comportamiento cuando no hay modelo cargado."""
        # Forzar _model a None
        with patch('src.pipelines.infer._model', None):
            from src.pipelines.infer import predict

            result = predict(sample_image_bytes)

            assert 'error' in result
            assert result['error'] == 'Model not loaded'

    def test_predict_returns_expected_structure(self, sample_image_bytes, mock_model):
        """Verifica que predict retorna la estructura esperada."""
        with patch('src.pipelines.infer._model', mock_model):
            with patch('src.pipelines.infer._labels', ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']):
                from src.pipelines.infer import predict

                result = predict(sample_image_bytes)

                # Verificar estructura del resultado
                assert 'predicted_label' in result
                assert 'predicted_index' in result
                assert 'confidence' in result
                assert 'all_probabilities' in result

    def test_predict_selects_max_probability(self, sample_image_bytes, mock_model):
        """Verifica que se selecciona la clase con mayor probabilidad."""
        with patch('src.pipelines.infer._model', mock_model):
            with patch('src.pipelines.infer._labels', ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']):
                from src.pipelines.infer import predict

                result = predict(sample_image_bytes)

                # El mock retorna [0.1, 0.6, 0.2, 0.1], índice 1 es el máximo
                assert result['predicted_index'] == 1
                assert result['predicted_label'] == 'Common_Rust'
                assert result['confidence'] == pytest.approx(0.6)

    def test_predict_probabilities_sum_to_one(self, sample_image_bytes, mock_model):
        """Verifica que las probabilidades suman aproximadamente 1."""
        with patch('src.pipelines.infer._model', mock_model):
            with patch('src.pipelines.infer._labels', ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']):
                from src.pipelines.infer import predict

                result = predict(sample_image_bytes)

                total_prob = sum(result['all_probabilities'].values())
                assert total_prob == pytest.approx(1.0, abs=0.01)

    def test_predict_handles_exception(self, sample_image_bytes):
        """Verifica manejo de excepciones durante predicción."""
        mock_model_error = MagicMock()
        mock_model_error.predict.side_effect = Exception("Test error")

        with patch('src.pipelines.infer._model', mock_model_error):
            from src.pipelines.infer import predict

            result = predict(sample_image_bytes)

            assert 'error' in result
            assert 'message' in result


class TestModelLoading:
    """Tests para la carga del modelo."""

    def test_model_path_configuration(self):
        """Verifica que MODEL_PATH usa el sistema de rutas centralizado."""
        from src.pipelines.infer import MODEL_PATH

        # MODEL_PATH debería usar paths.get_model_path()
        if MODEL_PATH is not None:
            assert isinstance(MODEL_PATH, Path) or isinstance(MODEL_PATH, type(Path()))
            assert 'models' in str(MODEL_PATH).lower()

    def test_labels_loaded_from_config(self):
        """Verifica que las etiquetas se cargan desde configuración."""
        from src.pipelines.infer import _labels, NUM_CLASSES

        if _labels is not None:
            assert isinstance(_labels, list)
            assert len(_labels) == NUM_CLASSES


class TestInferenceConsistency:
    """Tests para verificar consistencia en inferencia."""

    @pytest.fixture
    def sample_image_bytes(self):
        """Genera bytes de imagen de prueba."""
        img = Image.new('RGB', (224, 224), color='blue')
        byte_arr = io.BytesIO()
        img.save(byte_arr, format='PNG')
        return byte_arr.getvalue()

    def test_same_image_same_prediction(self, sample_image_bytes):
        """Verifica que la misma imagen produce la misma predicción."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.3, 0.5, 0.1, 0.1]])

        with patch('src.pipelines.infer._model', mock_model):
            with patch('src.pipelines.infer._labels', ['A', 'B', 'C', 'D']):
                from src.pipelines.infer import predict

                result1 = predict(sample_image_bytes)
                result2 = predict(sample_image_bytes)

                assert result1['predicted_index'] == result2['predicted_index']
                assert result1['confidence'] == result2['confidence']

    def test_num_classes_consistency(self):
        """Verifica consistencia entre NUM_CLASSES y predicción."""
        from src.pipelines.infer import NUM_CLASSES

        assert NUM_CLASSES == 4  # Según .env por defecto


class TestEdgeCases:
    """Tests para casos extremos."""

    def test_corrupted_image_bytes(self):
        """Verifica manejo de bytes de imagen corruptos."""
        from src.pipelines.infer import preprocess_image

        corrupted_bytes = b'not an image'

        with pytest.raises(Exception):
            preprocess_image(corrupted_bytes)

    def test_empty_image_bytes(self):
        """Verifica manejo de bytes vacíos."""
        from src.pipelines.infer import preprocess_image

        empty_bytes = b''

        with pytest.raises(Exception):
            preprocess_image(empty_bytes)

    def test_predict_with_invalid_model_output(self):
        """Verifica comportamiento con output de modelo inválido."""
        mock_model_invalid = MagicMock()
        # Retornar array vacío
        mock_model_invalid.predict.return_value = np.array([[]])

        sample_img = Image.new('RGB', (224, 224))
        byte_arr = io.BytesIO()
        sample_img.save(byte_arr, format='PNG')

        with patch('src.pipelines.infer._model', mock_model_invalid):
            from src.pipelines.infer import predict

            result = predict(byte_arr.getvalue())

            # Debería manejar el error gracefully
            assert 'error' in result or 'predicted_label' in result
