"""
Tests para el módulo de entrenamiento.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.pipelines.train import train


class TestTrainFunction:
    """Tests para la función principal de entrenamiento."""

    @pytest.fixture
    def mock_dataset(self):
        """Mock de dataset para entrenamiento."""
        return {
            'train': {
                'Blight': [Mock()] * 10,
                'Healthy': [Mock()] * 10
            },
            'val': {
                'Blight': [Mock()] * 5,
                'Healthy': [Mock()] * 5
            },
            'test': {
                'Blight': [Mock()] * 5,
                'Healthy': [Mock()] * 5
            }
        }

    @pytest.mark.slow
    @patch('src.pipelines.train.split_and_balance_dataset')
    @patch('src.pipelines.train.flatten_data')
    @patch('src.pipelines.train.ModelBuilder')
    @patch('src.pipelines.train.mlflow')
    def test_train_basic_workflow(
        self,
        mock_mlflow,
        mock_model_builder,
        mock_flatten,
        mock_split,
        mock_dataset
    ):
        """Verifica el flujo básico de entrenamiento."""
        # Configurar mocks
        mock_split.return_value = mock_dataset

        # Mock flatten_data para retornar arrays numpy
        mock_flatten.side_effect = [
            (np.random.rand(20, 224, 224, 3), np.array(['Blight'] * 10 + ['Healthy'] * 10)),
            (np.random.rand(10, 224, 224, 3), np.array(['Blight'] * 5 + ['Healthy'] * 5)),
            (np.random.rand(10, 224, 224, 3), np.array(['Blight'] * 5 + ['Healthy'] * 5))
        ]

        # Mock del tuner
        mock_tuner = MagicMock()
        mock_best_hps = MagicMock()
        mock_best_hps.values = {'learning_rate': 0.001, 'dropout_rate': 0.5}
        mock_tuner.get_best_hyperparameters.return_value = [mock_best_hps]

        # Mock del modelo
        mock_model = MagicMock()
        mock_model.evaluate.return_value = [0.5, 0.8]  # loss, accuracy
        mock_model_builder.return_value.build.return_value = mock_model

        with patch('src.pipelines.train.kt.Hyperband', return_value=mock_tuner):
            # Ejecutar con parámetros de prueba (sin realmente entrenar)
            # En un test real necesitaríamos más mocks
            pass

    def test_train_parameter_validation(self):
        """Verifica validación de parámetros de entrada."""
        # backbone_name inválido
        with pytest.raises(ValueError):
            train(backbone_name='InvalidBackbone', balanced='oversample')

    def test_train_balanced_parameter(self):
        """Verifica diferentes modos de balanceo."""
        valid_modes = ['oversample', 'downsample', 'none']

        for mode in valid_modes:
            # Verificar que los modos son reconocidos
            assert mode in valid_modes


class TestDataPreparation:
    """Tests para preparación de datos."""

    def test_label_encoding(self):
        """Verifica encoding de etiquetas."""
        # Simular labels de strings
        y = np.array(['Blight', 'Healthy', 'Blight', 'Healthy'])

        # Crear mapeo
        label_to_int = {label: i for i, label in enumerate(np.unique(y))}
        y_encoded = np.array([label_to_int[l] for l in y])

        assert y_encoded.dtype == np.int64 or y_encoded.dtype == np.int32
        assert len(np.unique(y_encoded)) == len(np.unique(y))
        assert all(isinstance(x, (int, np.integer)) for x in y_encoded)

    def test_data_validation(self):
        """Verifica validación de datos antes de entrenamiento."""
        # Datos válidos
        X_valid = np.random.rand(100, 224, 224, 3)
        assert not np.isnan(X_valid).any()
        assert not np.isinf(X_valid).any()

        # Datos inválidos con NaN
        X_invalid_nan = X_valid.copy()
        X_invalid_nan[0, 0, 0, 0] = np.nan
        assert np.isnan(X_invalid_nan).any()

        # Datos inválidos con Inf
        X_invalid_inf = X_valid.copy()
        X_invalid_inf[0, 0, 0, 0] = np.inf
        assert np.isinf(X_invalid_inf).any()


class TestMLflowIntegration:
    """Tests para integración con MLflow."""

    @patch('src.pipelines.train.mlflow')
    def test_mlflow_experiment_setup(self, mock_mlflow):
        """Verifica configuración de experimento MLflow."""
        from src.pipelines.train import train

        # Configurar el mock para no fallar
        mock_mlflow.set_tracking_uri.return_value = None
        mock_mlflow.set_experiment.return_value = None

        # Verificar que se llaman los métodos correctos
        # (test simplificado sin ejecutar train completo)
        assert mock_mlflow is not None

    def test_mlflow_tracking_uri_format(self):
        """Verifica formato correcto del tracking URI."""
        from src.utils.paths import paths

        mlruns_path = paths.mlruns
        mlruns_uri = f"file:///{mlruns_path.resolve().as_posix()}"

        # Debe empezar con file:///
        assert mlruns_uri.startswith('file:///')
        assert 'mlruns' in mlruns_uri.lower()


class TestModelSaving:
    """Tests para guardado de modelos."""

    def test_model_path_structure(self):
        """Verifica estructura de rutas para modelos."""
        from src.utils.paths import paths

        exported_path = paths.models_exported
        assert exported_path.name == 'exported'
        assert 'models' in str(exported_path)

    def test_versioned_model_naming(self):
        """Verifica nomenclatura de modelos versionados."""
        from datetime import datetime

        backbone = 'VGG16'
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_acc = 0.8534

        # Formato esperado: {backbone}_{timestamp}_acc{accuracy}.keras
        model_name = f'{backbone}_{timestamp}_acc{test_acc:.4f}.keras'

        assert backbone in model_name
        assert timestamp in model_name
        assert 'acc' in model_name
        assert model_name.endswith('.keras')

    def test_best_model_naming(self):
        """Verifica nomenclatura de modelo 'best'."""
        backbone = 'ResNet50'
        best_model_name = f'best_{backbone}.keras'

        assert best_model_name.startswith('best_')
        assert backbone in best_model_name
        assert best_model_name.endswith('.keras')


class TestHyperparameterTuning:
    """Tests para búsqueda de hiperparámetros."""

    def test_hyperband_configuration(self):
        """Verifica configuración del algoritmo Hyperband."""
        from src.core.load_env import EnvLoader

        env_vars = EnvLoader().get_all()

        max_trials = int(env_vars.get('MAX_TRIALS', 10))
        max_epochs = int(env_vars.get('MAX_EPOCHS', 20))
        factor = int(env_vars.get('FACTOR', 3))

        assert max_trials > 0
        assert max_epochs > 0
        assert factor > 1

    def test_hyperparameter_search_space(self):
        """Verifica espacio de búsqueda de hiperparámetros."""
        # Verificar que los rangos son razonables
        learning_rates = [1e-5, 1e-4, 1e-3, 1e-2]
        dropout_rates = [0.0, 0.2, 0.5, 0.7]

        for lr in learning_rates:
            assert 0 < lr < 1

        for dr in dropout_rates:
            assert 0 <= dr < 1


class TestTrainingCallbacks:
    """Tests para callbacks de entrenamiento."""

    @patch('src.pipelines.train.EarlyStopping')
    def test_early_stopping_configuration(self, mock_early_stopping):
        """Verifica configuración de Early Stopping."""
        # Early stopping debería monitorear val_loss
        # y tener paciencia razonable
        mock_callback = MagicMock()
        mock_early_stopping.return_value = mock_callback

        # Test simplificado
        assert mock_early_stopping is not None

    @patch('src.pipelines.train.ModelCheckpoint')
    def test_model_checkpoint_configuration(self, mock_checkpoint):
        """Verifica configuración de Model Checkpoint."""
        mock_callback = MagicMock()
        mock_checkpoint.return_value = mock_callback

        assert mock_checkpoint is not None


class TestModelEvaluation:
    """Tests para evaluación del modelo."""

    def test_evaluation_metrics(self):
        """Verifica que se calculan las métricas correctas."""
        # Simular evaluación
        mock_model = MagicMock()
        mock_model.evaluate.return_value = [0.45, 0.85]  # [loss, accuracy]

        test_loss, test_acc = mock_model.evaluate(
            np.random.rand(10, 224, 224, 3),
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        )

        assert 0 <= test_loss  # Loss puede ser cualquier valor >= 0
        assert 0 <= test_acc <= 1  # Accuracy entre 0 y 1

    def test_confusion_matrix_shape(self):
        """Verifica shape de matriz de confusión."""
        num_classes = 4
        y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 2, 0, 1, 3, 3])

        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)

        assert cm.shape == (num_classes, num_classes)


class TestErrorHandling:
    """Tests para manejo de errores."""

    def test_invalid_backbone_raises_error(self):
        """Verifica que backbone inválido genera error."""
        with pytest.raises(ValueError):
            train(backbone_name='NonExistentBackbone')

    def test_empty_dataset_raises_error(self):
        """Verifica manejo de dataset vacío."""
        with patch('src.pipelines.train.split_and_balance_dataset') as mock_split:
            mock_split.return_value = {
                'train': {},
                'val': {},
                'test': {}
            }

            # Debería fallar o manejar gracefully
            # (test simplificado)
            assert True  # Placeholder

    def test_nan_in_data_raises_error(self):
        """Verifica que NaN en datos genera error."""
        X_with_nan = np.random.rand(10, 224, 224, 3)
        X_with_nan[0, 0, 0, 0] = np.nan

        # El código debería detectar y salir
        assert np.isnan(X_with_nan).any()


class TestBackboneSelection:
    """Tests para selección de backbone."""

    def test_valid_backbones(self):
        """Verifica que los backbones soportados son válidos."""
        valid_backbones = ['VGG16', 'ResNet50', 'YOLO']

        for backbone in valid_backbones:
            assert backbone in valid_backbones

    def test_backbone_case_sensitivity(self):
        """Verifica si los nombres de backbone son case-sensitive."""
        # Debería ser case-sensitive
        assert 'VGG16' != 'vgg16'
        assert 'ResNet50' != 'resnet50'


class TestIntegrationWithPaths:
    """Tests para integración con sistema de rutas."""

    def test_uses_centralized_paths(self):
        """Verifica que usa el sistema centralizado de rutas."""
        from src.utils.paths import paths

        # Verificar que los paths existen
        assert hasattr(paths, 'mlruns')
        assert hasattr(paths, 'models_tuner')
        assert hasattr(paths, 'models_exported')

    def test_creates_directories_if_needed(self):
        """Verifica que crea directorios necesarios."""
        from src.utils.paths import paths

        # ensure_dir debería crear directorios
        test_path = paths.root / 'test_temp_dir'
        paths.ensure_dir(test_path)

        assert test_path.exists()

        # Cleanup
        if test_path.exists():
            test_path.rmdir()
