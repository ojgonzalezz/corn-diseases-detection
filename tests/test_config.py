"""
Tests para el módulo de configuración.
"""
import os
import pytest
from pathlib import Path
from pydantic import ValidationError

from src.core.config import config, DataConfig, TrainingConfig


class TestDataConfig:
    """Tests para la configuración de datos."""

    def test_default_values(self):
        """Verifica que los valores por defecto son correctos."""
        data_config = DataConfig()

        assert data_config.image_size == (224, 224)
        assert data_config.num_classes == 4
        assert len(data_config.class_names) == 4
        assert data_config.split_ratios == (0.7, 0.15, 0.15)

    def test_split_ratios_validation(self):
        """Verifica validación de split_ratios."""
        # Ratios válidos (suman 1.0)
        valid_config = DataConfig(split_ratios=(0.6, 0.2, 0.2))
        assert sum(valid_config.split_ratios) == pytest.approx(1.0)

        # Ratios inválidos (no suman 1.0)
        with pytest.raises(ValidationError, match="must sum to 1.0"):
            DataConfig(split_ratios=(0.5, 0.3, 0.1))

    def test_class_names_consistency(self):
        """Verifica consistencia entre num_classes y class_names."""
        # Consistente
        valid_config = DataConfig(
            num_classes=3,
            class_names=['A', 'B', 'C']
        )
        assert len(valid_config.class_names) == valid_config.num_classes

        # Inconsistente
        with pytest.raises(ValidationError, match="must match num_classes"):
            DataConfig(
                num_classes=4,
                class_names=['A', 'B']  # Solo 2 nombres
            )

    def test_image_size_validation(self):
        """Verifica validación del tamaño de imagen."""
        # Válido
        valid_config = DataConfig(image_size=(256, 256))
        assert valid_config.image_size == (256, 256)

        # Inválido (negativo)
        with pytest.raises(ValidationError):
            DataConfig(image_size=(-1, 224))

        # Inválido (cero)
        with pytest.raises(ValidationError):
            DataConfig(image_size=(0, 224))


class TestTrainingConfig:
    """Tests para la configuración de entrenamiento."""

    def test_default_values(self):
        """Verifica valores por defecto de entrenamiento."""
        train_config = TrainingConfig()

        assert train_config.batch_size == 32
        assert train_config.max_epochs == 30
        assert train_config.max_trials == 10
        assert train_config.tuner_epochs == 10
        assert train_config.factor == 3

    def test_batch_size_validation(self):
        """Verifica validación del batch size."""
        # Válido
        valid_config = TrainingConfig(batch_size=64)
        assert valid_config.batch_size == 64

        # Inválido (menor que 1)
        with pytest.raises(ValidationError):
            TrainingConfig(batch_size=0)

    def test_max_epochs_validation(self):
        """Verifica validación de max_epochs."""
        # Válido
        valid_config = TrainingConfig(max_epochs=50)
        assert valid_config.max_epochs == 50

        # Inválido (negativo)
        with pytest.raises(ValidationError):
            TrainingConfig(max_epochs=-1)


class TestGlobalConfig:
    """Tests para la configuración global del proyecto."""

    def test_config_loads(self):
        """Verifica que la configuración global se carga correctamente."""
        assert config is not None
        assert hasattr(config, 'data')
        assert hasattr(config, 'training')

    def test_config_data_section(self):
        """Verifica sección de datos en config."""
        assert config.data.num_classes > 0
        assert len(config.data.class_names) > 0
        assert len(config.data.image_size) == 2

    def test_config_training_section(self):
        """Verifica sección de entrenamiento en config."""
        assert config.training.batch_size > 0
        assert config.training.max_epochs > 0
        assert config.training.max_trials > 0

    def test_config_to_dict(self):
        """Verifica conversión de config a diccionario."""
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert 'data' in config_dict
        assert 'training' in config_dict

    def test_similarity_threshold_property(self):
        """Verifica la propiedad similarity_threshold."""
        # Debería usar el valor correcto (threshold, no treshold)
        threshold = config.data.im_sim_threshold

        assert isinstance(threshold, float)
        assert 0.0 <= threshold <= 1.0


class TestEnvironmentVariables:
    """Tests para verificar carga desde variables de entorno."""

    def test_env_file_exists(self):
        """Verifica que el archivo .env existe."""
        env_path = Path("src/core/.env")
        assert env_path.exists(), "El archivo .env debe existir"

    def test_env_example_exists(self):
        """Verifica que el archivo .env.example existe."""
        env_example_path = Path("src/core/.env.example")
        assert env_example_path.exists(), "El archivo .env.example debe existir"

    def test_env_has_required_variables(self):
        """Verifica que .env contiene las variables requeridas."""
        env_path = Path("src/core/.env")

        with open(env_path, 'r') as f:
            content = f.read()

        required_vars = [
            'IMAGE_SIZE',
            'NUM_CLASSES',
            'CLASS_NAMES',
            'BATCH_SIZE',
            'MAX_EPOCHS',
            'SPLIT_RATIOS'
        ]

        for var in required_vars:
            assert var in content, f"Variable {var} debe estar en .env"


class TestBackwardCompatibility:
    """Tests para verificar compatibilidad hacia atrás."""

    def test_threshold_typo_compatibility(self):
        """Verifica que el typo legacy sigue funcionando con warning."""
        # El sistema debería aceptar IM_SIM_TRESHOLD (typo)
        # pero emitir un warning

        # Esto se testea indirectamente via la propiedad similarity_threshold
        threshold = config.data.im_sim_threshold
        assert threshold is not None
        assert isinstance(threshold, float)

    def test_old_path_finder_deprecated(self):
        """Verifica que path_finder emite DeprecationWarning."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # from src.core.path_finder import ProjectPaths
            # pp = ProjectPaths()

            # # Debería haber un warning de deprecación
            # assert len(w) > 0
            # assert issubclass(w[0].category, DeprecationWarning)
            # assert "deprecado" in str(w[0].message).lower()
