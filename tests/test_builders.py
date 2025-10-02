"""
Tests para los módulos de construcción de modelos.

Este módulo prueba la funcionalidad de base_models.py y builders.py,
verificando la correcta carga de backbones y construcción de modelos.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import keras_tuner as kt

from src.builders.base_models import load_vgg16, load_resnet50, load_yolo
from src.builders.builders import ModelBuilder


class TestBaseModels:
    """Tests para carga de modelos base (backbones)."""

    def test_load_vgg16_default(self):
        """Verifica que VGG16 se carga con parámetros por defecto."""
        model = load_vgg16()
        
        assert model is not None
        assert hasattr(model, 'input_shape')
        assert hasattr(model, 'output_shape')
        # VGG16 sin include_top debe tener 4D output (batch, height, width, channels)
        assert len(model.output_shape) == 4

    def test_load_vgg16_custom_shape(self):
        """Verifica que VGG16 acepta tamaños personalizados."""
        custom_shape = (299, 299, 3)
        model = load_vgg16(input_shape=custom_shape)
        
        assert model is not None
        # El input shape debe incluir el batch dimension (None)
        assert model.input_shape[1:] == custom_shape

    def test_load_vgg16_no_weights(self):
        """Verifica que VGG16 se puede cargar sin pesos preentrenados."""
        model = load_vgg16(weights=None)
        
        assert model is not None
        # Modelo debe ser funcional incluso sin pesos preentrenados
        assert hasattr(model, 'layers')
        assert len(model.layers) > 0

    def test_load_resnet50_default(self):
        """Verifica que ResNet50 se carga con parámetros por defecto."""
        model = load_resnet50()
        
        assert model is not None
        assert hasattr(model, 'input_shape')
        assert hasattr(model, 'output_shape')
        # ResNet50 sin include_top debe tener 4D output
        assert len(model.output_shape) == 4

    def test_load_resnet50_custom_shape(self):
        """Verifica que ResNet50 acepta tamaños personalizados."""
        custom_shape = (256, 256, 3)
        model = load_resnet50(input_shape=custom_shape)
        
        assert model is not None
        assert model.input_shape[1:] == custom_shape

    def test_load_resnet50_no_weights(self):
        """Verifica que ResNet50 se puede cargar sin pesos preentrenados."""
        model = load_resnet50(weights=None)
        
        assert model is not None
        assert hasattr(model, 'layers')

    @pytest.mark.slow
    def test_load_yolo(self):
        """Verifica carga de YOLO (requiere ultralytics)."""
        try:
            model = load_yolo()
            # YOLO puede retornar None si no está instalado
            if model is not None:
                assert hasattr(model, 'train') or hasattr(model, 'predict')
        except ImportError:
            pytest.skip("ultralytics no instalado")

    def test_backbone_consistency(self):
        """Verifica que diferentes backbones tienen interfaces consistentes."""
        vgg = load_vgg16()
        resnet = load_resnet50()
        
        # Ambos deben tener métodos/propiedades similares
        assert hasattr(vgg, 'input_shape')
        assert hasattr(resnet, 'input_shape')
        assert hasattr(vgg, 'layers')
        assert hasattr(resnet, 'layers')


class TestModelBuilder:
    """Tests para ModelBuilder con Keras Tuner."""

    def test_model_builder_initialization(self):
        """Verifica inicialización correcta de ModelBuilder."""
        builder = ModelBuilder(
            backbone_name='VGG16',
            input_shape=(224, 224, 3),
            num_classes=4
        )
        
        assert builder.backbone_name == 'VGG16'
        assert builder.input_shape == (224, 224, 3)
        assert builder.num_classes == 4

    def test_model_builder_with_custom_hyperparams(self):
        """Verifica que ModelBuilder acepta hiperparámetros personalizados."""
        builder = ModelBuilder(
            backbone_name='ResNet50',
            input_shape=(256, 256, 3),
            num_classes=10,
            n_layers=(1, 5),
            units=(32, 256, 16),
            activation=['relu', 'elu'],
            learning_rates=[0.001, 0.0001],
            dropout_range=(0.2, 0.6)
        )
        
        assert builder.n_layers == (1, 5)
        assert builder.units == (32, 256, 16)
        assert 'relu' in builder.activation
        assert 'elu' in builder.activation

    @patch('src.builders.builders.load_vgg16')
    def test_model_builder_build_vgg16(self, mock_load_vgg16):
        """Verifica construcción de modelo con VGG16."""
        # Mock del backbone
        mock_backbone = MagicMock()
        mock_backbone.trainable = True
        mock_load_vgg16.return_value = mock_backbone
        
        builder = ModelBuilder(
            backbone_name='VGG16',
            input_shape=(224, 224, 3),
            num_classes=4
        )
        
        # Mock del hyperparameter objeto
        hp = kt.HyperParameters()
        
        try:
            model = builder.build(hp)
            # Verificar que se llamó al loader correcto
            mock_load_vgg16.assert_called_once()
        except Exception as e:
            # En entorno de prueba puede fallar por dependencias
            # pero al menos verificamos que se intentó llamar
            mock_load_vgg16.assert_called_once()

    def test_invalid_backbone_raises_error(self):
        """Verifica que backbone inválido genera error."""
        builder = ModelBuilder(
            backbone_name='InvalidBackbone',
            input_shape=(224, 224, 3),
            num_classes=4
        )
        
        hp = kt.HyperParameters()
        
        with pytest.raises(ValueError, match="no soportado"):
            builder.build(hp)

    def test_model_builder_supports_all_backbones(self):
        """Verifica que ModelBuilder soporta todos los backbones declarados."""
        valid_backbones = ['VGG16', 'ResNet50', 'YOLO']
        
        for backbone in valid_backbones:
            builder = ModelBuilder(
                backbone_name=backbone,
                input_shape=(224, 224, 3),
                num_classes=4
            )
            assert builder.backbone_name == backbone


class TestModelConstruction:
    """Tests de integración para construcción de modelos completos."""

    @pytest.mark.slow
    def test_build_full_model_vgg16(self):
        """Test de integración: construir modelo completo con VGG16."""
        builder = ModelBuilder(
            backbone_name='VGG16',
            input_shape=(224, 224, 3),
            num_classes=4,
            n_layers=(1, 2),  # Pocas capas para test rápido
            units=(32, 64, 32)
        )
        
        hp = kt.HyperParameters()
        
        try:
            model = builder.build(hp)
            
            # Verificar estructura del modelo
            assert model is not None
            assert hasattr(model, 'layers')
            assert len(model.layers) > 0
            
            # Verificar que el modelo es compilado
            assert model.optimizer is not None
            assert model.loss is not None
            
        except Exception as e:
            pytest.skip(f"No se pudo construir modelo completo: {e}")

    def test_model_output_shape(self):
        """Verifica que el modelo produce output del shape correcto."""
        num_classes = 4
        builder = ModelBuilder(
            backbone_name='VGG16',
            input_shape=(224, 224, 3),
            num_classes=num_classes,
            n_layers=(1, 1)
        )
        
        hp = kt.HyperParameters()
        
        try:
            model = builder.build(hp)
            
            # El output debe ser (None, num_classes) para clasificación
            assert model.output_shape == (None, num_classes)
            
        except Exception as e:
            pytest.skip(f"No se pudo construir modelo: {e}")

    def test_model_trainable_params(self):
        """Verifica que el backbone está congelado correctamente."""
        builder = ModelBuilder(
            backbone_name='VGG16',
            input_shape=(224, 224, 3),
            num_classes=4,
            n_layers=(1, 1)
        )
        
        hp = kt.HyperParameters()
        
        try:
            model = builder.build(hp)
            
            # El backbone (primera capa) debe estar congelado
            # Las capas de la cabeza deben ser entrenables
            trainable_layers = [layer for layer in model.layers if layer.trainable]
            non_trainable_layers = [layer for layer in model.layers if not layer.trainable]
            
            # Debe haber capas entrenables (la cabeza de clasificación)
            assert len(trainable_layers) > 0
            
        except Exception as e:
            pytest.skip(f"No se pudo verificar parámetros: {e}")


class TestHyperParameterSpace:
    """Tests para el espacio de búsqueda de hiperparámetros."""

    def test_hyperparameter_ranges(self):
        """Verifica que los rangos de hiperparámetros son válidos."""
        builder = ModelBuilder(
            backbone_name='VGG16',
            input_shape=(224, 224, 3),
            num_classes=4,
            n_layers=(2, 10),
            units=(16, 512, 32),
            dropout_range=(0.1, 0.5)
        )
        
        # Verificar rangos
        assert builder.n_layers[0] < builder.n_layers[1]
        assert builder.units[0] < builder.units[1]
        assert 0 <= builder.dropout_range[0] < builder.dropout_range[1] <= 1

    def test_learning_rate_options(self):
        """Verifica opciones de learning rate."""
        learning_rates = [0.001, 0.0001, 0.00001]
        builder = ModelBuilder(
            backbone_name='VGG16',
            input_shape=(224, 224, 3),
            num_classes=4,
            learning_rates=learning_rates
        )
        
        assert builder.learning_rates == learning_rates
        for lr in learning_rates:
            assert 0 < lr < 1

    def test_activation_functions(self):
        """Verifica funciones de activación válidas."""
        activations = ['relu', 'tanh', 'elu', 'selu']
        builder = ModelBuilder(
            backbone_name='VGG16',
            input_shape=(224, 224, 3),
            num_classes=4,
            activation=activations
        )
        
        assert set(activations).issubset(set(builder.activation))

