"""
Tests para el módulo de preprocesamiento de datos.
"""
import pytest
import math
import numpy as np
from collections import defaultdict

from src.utils.utils import stratified_split_dataset, flatten_data


class TestStratifiedSplit:
    """Tests para la función stratified_split_dataset."""

    def test_split_ratios_sum_to_one(self):
        """Verifica que los ratios de división sumen 1.0."""
        split_ratios = (0.7, 0.15, 0.15)
        assert math.isclose(sum(split_ratios), 1.0), "Los ratios deben sumar 1.0"

    def test_stratified_split_basic(self, sample_dataset):
        """Verifica que la división estratificada funciona correctamente."""
        split_ratios = (0.7, 0.15, 0.15)
        result = stratified_split_dataset(sample_dataset, split_ratios)

        # Verificar que retorna las tres divisiones
        assert 'train' in result
        assert 'val' in result
        assert 'test' in result

        # Verificar que cada división tiene todas las categorías
        for split_name in ['train', 'val', 'test']:
            for category in sample_dataset.keys():
                assert category in result[split_name]

    def test_stratified_split_proportions(self, sample_dataset):
        """Verifica que las proporciones se respetan aproximadamente."""
        split_ratios = (0.7, 0.15, 0.15)
        result = stratified_split_dataset(sample_dataset, split_ratios)

        for category in sample_dataset.keys():
            total = len(sample_dataset[category])
            train_count = len(result['train'][category])
            val_count = len(result['val'][category])
            test_count = len(result['test'][category])

            # Verificar que la suma es correcta
            assert train_count + val_count + test_count == total

            # Verificar proporciones aproximadas (±1 imagen de tolerancia)
            expected_train = math.floor(total * split_ratios[0])
            expected_val = math.floor(total * split_ratios[1])

            assert abs(train_count - expected_train) <= 1
            assert abs(val_count - expected_val) <= 1

    def test_stratified_split_invalid_ratios(self, sample_dataset):
        """Verifica que se rechacen ratios inválidos."""
        invalid_ratios = (0.5, 0.3, 0.1)  # Suma 0.9, no 1.0

        with pytest.raises(ValueError, match="deben sumar 1.0"):
            stratified_split_dataset(sample_dataset, invalid_ratios)

    def test_stratified_split_empty_category(self):
        """Verifica el manejo de categorías vacías."""
        dataset = {
            'Blight': [None] * 10,
            'Empty': []  # Categoría vacía
        }

        split_ratios = (0.7, 0.15, 0.15)
        result = stratified_split_dataset(dataset, split_ratios)

        # La categoría vacía debe estar presente pero vacía
        assert 'Empty' in result['train']
        assert len(result['train']['Empty']) == 0


class TestFlattenData:
    """Tests para la función flatten_data."""

    def test_flatten_data_basic(self, sample_dataset):
        """Verifica que flatten_data convierte correctamente el dataset."""
        X, y = flatten_data(sample_dataset, image_size=(224, 224))

        # Verificar shapes
        assert X.shape[0] == 40  # 4 clases * 10 imágenes
        assert X.shape[1:] == (224, 224, 3)  # Dimensiones de imagen
        assert y.shape[0] == 40

        # Verificar tipos
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

    def test_flatten_data_resize(self, sample_dataset):
        """Verifica que el redimensionamiento funciona correctamente."""
        target_size = (128, 128)
        X, y = flatten_data(sample_dataset, image_size=target_size)

        assert X.shape[1:3] == target_size

    def test_flatten_data_labels(self, sample_dataset):
        """Verifica que las etiquetas se asignan correctamente."""
        X, y = flatten_data(sample_dataset, image_size=(224, 224))

        # Verificar que todas las clases están presentes
        unique_labels = set(y)
        expected_labels = set(sample_dataset.keys())
        assert unique_labels == expected_labels

        # Verificar que cada clase tiene el número correcto de muestras
        from collections import Counter
        label_counts = Counter(y)

        for class_name in sample_dataset.keys():
            assert label_counts[class_name] == len(sample_dataset[class_name])

    def test_flatten_data_empty_dataset(self):
        """Verifica el manejo de un dataset vacío."""
        empty_dataset = {'Blight': []}
        X, y = flatten_data(empty_dataset, image_size=(224, 224))

        assert X.shape[0] == 0
        assert y.shape[0] == 0

    def test_flatten_data_pixel_values(self, sample_dataset):
        """Verifica que los valores de píxeles estén en el rango correcto."""
        X, y = flatten_data(sample_dataset, image_size=(224, 224))

        # Valores de píxeles deben estar entre 0 y 255
        assert X.min() >= 0
        assert X.max() <= 255
        assert X.dtype == np.uint8 or X.dtype == np.int64
