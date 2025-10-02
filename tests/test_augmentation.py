"""
Tests para el módulo de augmentación de imágenes.
"""
import pytest
import numpy as np
from PIL import Image

from src.utils.image_modifier import ImageAugmentor
from src.utils.data_augmentator import DataAugmenter


class TestImageAugmentor:
    """Tests para la clase ImageAugmentor."""

    @pytest.fixture
    def augmentor(self):
        """Fixture que proporciona una instancia de ImageAugmentor."""
        return ImageAugmentor()

    def test_format_checker_numpy(self, augmentor, sample_image):
        """Verifica que format_checker acepta arrays de numpy."""
        img_array = np.array(sample_image)
        result = augmentor.format_checker(img_array)
        assert isinstance(result, np.ndarray)
        assert result.shape == img_array.shape

    def test_format_checker_pil(self, augmentor, sample_image):
        """Verifica que format_checker convierte PIL.Image a numpy."""
        result = augmentor.format_checker(sample_image)
        assert isinstance(result, np.ndarray)
        assert result.shape == (224, 224, 3)

    def test_format_checker_invalid(self, augmentor):
        """Verifica que format_checker rechaza tipos inválidos."""
        with pytest.raises(TypeError, match="Formato de imagen no soportado"):
            augmentor.format_checker("not_an_image")

    def test_downsample_shape(self, augmentor, sample_image):
        """Verifica que downsample mantiene las dimensiones originales."""
        original_shape = np.array(sample_image).shape
        downsampled = augmentor.downsample(sample_image, factor=2)

        assert downsampled.shape == original_shape

    def test_downsample_quality_degradation(self, augmentor, sample_image):
        """Verifica que downsample efectivamente degrada la calidad."""
        img_array = np.array(sample_image)
        downsampled = augmentor.downsample(img_array, factor=4)

        # La imagen downsampled debe ser diferente (no idéntica)
        assert not np.array_equal(img_array, downsampled)

    def test_distort_horizontal(self, augmentor, sample_image):
        """Verifica distorsión horizontal."""
        original_shape = np.array(sample_image).shape
        distorted = augmentor.distort(sample_image, axis='horizontal', factor=1.5)

        assert distorted.shape == original_shape

    def test_distort_vertical(self, augmentor, sample_image):
        """Verifica distorsión vertical."""
        original_shape = np.array(sample_image).shape
        distorted = augmentor.distort(sample_image, axis='vertical', factor=1.5)

        assert distorted.shape == original_shape

    def test_distort_invalid_axis(self, augmentor, sample_image):
        """Verifica que se rechaza un eje inválido."""
        with pytest.raises(ValueError, match="axis debe ser"):
            augmentor.distort(sample_image, axis='diagonal', factor=1.5)

    def test_add_noise(self, augmentor, sample_image):
        """Verifica que add_noise mantiene el shape y agrega variación."""
        img_array = np.array(sample_image)
        noisy = augmentor.add_noise(img_array, amount=20)

        # Shape debe mantenerse
        assert noisy.shape == img_array.shape

        # Valores deben estar en rango válido
        assert noisy.min() >= 0
        assert noisy.max() <= 255

        # Debe haber diferencias (ruido agregado)
        assert not np.array_equal(img_array, noisy)

    def test_chop(self, augmentor, sample_image):
        """Verifica que chop recorta y reescala correctamente."""
        original_shape = np.array(sample_image).shape
        chopped = augmentor.chop(sample_image, x1=50, y1=50, x2=150, y2=150)

        # Debe mantener shape original después de reescalar
        assert chopped.shape == original_shape

    def test_adjust_contrast(self, augmentor, sample_image):
        """Verifica ajuste de contraste."""
        img_array = np.array(sample_image)
        adjusted = augmentor.adjust_contrast(img_array, factor=1.5)

        assert adjusted.shape == img_array.shape
        assert adjusted.min() >= 0
        assert adjusted.max() <= 255

    def test_adjust_brightness(self, augmentor, sample_image):
        """Verifica ajuste de brillo."""
        img_array = np.array(sample_image)

        # Aumentar brillo
        brighter = augmentor.adjust_brightness(img_array, delta=50)
        assert brighter.shape == img_array.shape

        # Disminuir brillo
        darker = augmentor.adjust_brightness(img_array, delta=-50)
        assert darker.shape == img_array.shape

    def test_adjust_color_intensity(self, augmentor, sample_image):
        """Verifica ajuste de intensidad de color."""
        img_array = np.array(sample_image)

        for channel in [0, 1, 2]:  # B, G, R
            adjusted = augmentor.adjust_color_intensity(img_array, channel=channel, factor=1.5)
            assert adjusted.shape == img_array.shape
            assert adjusted.min() >= 0
            assert adjusted.max() <= 255

    def test_adjust_sharpness(self, augmentor, sample_image):
        """Verifica ajuste de nitidez."""
        img_array = np.array(sample_image)
        sharpened = augmentor.adjust_sharpness(img_array, amount=1.5)

        assert sharpened.shape == img_array.shape
        assert sharpened.min() >= 0
        assert sharpened.max() <= 255


class TestDataAugmenter:
    """Tests para la clase DataAugmenter."""

    @pytest.fixture
    def augmenter(self):
        """Fixture que proporciona una instancia de DataAugmenter."""
        return DataAugmenter(seed=42)

    def test_augment_dataset_basic(self, augmenter, sample_image_batch):
        """Verifica augmentación básica de un dataset."""
        labels = ['Blight'] * len(sample_image_batch)
        augmented_images, augmented_labels = augmenter.augment_dataset(
            sample_image_batch, labels, p=0.5
        )

        # Debe haber más imágenes después de la augmentación
        assert len(augmented_images) >= len(sample_image_batch)
        assert len(augmented_images) == len(augmented_labels)

    def test_augment_dataset_shape_preservation(self, augmenter, sample_image_batch):
        """Verifica que las imágenes aumentadas mantienen el shape."""
        labels = ['Healthy'] * len(sample_image_batch)
        original_size = sample_image_batch[0].size

        augmented_images, _ = augmenter.augment_dataset(
            sample_image_batch, labels, p=1.0
        )

        # Todas las imágenes aumentadas deben tener el mismo tamaño
        for img in augmented_images:
            assert img.size == original_size

    def test_augment_dataset_labels_consistency(self, augmenter, sample_image_batch):
        """Verifica que las etiquetas se mantienen consistentes."""
        labels = ['Blight', 'Rust', 'Spot', 'Healthy', 'Blight']
        augmented_images, augmented_labels = augmenter.augment_dataset(
            sample_image_batch, labels, p=0.5
        )

        # Cada etiqueta original debe aparecer al menos una vez
        for original_label in labels:
            assert original_label in augmented_labels

    def test_augment_dataset_probability_zero(self, augmenter, sample_image_batch):
        """Verifica comportamiento con p=0 (solo transformaciones clásicas)."""
        labels = ['Blight'] * len(sample_image_batch)
        augmented_images, augmented_labels = augmenter.augment_dataset(
            sample_image_batch, labels, p=0.0
        )

        # Con p=0, solo se aplican transformaciones clásicas (1 por imagen)
        assert len(augmented_images) == len(sample_image_batch)

    def test_augment_dataset_probability_one(self, augmenter, sample_image_batch):
        """Verifica comportamiento con p=1.0 (máxima augmentación)."""
        labels = ['Blight'] * len(sample_image_batch)
        augmented_images, augmented_labels = augmenter.augment_dataset(
            sample_image_batch, labels, p=1.0
        )

        # Con p=1.0, cada imagen genera 2 aumentaciones (clásica + modificador)
        assert len(augmented_images) == 2 * len(sample_image_batch)

    def test_augment_reproducibility(self):
        """Verifica que la augmentación es reproducible con la misma semilla."""
        # Crear imagen de prueba
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='RGB')
        images = [img]
        labels = ['Blight']

        # Primera augmentación
        augmenter1 = DataAugmenter(seed=42)
        aug1_images, aug1_labels = augmenter1.augment_dataset(images, labels, p=0.5)

        # Segunda augmentación con la misma semilla
        augmenter2 = DataAugmenter(seed=42)
        aug2_images, aug2_labels = augmenter2.augment_dataset(images, labels, p=0.5)

        # Resultados deben ser idénticos
        assert len(aug1_images) == len(aug2_images)
        assert aug1_labels == aug2_labels
