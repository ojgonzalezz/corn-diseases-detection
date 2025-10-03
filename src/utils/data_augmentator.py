#####################################################################################
# ----------------------------------- Model Trainer ---------------------------------
#####################################################################################

#########################
# ---- Depdendencies ----
#########################

import random
from typing import List, Tuple
from collections import defaultdict

import numpy as np
import cv2
from PIL import Image

# Asume que el archivo image_modifier.py está en la misma carpeta o en una ruta accesible.
from src.utils.image_modifier import ImageAugmentor

##########################
# ---- Data Augmenter ----
##########################

class DataAugmenter:
    """
    🚀 PRIORIDAD CRÍTICA - Data Augmentation Agresiva
    Clase para realizar operaciones de aumento de datos agresivas (+20-40% mejora).

    Implementa configuración agresiva con técnicas avanzadas:
    - Spatial transforms: flip, rotation, zoom, shear
    - Color jitter: brightness, contrast, saturation, hue
    - Noise: gaussian noise, random erasing
    - Advanced: CutMix, MixUp
    """
    def __init__(self, seed: int = 42, config: dict = None):
        self.rng = random.Random(seed)
        self.image_modifier = ImageAugmentor()

        # 🚀 Configuración agresiva por defecto
        self.augmentation_config = config or {
            'random_flip': True,  # horizontal y vertical
            'random_rotation': 30,  # grados
            'random_zoom': (0.8, 1.2),
            'random_shear': 0.2,
            'color_jitter': {
                'brightness': 0.3,
                'contrast': 0.3,
                'saturation': 0.3,
                'hue': 0.1
            },
            'gaussian_noise': 0.05,
            'random_erasing': 0.2,  # probabilidad
            'cutmix': True,  # alpha=1.0
            'mixup': True,   # alpha=0.2
        }

    def _apply_aggressive_spatial_transform(self, image: Image.Image) -> Image.Image:
        """🚀 Aplica transformación espacial agresiva usando configuración."""
        img_np = np.array(image)

        # Aplicar transforms en secuencia (múltiples por imagen)
        transforms_applied = []

        # 1. Random flip (siempre aplicado)
        if self.augmentation_config.get('random_flip', True):
            img_np = self.image_modifier.random_flip(img_np)
            transforms_applied.append('flip')

        # 2. Random rotation
        if self.rng.random() > 0.3:  # 70% chance
            max_angle = self.augmentation_config.get('random_rotation', 30)
            img_np = self.image_modifier.random_rotation(img_np, max_angle)
            transforms_applied.append('rotation')

        # 3. Random zoom
        if self.rng.random() > 0.4:  # 60% chance
            zoom_range = self.augmentation_config.get('random_zoom', (0.8, 1.2))
            img_np = self.image_modifier.random_zoom(img_np, zoom_range)
            transforms_applied.append('zoom')

        # 4. Random shear
        if self.rng.random() > 0.5:  # 50% chance
            shear_factor = self.augmentation_config.get('random_shear', 0.2)
            img_np = self.image_modifier.random_shear(img_np, shear_factor)
            transforms_applied.append('shear')

        return Image.fromarray(img_np), transforms_applied

    def _zoom(self, image: Image.Image, factor: float) -> Image.Image:
        """Aplica un zoom aleatorio a la imagen."""
        width, height = image.size
        new_width, new_height = int(width / factor), int(height / factor)
        left = (width - new_width) // 2
        top = (height - new_height) // 2
        right = left + new_width
        bottom = top + new_height
        zoomed_img = image.crop((left, top, right, bottom))
        return zoomed_img.resize((width, height))

    def _apply_aggressive_color_transform(self, image: Image.Image) -> tuple:
        """🚀 Aplica transformación de color agresiva usando configuración."""
        img_np = np.array(image)
        transforms_applied = []

        # 1. Color jitter (brightness, contrast, saturation, hue)
        if self.rng.random() > 0.2:  # 80% chance
            color_config = self.augmentation_config.get('color_jitter', {})
            img_np = self.image_modifier.color_jitter(
                img_np,
                brightness=color_config.get('brightness', 0.3),
                contrast=color_config.get('contrast', 0.3),
                saturation=color_config.get('saturation', 0.3),
                hue=color_config.get('hue', 0.1)
            )
            transforms_applied.append('color_jitter')

        # 2. Gaussian noise
        if self.rng.random() > 0.7:  # 30% chance (menos frecuente)
            noise_std = self.augmentation_config.get('gaussian_noise', 0.05)
            img_np = self.image_modifier.gaussian_noise(img_np, noise_std)
            transforms_applied.append('gaussian_noise')

        # 3. Random erasing
        if self.rng.random() > 0.8:  # 20% chance (técnica agresiva)
            erasing_prob = self.augmentation_config.get('random_erasing', 0.2)
            img_np = self.image_modifier.random_erasing(img_np, erasing_prob)
            transforms_applied.append('random_erasing')

        return Image.fromarray(img_np), transforms_applied

    def augment_dataset_aggressive(self,
                                   images: List[Image.Image],
                                   labels: List,
                                   num_augments_per_image: int = 3
                                  ) -> Tuple[List[Image.Image], List]:
        """
        🚀 PRIORIDAD CRÍTICA - Data Augmentation Agresiva (+20-40% mejora)

        Aumenta el dataset aplicando múltiples técnicas agresivas por imagen.

        Args:
            images (List[Image.Image]): Lista de imágenes de entrada.
            labels (List): Lista de etiquetas correspondientes a las imágenes.
            num_augments_per_image (int): Número de augmentations por imagen original.

        Returns:
            Tuple[List[Image.Image], List]: Tupla con las imágenes y etiquetas aumentadas.
        """
        augmented_images = []
        augmented_labels = []

        # Mantener imágenes originales
        augmented_images.extend(images)
        augmented_labels.extend(labels)

        # Aplicar augmentation agresiva
        for image, label in zip(images, labels):
            for _ in range(num_augments_per_image):
                # 1. Aplicar transformación espacial agresiva
                spatial_augmented, spatial_transforms = self._apply_aggressive_spatial_transform(image)

                # 2. Aplicar transformación de color agresiva
                final_augmented, color_transforms = self._apply_aggressive_color_transform(spatial_augmented)

                augmented_images.append(final_augmented)
                augmented_labels.append(label)

        # 3. Aplicar CutMix y MixUp si hay suficientes imágenes
        if len(images) >= 2 and self.augmentation_config.get('cutmix', True):
            self._apply_cutmix_augmentation(images, labels, augmented_images, augmented_labels)

        if len(images) >= 2 and self.augmentation_config.get('mixup', True):
            self._apply_mixup_augmentation(images, labels, augmented_images, augmented_labels)

        return augmented_images, augmented_labels

    def _apply_cutmix_augmentation(self, original_images: List[Image.Image],
                                   original_labels: List,
                                   augmented_images: List[Image.Image],
                                   augmented_labels: List) -> None:
        """Aplica CutMix entre pares de imágenes aleatorias."""
        num_cutmix = min(len(original_images) // 2, 5)  # Máximo 5 cutmix

        for _ in range(num_cutmix):
            # Seleccionar dos imágenes aleatorias
            idx1, idx2 = self.rng.sample(range(len(original_images)), 2)
            img1, img2 = original_images[idx1], original_images[idx2]

            # Aplicar CutMix
            mixed_img, lam = self.image_modifier.cutmix(np.array(img1), np.array(img2), alpha=1.0)

            # Calcular etiqueta mixta (weighted average)
            label1, label2 = original_labels[idx1], original_labels[idx2]
            mixed_label = lam * np.array(label1) + (1 - lam) * np.array(label2)

            augmented_images.append(Image.fromarray(mixed_img))
            augmented_labels.append(mixed_label.tolist())

    def _apply_mixup_augmentation(self, original_images: List[Image.Image],
                                  original_labels: List,
                                  augmented_images: List[Image.Image],
                                  augmented_labels: List) -> None:
        """Aplica MixUp entre pares de imágenes aleatorias."""
        num_mixup = min(len(original_images) // 2, 5)  # Máximo 5 mixup

        for _ in range(num_mixup):
            # Seleccionar dos imágenes aleatorias
            idx1, idx2 = self.rng.sample(range(len(original_images)), 2)
            img1, img2 = original_images[idx1], original_images[idx2]

            # Aplicar MixUp
            mixed_img, lam = self.image_modifier.mixup(np.array(img1), np.array(img2), alpha=0.2)

            # Calcular etiqueta mixta
            label1, label2 = original_labels[idx1], original_labels[idx2]
            mixed_label = lam * np.array(label1) + (1 - lam) * np.array(label2)

            augmented_images.append(Image.fromarray(mixed_img))
            augmented_labels.append(mixed_label.tolist())

    # Método backward compatibility
    def augment_dataset(self,
                        images: List[Image.Image],
                        labels: List,
                        p: float = 0.5
                       ) -> Tuple[List[Image.Image], List]:
        """Método legacy - redirige al método agresivo."""
        return self.augment_dataset_aggressive(images, labels, num_augments_per_image=2)