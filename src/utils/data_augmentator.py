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
    Clase para realizar operaciones de aumento de datos de manera reproducible.

    Combina transformaciones espaciales clásicas con operaciones de
    modificación de características.
    """
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.image_modifier = ImageAugmentor()

    def _apply_classic_transform(self, image: Image.Image) -> Image.Image:
        """Aplica una de las transformaciones espaciales clásicas."""
        transforms = [
            lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
            lambda img: img.transpose(Image.FLIP_TOP_BOTTOM),
            lambda img: img.rotate(self.rng.randint(-30, 30)),
            lambda img: self._zoom(img, self.rng.uniform(1.0, 1.2)),
        ]
        chosen_transform = self.rng.choice(transforms)
        return chosen_transform(image)

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

    def _apply_modifier_transform(self, image: Image.Image) -> Image.Image:
        """Aplica una transformación de características del ImageAugmentor con probabilidad."""
        modifiers = [
            self.image_modifier.downsample,
            self.image_modifier.distort,
            self.image_modifier.add_noise,
            self.image_modifier.adjust_contrast,
            self.image_modifier.adjust_brightness,
            self.image_modifier.adjust_color_intensity,
            self.image_modifier.adjust_sharpness,
        ]
        chosen_modifier = self.rng.choice(modifiers)
        
        # Las funciones de ImageAugmentor operan en arrays de NumPy
        img_np = np.array(image)
        
        # Usamos try-except para manejar parámetros por defecto
        try:
            transformed_img_np = chosen_modifier(img_np)
        except TypeError:
            # Algunas funciones requieren parámetros específicos
            if chosen_modifier == self.image_modifier.distort:
                transformed_img_np = self.image_modifier.distort(
                    img_np, axis=self.rng.choice(['horizontal', 'vertical'])
                )
            elif chosen_modifier == self.image_modifier.adjust_color_intensity:
                transformed_img_np = self.image_modifier.adjust_color_intensity(
                    img_np, channel=self.rng.randint(0, 2)
                )
            else:
                raise
        
        return Image.fromarray(transformed_img_np)

    def augment_dataset(self, 
                        images: List[Image.Image], 
                        labels: List, 
                        p: float = 0.5
                       ) -> Tuple[List[Image.Image], List]:
        """
        Aumenta un conjunto de datos aplicando transformaciones clásicas y de características.

        Args:
            images (List[Image.Image]): Lista de imágenes de entrada.
            labels (List): Lista de etiquetas correspondientes a las imágenes.
            p (float): Probabilidad de aplicar una transformación de características.

        Returns:
            Tuple[List[Image.Image], List]: Tupla con las imágenes y etiquetas aumentadas.
        """
        augmented_images = []
        augmented_labels = []

        for image, label in zip(images, labels):
            # 1. Aplicar una transformación espacial clásica
            spatial_transformed_image = self._apply_classic_transform(image)
            augmented_images.append(spatial_transformed_image)
            augmented_labels.append(label)

            # 2. Con probabilidad 'p', aplicar una transformación de características
            if self.rng.random() < p:
                feature_transformed_image = self._apply_modifier_transform(image)
                augmented_images.append(feature_transformed_image)
                augmented_labels.append(label)

        return augmented_images, augmented_labels