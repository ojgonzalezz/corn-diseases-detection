#####################################################################################
# ---------------------------------- Image Modifier ---------------------------------
#####################################################################################

#########################
# ---- Depdendencies ----
#########################

import cv2
import numpy as np
from PIL import Image

###########################
# ---- Image Augmentor ----
###########################

class ImageAugmentor:
    def __init__(self):
        pass

    def format_checker(self, image):
        """
        Verifica y convierte la imagen a np.ndarray si es necesario.
        Soporta: numpy.ndarray y PIL.Image.Image
        """
        if isinstance(image, np.ndarray):
            return image
        elif isinstance(image, Image.Image):
            return np.array(image)
        else:
            raise TypeError(f"Formato de imagen no soportado: {type(image)}")

    def downsample(self, input_image: np.ndarray, factor=2):
        """Reduce resolución y vuelve a escalar a tamaño original (degrada calidad)."""
        image = self.format_checker(input_image)
        height, width = image.shape[:2]
        small = cv2.resize(image, (width // factor, height // factor))
        return cv2.resize(small, (width, height))

    def distort(self, input_image: np.ndarray, axis='horizontal', factor=1.5):
        """Distorsiona horizontal o verticalmente."""
        image = self.format_checker(input_image)
        height, width = image.shape[:2]
        if axis == 'horizontal':
            new_w = int(width * factor)
            distorted = cv2.resize(image, (new_w, height))
        elif axis == 'vertical':
            new_h = int(height * factor)
            distorted = cv2.resize(image, (width, new_h))
        else:
            raise ValueError("axis debe ser 'horizontal' o 'vertical'")
        return cv2.resize(distorted, (width, height))

    def add_noise(self, input_image: np.ndarray, amount=20):
        """Agrega ruido gaussiano. amount = 0-100 (intensidad)."""
        image = self.format_checker(input_image)
        stddev = amount / 100 * 50
        noise = np.random.normal(0, stddev, image.shape).astype(np.float32)
        noisy = image.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def chop(self, input_image: np.ndarray, x1=0, y1=0, x2=None, y2=None):
        """Recorta la imagen en un área y la reescala al tamaño original."""
        image = self.format_checker(input_image)
        height, width = image.shape[:2]
        if x2 is None: x2 = width
        if y2 is None: y2 = height
        cropped = image[y1:y2, x1:x2]
        return cv2.resize(cropped, (width, height))

    def adjust_contrast(self, input_image: np.ndarray, factor=1.5):
        """Cambia el contraste (factor >1 aumenta contraste)."""
        image = self.format_checker(input_image)
        f = image.astype(np.float32)
        mean = np.mean(f, axis=(0, 1), keepdims=True)
        adjusted = (f - mean) * factor + mean
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    def adjust_brightness(self, input_image: np.ndarray, delta=50):
        """Cambia brillo sumando un delta."""
        image = self.format_checker(input_image)
        bright = image.astype(np.int16) + delta
        return np.clip(bright, 0, 255).astype(np.uint8)

    def adjust_color_intensity(self, input_image: np.ndarray, channel=0, factor=1.5):
        """
        Cambia intensidad de un canal RGB.
        channel = 0 (B), 1 (G), 2 (R)
        """
        image = self.format_checker(input_image)
        img = image.copy().astype(np.float32)
        img[..., channel] *= factor
        return np.clip(img, 0, 255).astype(np.uint8)

    def adjust_sharpness(self, input_image: np.ndarray, amount=1.0):
        """Cambia nitidez usando un filtro de realce."""
        image = self.format_checker(input_image)
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]], dtype=np.float32) * amount
        sharp = cv2.filter2D(image, -1, kernel)
        return np.clip(sharp, 0, 255).astype(np.uint8)
