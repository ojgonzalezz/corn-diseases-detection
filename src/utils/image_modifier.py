#####################################################################################
# ---------------------------------- Image Modifier ---------------------------------
#####################################################################################

#########################
# ---- Depdendencies ----
#########################

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import random

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

    # 🚀 PRIORIDAD CRÍTICA - Nuevas técnicas agresivas de augmentation

    def random_flip(self, input_image: np.ndarray) -> np.ndarray:
        """Flip horizontal y vertical aleatorio."""
        image = self.format_checker(input_image)
        # Horizontal flip (50% chance)
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
        # Vertical flip (50% chance)
        if random.random() > 0.5:
            image = cv2.flip(image, 0)
        return image

    def random_rotation(self, input_image: np.ndarray, max_angle: int = 30) -> np.ndarray:
        """Rotación aleatoria entre -max_angle y +max_angle grados."""
        image = self.format_checker(input_image)
        pil_image = Image.fromarray(image)

        angle = random.uniform(-max_angle, max_angle)
        rotated = pil_image.rotate(angle, expand=False, fillcolor=(128, 128, 128))
        return np.array(rotated)

    def random_zoom(self, input_image: np.ndarray, zoom_range: tuple = (0.8, 1.2)) -> np.ndarray:
        """Zoom aleatorio dentro del rango especificado."""
        image = self.format_checker(input_image)
        height, width = image.shape[:2]

        zoom_factor = random.uniform(zoom_range[0], zoom_range[1])

        # Calcular nuevas dimensiones
        new_height = int(height * zoom_factor)
        new_width = int(width * zoom_factor)

        # Resize
        zoomed = cv2.resize(image, (new_width, new_height))

        # Crop o pad para mantener tamaño original
        if zoom_factor > 1.0:  # Zoom in - crop
            y_start = (new_height - height) // 2
            x_start = (new_width - width) // 2
            cropped = zoomed[y_start:y_start+height, x_start:x_start+width]
            return cropped
        else:  # Zoom out - pad
            result = np.full_like(image, 128)  # Gray padding
            y_start = (height - new_height) // 2
            x_start = (width - new_width) // 2
            result[y_start:y_start+new_height, x_start:x_start+new_width] = zoomed
            return result

    def random_shear(self, input_image: np.ndarray, shear_factor: float = 0.2) -> np.ndarray:
        """Shear aleatorio."""
        image = self.format_checker(input_image)
        height, width = image.shape[:2]

        shear_factor = random.uniform(-shear_factor, shear_factor)

        # Matriz de transformación
        M = np.array([[1, shear_factor, 0],
                      [0, 1, 0]], dtype=np.float32)

        sheared = cv2.warpAffine(image, M, (width, height),
                                borderMode=cv2.BORDER_REFLECT)
        return sheared

    def color_jitter(self, input_image: np.ndarray,
                     brightness: float = 0.3,
                     contrast: float = 0.3,
                     saturation: float = 0.3,
                     hue: float = 0.1) -> np.ndarray:
        """Color jitter agresivo: brightness, contrast, saturation, hue."""
        image = self.format_checker(input_image)
        pil_image = Image.fromarray(image)

        # Brightness
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(random.uniform(1-brightness, 1+brightness))

        # Contrast
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(random.uniform(1-contrast, 1+contrast))

        # Color (saturation)
        enhancer = ImageEnhance.Color(pil_image)
        pil_image = enhancer.enhance(random.uniform(1-saturation, 1+saturation))

        # Convert back to numpy
        result = np.array(pil_image)

        # Hue adjustment (manual implementation)
        if hue > 0:
            hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV).astype(np.float32)
            hue_shift = random.uniform(-hue, hue) * 180  # HSV hue range is 0-180
            hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
            result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        return result

    def gaussian_noise(self, input_image: np.ndarray, stddev: float = 0.05) -> np.ndarray:
        """Ruido gaussiano con stddev relativo al rango de la imagen."""
        image = self.format_checker(input_image).astype(np.float32)

        # Calcular stddev relativo al rango de la imagen (0-255)
        noise_std = stddev * 255.0
        noise = np.random.normal(0, noise_std, image.shape)

        noisy = image + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def random_erasing(self, input_image: np.ndarray, probability: float = 0.2,
                      max_area: float = 0.3) -> np.ndarray:
        """Random erasing: borra un rectángulo aleatorio de la imagen."""
        image = self.format_checker(input_image)

        if random.random() > probability:
            return image  # No aplicar erasing

        height, width = image.shape[:2]

        # Generar rectángulo aleatorio
        area = random.uniform(0.02, max_area)  # Entre 2% y max_area del área total
        aspect_ratio = random.uniform(0.3, 3.0)  # Relación de aspecto variable

        # Calcular dimensiones del rectángulo
        rect_area = area * height * width
        rect_height = int(np.sqrt(rect_area / aspect_ratio))
        rect_width = int(rect_area / rect_height)

        # Asegurar que no exceda los límites
        rect_height = min(rect_height, height)
        rect_width = min(rect_width, width)

        # Posición aleatoria
        y1 = random.randint(0, height - rect_height)
        x1 = random.randint(0, width - rect_width)
        y2 = y1 + rect_height
        x2 = x1 + rect_width

        # Borrar el rectángulo (llenar con valor medio)
        erased = image.copy()
        erased[y1:y2, x1:x2] = 128  # Valor medio

        return erased

    def cutmix(self, image1: np.ndarray, image2: np.ndarray, alpha: float = 1.0) -> tuple:
        """
        CutMix: combina dos imágenes cortando y pegando regiones.
        Retorna: (imagen_resultante, lambda) donde lambda es el peso de image1
        """
        image1 = self.format_checker(image1)
        image2 = self.format_checker(image2)

        height, width = image1.shape[:2]

        # Generar bounding box aleatoria
        lam = np.random.beta(alpha, alpha)
        cut_ratio = np.sqrt(1.0 - lam)

        center_h = np.random.uniform(0, height)
        center_w = np.random.uniform(0, width)

        bb_h = int(cut_ratio * height * 0.5)
        bb_w = int(cut_ratio * width * 0.5)

        y1 = max(0, int(center_h - bb_h))
        y2 = min(height, int(center_h + bb_h))
        x1 = max(0, int(center_w - bb_w))
        x2 = min(width, int(center_w + bb_w))

        # Combinar imágenes
        result = image1.copy()
        result[y1:y2, x1:x2] = image2[y1:y2, x1:x2]

        return result, lam

    def mixup(self, image1: np.ndarray, image2: np.ndarray, alpha: float = 0.2) -> tuple:
        """
        MixUp: interpola linealmente entre dos imágenes.
        Retorna: (imagen_resultante, lambda) donde lambda es el peso de image1
        """
        image1 = self.format_checker(image1).astype(np.float32)
        image2 = self.format_checker(image2).astype(np.float32)

        lam = np.random.beta(alpha, alpha)

        result = lam * image1 + (1 - lam) * image2
        return result.astype(np.uint8), lam
