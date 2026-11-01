"""Transformaciones de imagen para entrenar y evaluar.

basico  recorte aleatorio suave y espejo horizontal
fuerte  recortes más agresivos, giros, cambios de color y desenfoque
fondo   lo mismo que fuerte y, además, reemplaza el fondo blanco de Saposoa por
        uno aleatorio (color liso o manchas de colores), para que el modelo no pueda
        usar el fondo para reconocer la clase
"""
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import color
from torchvision import transforms

MEDIA, DESV = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
UMBRAL_COLOR = 10  # distancia de color (a*, b* en Lab) a partir de la cual un píxel no es fondo
CROMA_MIN = 12     # intensidad de color mínima de la hoja: las sombras son casi grises
AREA_VALIDA = (0.03, 0.60)  # fracción de la imagen que puede ocupar una hoja


def mascara_hoja(im):
    """Máscara de la hoja, o None si no se pudo separar de forma confiable.

    El color del fondo se estima con la mediana del borde de la imagen. Un píxel es hoja
    si su color (sin contar el brillo) se aleja de ese fondo y además tiene color
    (croma) suficiente. Las sombras son casi grises: quedan como fondo.
    """
    lab = color.rgb2lab(np.asarray(im, float) / 255)
    ab = lab[..., 1:]
    borde = np.concatenate([ab[:8].reshape(-1, 2), ab[-8:].reshape(-1, 2),
                            ab[:, :8].reshape(-1, 2), ab[:, -8:].reshape(-1, 2)])
    distancia = np.linalg.norm(ab - np.median(borde, axis=0), axis=-1)
    croma = np.linalg.norm(ab, axis=-1)
    m = ndimage.binary_opening((distancia > UMBRAL_COLOR) & (croma > CROMA_MIN), iterations=2)
    etiquetas, n = ndimage.label(m)
    if n == 0:
        return None
    tam = ndimage.sum(m, etiquetas, range(1, n + 1))
    m = etiquetas == (np.argmax(tam) + 1)
    m = ndimage.binary_fill_holes(ndimage.binary_closing(m, iterations=5))
    return m if AREA_VALIDA[0] <= m.mean() <= AREA_VALIDA[1] else None


class FondoAleatorio:
    def __init__(self, prob=0.8, semilla=None):
        self.prob = prob
        self.rng = np.random.default_rng(semilla)

    def fondo(self, alto, ancho):
        if self.rng.random() < 0.5:  # color liso
            return np.ones((alto, ancho, 3)) * self.rng.uniform(0, 255, 3)
        chico = self.rng.uniform(0, 255, (self.rng.integers(3, 10), self.rng.integers(3, 10), 3))
        return np.array(Image.fromarray(chico.astype(np.uint8)).resize((ancho, alto), Image.BILINEAR), float)

    def __call__(self, im, mascara):
        """Si no hay máscara confiable, la imagen queda igual."""
        if mascara is None or self.rng.random() > self.prob:
            return im
        a = np.array(im, float)
        alfa = ndimage.gaussian_filter(mascara.astype(float), 1.5)[..., None]
        mezcla = alfa * a + (1 - alfa) * self.fondo(*a.shape[:2])
        return Image.fromarray(mezcla.clip(0, 255).astype(np.uint8))


def transformacion(tipo, tam=224):
    final = [transforms.ToTensor(), transforms.Normalize(MEDIA, DESV)]
    if tipo == "evaluacion":
        return transforms.Compose([transforms.Resize(tam), transforms.CenterCrop(tam)] + final)
    if tipo == "basico":
        return transforms.Compose([transforms.RandomResizedCrop(tam, scale=(0.7, 1.0)),
                                   transforms.RandomHorizontalFlip()] + final)
    if tipo in ("fuerte", "fondo"):
        return transforms.Compose([
            transforms.RandomResizedCrop(tam, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.05),
            transforms.RandomApply([transforms.GaussianBlur(5)], p=0.3),
        ] + final)
    raise ValueError(f"Aumento desconocido: {tipo}")
