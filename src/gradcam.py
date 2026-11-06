"""Grad-CAM: en qué parte de la imagen se fija cada modelo para decidir.

Hace dos cosas:
1. Medición sobre todas las imágenes de Saposoa test: qué fracción del mapa de Grad-CAM
   cae dentro de la hoja (usando la máscara de la hoja). Si el modelo mira el fondo,
   esa fracción es baja.
2. Figura con las mismas imágenes de Saposoa, RoCoLe y Uganda vistas por cada modelo.

La lógica de cálculo y visualización no se altera aquí: se agrega documentación para
hacer más claro qué representa el análisis y cómo se usa para interpretar los modelos.

Uso:
    python src/gradcam.py --modelos resnet50_basico resnet50_fondo vit_b_16_basico vit_b_16_fondo
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from aumentos import mascara_hoja, transformacion
from entrenar import crear_modelo
from metricas import CLASES


class GradCAM:
    """Grad-CAM sobre la última capa convolucional (ResNet50) o el último bloque (ViT).

    El objetivo es producir un mapa de atención sobre la imagen para saber si el modelo
    considera la hoja o el fondo al tomar la decisión.
    """

    def __init__(self, modelo, tipo):
        self.modelo, self.tipo = modelo.eval(), tipo
        capa = modelo.layer4[-1] if tipo == "resnet50" else modelo.encoder.layers[-1].ln_1
        capa.register_forward_hook(self._guardar)

    def _guardar(self, _modulo, _entrada, salida):
        self.activacion = salida
        salida.register_hook(lambda g: setattr(self, "gradiente", g))

    def _a_mapa(self, t):
        if self.tipo == "resnet50":
            return t  # (B, C, 7, 7)
        lado = int((t.shape[1] - 1) ** 0.5)  # ViT: se descarta el token de clase y se arma 14x14
        return t[:, 1:].reshape(t.shape[0], lado, lado, -1).permute(0, 3, 1, 2)

    def __call__(self, x):
        x = x.requires_grad_(True)
        logits = self.modelo(x)
        proba = torch.softmax(logits, 1)
        clase = logits.argmax(1)
        self.modelo.zero_grad()
        logits.gather(1, clase[:, None]).sum().backward()
        a, g = self._a_mapa(self.activacion), self._a_mapa(self.gradiente)
        cam = F.relu((g.mean(dim=(2, 3), keepdim=True) * a).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        cam = cam / (cam.flatten(1).max(1).values[:, None, None] + 1e-8)
        return cam.detach().cpu().numpy(), proba.detach().cpu().numpy()


def cargar(nombre, carpeta_modelos, semilla, dispositivo):
    # Se lee la configuración del experimento para reconstruir el tipo de modelo
    # y cargar exactamente los pesos del checkpoint correcto.
    cfg = yaml.safe_load(open(f"configs/{nombre}.yaml"))
    m = crear_modelo(cfg["modelo"], preentrenado=False)
    m.load_state_dict(torch.load(os.path.join(carpeta_modelos, f"{nombre}_s{semilla}.pt"),
                                 map_location=dispositivo))
    return GradCAM(m.to(dispositivo), cfg["modelo"])


def recorte(im, tam=224):
    """La misma vista que recibe el modelo al evaluar: lado menor 224 y recorte central."""
    escala = tam / min(im.size)
    im = im.resize((round(im.width * escala), round(im.height * escala)))
    izq, arr = (im.width - tam) // 2, (im.height - tam) // 2
    return im.crop((izq, arr, izq + tam, arr + tam))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modelos", nargs="+", required=True)
    p.add_argument("--semilla", type=int, default=0)
    p.add_argument("--splits", default="data/splits.csv")
    p.add_argument("--imagenes", default="data/processed/imagenes")
    p.add_argument("--carpeta_modelos", default="data/processed/modelos")
    p.add_argument("--salida", default="results/gradcam")
    args = p.parse_args()
    os.makedirs(args.salida, exist_ok=True)
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    t = transformacion("evaluacion")

    df = pd.read_csv(args.splits)
    test = df[(df["dataset"] == "saposoa") & (df["particion"] == "test")].reset_index(drop=True)
    vistas = [recorte(Image.open(os.path.join(args.imagenes, r)).convert("RGB")) for r in test["ruta"]]
    mascaras = [mascara_hoja(v) for v in vistas]

    muestra = pd.concat([
        test.groupby("clase").sample(1, random_state=1),
        df[df["dataset"] == "rocole"].groupby("clase").sample(2, random_state=1),
        df[df["dataset"] == "uganda"].groupby("clase").sample(2, random_state=1),
    ]).reset_index(drop=True)
    vistas_m = [recorte(Image.open(os.path.join(args.imagenes, r)).convert("RGB")) for r in muestra["ruta"]]

    filas, mapas = [], {}
    for nombre in args.modelos:
        gc = cargar(nombre, args.carpeta_modelos, args.semilla, dispositivo)
        dentro = []
        for i in range(0, len(vistas), 32):
            x = torch.stack([t(v) for v in vistas[i:i + 32]]).to(dispositivo)
            cam, _ = gc(x)
            for c, m in zip(cam, mascaras[i:i + 32]):
                if m is not None:
                    dentro.append((c * m).sum() / (c.sum() + 1e-8))
        area = np.mean([m.mean() for m in mascaras if m is not None])
        filas.append({"modelo": nombre, "imagenes": len(dentro),
                      "fraccion_cam_en_hoja": np.mean(dentro), "desv": np.std(dentro),
                      "area_hoja": area, "concentracion": np.mean(dentro) / area})
        x = torch.stack([t(v) for v in vistas_m]).to(dispositivo)
        mapas[nombre] = gc(x)
        del gc
        torch.cuda.empty_cache()

    tabla = pd.DataFrame(filas)
    tabla.to_csv(os.path.join(args.salida, "cam_en_hoja.csv"), index=False)
    print("Saposoa test: fracción del mapa de Grad-CAM que cae dentro de la hoja")
    print("(concentración = fracción / área de la hoja; 1 = el modelo mira la hoja tanto como al azar)")
    print(tabla.round(3).to_string(index=False))

    # La figura final ordena por fila: imagen real y luego la misma imagen con el mapa de atención
    # de cada modelo, para comparar de forma directa qué región activó la decisión.
    fig, ejes = plt.subplots(len(muestra), len(args.modelos) + 1,
                             figsize=(2.1 * (len(args.modelos) + 1), 2.3 * len(muestra)))
    for i, r in muestra.iterrows():
        ejes[i, 0].imshow(vistas_m[i])
        ejes[i, 0].set_title(f"{r['dataset']}\nreal: {r['clase']}", fontsize=7)
        for j, nombre in enumerate(args.modelos, 1):
            cam, proba = mapas[nombre]
            ejes[i, j].imshow(vistas_m[i])
            ejes[i, j].imshow(cam[i], cmap="jet", alpha=0.45)
            k = proba[i].argmax()
            ejes[i, j].set_title(f"{nombre}\n{CLASES[k]} ({proba[i, k]:.0%})", fontsize=6)
        for e in ejes[i]:
            e.axis("off")
    fig.tight_layout()
    salida = os.path.join(args.salida, "gradcam.png")
    fig.savefig(salida, dpi=90)
    print("\nFigura:", salida)


if __name__ == "__main__":
    main()
