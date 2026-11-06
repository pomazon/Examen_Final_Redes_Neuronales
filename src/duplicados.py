"""Busca fotos casi iguales (misma hoja con otra luz, encuadre o giro) dentro de cada dataset.

Cada imagen se resume en un vector de 2048 valores con una ResNet50 preentrenada en
ImageNet (sin su última capa) y se comparan los vectores por similitud coseno.
Guarda los pares parecidos y una figura con ejemplos por tramo de similitud, para
elegir a ojo el umbral a partir del cual dos fotos se consideran la misma.

La lógica de comparación no cambia; aquí solo se dejan notas para aclarar el objetivo
analítico del script.

Uso:
    python src/duplicados.py --inventario results/inventario/inventario.csv --raiz data/raw
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

TRAMOS = [(0.98, 1.01), (0.96, 0.98), (0.94, 0.96), (0.92, 0.94), (0.90, 0.92)]
TRANSFORMAR = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class Imagenes(Dataset):
    """Dataset simple para extraer embeddings de cada imagen usando una ResNet50."""

    def __init__(self, rutas):
        self.rutas = rutas

    def __len__(self):
        return len(self.rutas)

    def __getitem__(self, i):
        with Image.open(self.rutas[i]) as im:
            im.draft("RGB", (512, 512))  # decodifica el JPEG ya reducido: mucho más rápido
            return TRANSFORMAR(im.convert("RGB"))


@torch.no_grad()
def vectores(rutas, modelo, dispositivo):
    """Genera embeddings normalizados para comparar similitud coseno entre imágenes."""
    datos = DataLoader(Imagenes(rutas), batch_size=64, num_workers=2)
    salida = []
    for x in datos:
        v = modelo(x.to(dispositivo))
        salida.append(torch.nn.functional.normalize(v, dim=1).cpu())
        print(f"\r  {sum(len(s) for s in salida)}/{len(rutas)}", end="")
    print()
    return torch.cat(salida).numpy()


def miniatura(ruta):
    with Image.open(ruta) as im:
        im.draft("RGB", (256, 256))
        im = im.convert("RGB")
        im.thumbnail((200, 200))
        return im


def figura(pares, raiz, salida, titulo):
    """Dibuja un panel con pares parecidos y su valor de similitud."""
    fig, ejes = plt.subplots(len(pares), 2, figsize=(5, 2.3 * len(pares)))
    ejes = np.atleast_2d(ejes)
    for fila, (_, p) in zip(ejes, pares.iterrows()):
        for eje, lado in zip(fila, "ab"):
            eje.imshow(miniatura(os.path.join(raiz, p[f"ruta_{lado}"])))
            eje.set_title(f"{p[f'clase_{lado}']}  {os.path.basename(p[f'ruta_{lado}'])[:16]}", fontsize=7)
            eje.axis("off")
        fila[0].text(-0.08, 0.5, f"{p['similitud']:.3f}", transform=fila[0].transAxes,
                     ha="right", va="center", fontsize=9)
    fig.suptitle(titulo, fontsize=9)
    fig.tight_layout()
    fig.savefig(salida, dpi=80)
    plt.close(fig)
    print("Figura:", salida)


def main():
    """Busca duplicados y fotos casi iguales dentro de cada dataset."""
    p = argparse.ArgumentParser()
    p.add_argument("--inventario", default="results/inventario/inventario.csv")
    p.add_argument("--raiz", default="data/raw")
    p.add_argument("--salida", default="results/duplicados")
    p.add_argument("--vectores", default="data/processed")
    p.add_argument("--minimo", type=float, default=0.90)
    p.add_argument("--semilla", type=int, default=42)
    args = p.parse_args()
    os.makedirs(args.salida, exist_ok=True)
    os.makedirs(args.vectores, exist_ok=True)
    rng = np.random.default_rng(args.semilla)

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    print("Dispositivo:", dispositivo)
    modelo = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    # Quitamos la capa clasificadora para comparar las imágenes por sus características visuales.
    modelo.fc = torch.nn.Identity()
    modelo.eval().to(dispositivo)

    inv = pd.read_csv(args.inventario)
    inv = inv[inv["legible"]].drop_duplicates("md5")  # las copias exactas ya se conocen

    todos = []
    for ds, sub in inv.groupby("dataset"):
        sub = sub.reset_index(drop=True)
        print(f"\n{ds}: {len(sub)} imágenes distintas")
        v = vectores([os.path.join(args.raiz, r) for r in sub["ruta"]], modelo, dispositivo)
        np.savez(os.path.join(args.vectores, f"vectores_{ds}.npz"), v=v, rutas=sub["ruta"].values)

        # Con vectores normalizados, el producto escalar mide similitud coseno.
        s = v @ v.T
        # Conservamos una sola orientación de cada par y excluimos la comparación de una foto consigo misma.
        i, j = np.triu_indices(len(v), k=1)
        m = s[i, j] >= args.minimo
        pares = pd.DataFrame({"dataset": ds,
                              "ruta_a": sub["ruta"].values[i[m]], "ruta_b": sub["ruta"].values[j[m]],
                              "clase_a": sub["clase_original"].values[i[m]],
                              "clase_b": sub["clase_original"].values[j[m]],
                              "similitud": s[i, j][m]})
        todos.append(pares)

        print(f"  pares por tramo de similitud (total | misma clase | distinta clase)")
        muestras = []
        for lo, hi in TRAMOS:
            t = pares[(pares["similitud"] >= lo) & (pares["similitud"] < hi)]
            misma = int((t["clase_a"] == t["clase_b"]).sum())
            print(f"    {lo:.2f} a {min(hi, 1.0):.2f}: {len(t):6d} | {misma:6d} | {len(t) - misma:6d}")
            if len(t):
                muestras.append(t.iloc[rng.choice(len(t), size=min(3, len(t)), replace=False)])
        if muestras:
            figura(pd.concat(muestras), args.raiz, os.path.join(args.salida, f"{ds}_pares.png"),
                   f"{ds}: 3 pares al azar por tramo de similitud")
        if len(pares):
            figura(pares.nlargest(10, "similitud"), args.raiz,
                   os.path.join(args.salida, f"{ds}_mas_parecidos.png"),
                   f"{ds}: los 10 pares más parecidos")

    archivo = os.path.join(args.salida, "pares.csv")
    pd.concat(todos).to_csv(archivo, index=False)
    print("\nPares guardados en:", archivo)


if __name__ == "__main__":
    main()
