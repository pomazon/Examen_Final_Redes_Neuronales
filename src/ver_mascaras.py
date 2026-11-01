"""Figura para revisar a ojo la máscara de la hoja y el reemplazo de fondo en Saposoa.

Uso:
    python src/ver_mascaras.py --splits data/splits.csv --imagenes data/processed/imagenes
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from aumentos import FondoAleatorio, mascara_hoja


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--splits", default="data/splits.csv")
    p.add_argument("--imagenes", default="data/processed/imagenes")
    p.add_argument("--n", type=int, default=4, help="imágenes por clase")
    p.add_argument("--salida", default="results/aumentos/mascaras.png")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.salida), exist_ok=True)

    df = pd.read_csv(args.splits)
    df = df[(df["dataset"] == "saposoa") & (df["particion"] == "train")]
    muestra = df.groupby("clase").sample(args.n, random_state=0)
    fondo = FondoAleatorio(prob=1.0, semilla=0)

    fig, ejes = plt.subplots(len(muestra), 3, figsize=(7, 2.3 * len(muestra)))
    for fila, (_, r) in zip(ejes, muestra.iterrows()):
        im = Image.open(os.path.join(args.imagenes, r["ruta"])).convert("RGB")
        m = mascara_hoja(im)
        fila[0].imshow(im)
        fila[0].set_title(f"{r['clase']}  {os.path.basename(r['ruta'])[:14]}", fontsize=7)
        fila[1].imshow(np.array(im) * m[..., None])
        fila[1].set_title(f"máscara ({m.mean():.0%} de la imagen)", fontsize=7)
        fila[2].imshow(fondo(im))
        fila[2].set_title("fondo reemplazado", fontsize=7)
        for e in fila:
            e.axis("off")
    fig.tight_layout()
    fig.savefig(args.salida, dpi=80)
    print("Figura:", args.salida)


if __name__ == "__main__":
    main()
