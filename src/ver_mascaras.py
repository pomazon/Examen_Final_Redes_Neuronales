"""Revisa la máscara de la hoja en todas las imágenes de entrenamiento de Saposoa.

Imprime cuántas máscaras son confiables y qué parte de la imagen ocupan, y guarda una
figura con ejemplos al azar y con los casos extremos (máscaras más chicas, más grandes y
las descartadas), que son los que conviene mirar.

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
    p.add_argument("--salida", default="results/aumentos")
    args = p.parse_args()
    os.makedirs(args.salida, exist_ok=True)

    df = pd.read_csv(args.splits)
    df = df[(df["dataset"] == "saposoa") & (df["particion"] == "train")].reset_index(drop=True)
    areas = []
    for k, r in enumerate(df["ruta"], 1):
        m = mascara_hoja(Image.open(os.path.join(args.imagenes, r)).convert("RGB"))
        # La media de una máscara booleana es la fracción de píxeles que ocupa la hoja; NaN indica que la máscara falló.
        areas.append(np.nan if m is None else m.mean())
        if k % 200 == 0:
            print(f"\r  {k}/{len(df)}", end="")
    print()
    df["area"] = areas
    df.to_csv(os.path.join(args.salida, "mascaras.csv"), index=False)

    print("Máscaras confiables por clase (las demás no reemplazan el fondo):")
    print(df.groupby("clase")["area"].apply(lambda a: f"{a.notna().sum()}/{len(a)}").to_string())
    print("\nFracción de la imagen ocupada por la hoja (percentiles 5 / 50 / 95):")
    print(df.groupby("clase")["area"].quantile([0.05, 0.5, 0.95]).unstack().round(3).to_string())

    ok = df.dropna(subset=["area"])
    casos = [("al azar", ok.groupby("clase").sample(2, random_state=0)),
             ("más chica", ok.nsmallest(3, "area")),
             ("más grande", ok.nlargest(3, "area")),
             ("descartada", df[df["area"].isna()].head(3))]
    filas = [(t, r) for t, sub in casos for _, r in sub.iterrows()]
    # Activamos el reemplazo en todos los ejemplos con máscara para inspeccionar su contorno.
    fondo = FondoAleatorio(prob=1.0, semilla=0)

    fig, ejes = plt.subplots(len(filas), 3, figsize=(7, 2.3 * len(filas)))
    for fila, (tipo, r) in zip(ejes, filas):
        im = Image.open(os.path.join(args.imagenes, r["ruta"])).convert("RGB")
        m = mascara_hoja(im)
        fila[0].imshow(im)
        fila[0].set_title(f"[{tipo}] {r['clase']}  {os.path.basename(r['ruta'])[:14]}", fontsize=7)
        if m is not None:
            fila[1].imshow(np.array(im) * m[..., None])
            fila[1].set_title(f"máscara ({m.mean():.0%})", fontsize=7)
            fila[2].imshow(fondo(im, m))
            fila[2].set_title("fondo reemplazado", fontsize=7)
        else:
            fila[1].set_title("sin máscara confiable: no se reemplaza", fontsize=7)
        for e in fila:
            e.axis("off")
    fig.tight_layout()
    salida = os.path.join(args.salida, "mascaras.png")
    fig.savefig(salida, dpi=80)
    print("\nFigura:", salida)


if __name__ == "__main__":
    main()
