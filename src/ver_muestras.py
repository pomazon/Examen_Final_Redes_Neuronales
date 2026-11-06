"""Figura con imágenes al azar de cada dataset y clase, para revisar los datos a ojo.

Se usa como comprobación visual rápida de que la limpieza y las etiquetas del inventario
son coherentes antes de entrenar. No modifica los datos, solo ayuda a detectar errores
obvios a simple vista.

Uso:
    python src/ver_muestras.py --inventario results/inventario/inventario.csv --raiz data/raw
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


def main():
    """Genera una vista visual por dataset y clase para revisión rápida del inventario."""
    p = argparse.ArgumentParser()
    p.add_argument("--inventario", default="results/inventario/inventario.csv")
    p.add_argument("--raiz", default="data/raw")
    p.add_argument("--por_clase", type=int, default=6)
    p.add_argument("--semilla", type=int, default=0)
    p.add_argument("--salida", default="results/inventario/muestras.png")
    args = p.parse_args()

    df = pd.read_csv(args.inventario)
    df = df[df["legible"]].dropna(subset=["clase_original"])
    grupos = list(df.groupby(["dataset", "clase_original"]))

    fig, ejes = plt.subplots(len(grupos), args.por_clase,
                             figsize=(2 * args.por_clase, 2.2 * len(grupos)))
    for fila, ((ds, clase), sub) in zip(ejes, grupos):
        muestra = sub.sample(min(args.por_clase, len(sub)), random_state=args.semilla)
        for eje in fila:
            eje.axis("off")
        for eje, ruta in zip(fila, muestra["ruta"]):
            with Image.open(os.path.join(args.raiz, ruta)) as im:
                im.draft("RGB", (256, 256))
                eje.imshow(im.convert("RGB"))
        fila[0].set_title(f"{ds} / {clase} ({len(sub)})", fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(args.salida, dpi=80)
    print("Figura:", args.salida)


if __name__ == "__main__":
    main()
