"""Dibuja grupos de casi-duplicados para revisarlos a ojo y cuenta las clases de RoCoLe.

Uso:
    python src/revisar_grupos.py --raiz data/raw --inventario results/inventario --dataset saposoa
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


def miniatura(ruta, lado=200):
    with Image.open(ruta) as im:
        im = im.convert("RGB")
        im.thumbnail((lado, lado))
        return im.copy()


def dibujar(grupos, raiz, salida, titulo, max_por_grupo=6):
    filas = len(grupos)
    fig, ejes = plt.subplots(filas, max_por_grupo, figsize=(2 * max_por_grupo, 2.2 * filas))
    if filas == 1:
        ejes = [ejes]
    for fila, (g, sub) in zip(ejes, grupos):
        for eje in fila:
            eje.axis("off")
        for eje, (_, r) in zip(fila, sub.head(max_por_grupo).iterrows()):
            eje.imshow(miniatura(os.path.join(raiz, r["ruta"])))
            eje.set_title(f"{r['carpeta']}\n{os.path.basename(r['ruta'])[:18]}", fontsize=7)
        fila[0].text(-0.15, 0.5, f"grupo {g}\n({len(sub)} img)", transform=fila[0].transAxes,
                     ha="right", va="center", fontsize=8)
    fig.suptitle(titulo, fontsize=10)
    fig.tight_layout()
    fig.savefig(salida, dpi=80)
    plt.close(fig)
    print("Figura:", salida)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raiz", default="data/raw")
    p.add_argument("--inventario", default="results/inventario")
    p.add_argument("--dataset", default="saposoa")
    args = p.parse_args()

    g = pd.read_csv(os.path.join(args.inventario, "grupos_casi_duplicados.csv"))
    sub = g[g["dataset"] == args.dataset]
    tam = sub.groupby("grupo").size()
    n_clases = sub.groupby("grupo")["carpeta"].nunique()

    mezclados = [(k, sub[sub["grupo"] == k]) for k in n_clases[n_clases > 1].index[:5]]
    if mezclados:
        dibujar(mezclados, args.raiz,
                os.path.join(args.inventario, f"{args.dataset}_grupos_mezclados.png"),
                f"{args.dataset}: grupos que mezclan clases")

    mayores = [(k, sub[sub["grupo"] == k]) for k in tam.sort_values(ascending=False).index[:4]]
    dibujar(mayores, args.raiz,
            os.path.join(args.inventario, f"{args.dataset}_grupos_mayores.png"),
            f"{args.dataset}: grupos más grandes")

    print("\nTamaño de los grupos con más de una imagen:")
    print(tam[tam > 1].value_counts().sort_index().to_dict())

    xlsx = glob.glob(os.path.join(args.raiz, "rocole", "*", "Annotations", "*classes*.xlsx"))
    if xlsx:
        t = pd.read_excel(xlsx[0])
        print("\nRoCoLe, etiqueta binaria:")
        print(t["Binary.Label"].value_counts().to_string())
        print("\nRoCoLe, etiqueta multiclase:")
        print(t["Multiclass.Label"].value_counts().to_string())


if __name__ == "__main__":
    main()
