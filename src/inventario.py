"""Inventario de las imágenes: conteo por clase, tamaños, archivos dañados y duplicados.

Uso:
    python src/inventario.py --raiz data/raw --salida results/inventario
"""
import argparse
import hashlib
import os

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

EXT_IMG = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def titulo(texto):
    print("\n" + "=" * 70 + "\n" + texto + "\n" + "=" * 70)


def arbol(ruta, nivel_max=3):
    base = ruta.rstrip("/").count("/")
    for actual, _, archivos in os.walk(ruta):
        nivel = actual.count("/") - base
        if nivel > nivel_max:
            continue
        n_img = sum(os.path.splitext(a)[1].lower() in EXT_IMG for a in archivos)
        print(f"{'    ' * nivel}{os.path.basename(actual)}/  "
              f"({n_img} imágenes, {len(archivos)} archivos)")


def leer_imagenes(raiz, datasets):
    filas, otros = [], []
    for ds in datasets:
        for actual, _, archivos in os.walk(os.path.join(raiz, ds)):
            for a in archivos:
                ruta = os.path.join(actual, a)
                ext = os.path.splitext(a)[1].lower()
                if ext not in EXT_IMG:
                    otros.append({"dataset": ds, "archivo": os.path.relpath(ruta, raiz)})
                    continue
                fila = {"dataset": ds, "ruta": os.path.relpath(ruta, raiz),
                        "carpeta": os.path.basename(actual), "ext": ext,
                        "bytes": os.path.getsize(ruta)}
                try:
                    with open(ruta, "rb") as f:
                        fila["md5"] = hashlib.md5(f.read()).hexdigest()
                    with Image.open(ruta) as im:
                        fila["ancho"], fila["alto"], fila["modo"] = im.width, im.height, im.mode
                        fila["phash"] = str(imagehash.phash(im.convert("RGB")))
                    fila["corrupta"] = False
                except Exception as e:
                    fila["corrupta"] = True
                    fila["error"] = str(e)[:80]
                filas.append(fila)
    return pd.DataFrame(filas), pd.DataFrame(otros)


def contar_casi_duplicados(phashes, umbral):
    # Distancia de Hamming entre pHash. Hasta 4 bits = prácticamente la misma imagen.
    h = np.array([int(x, 16) for x in phashes], dtype=np.uint64)
    pares = 0
    for i in range(len(h) - 1):
        x = np.bitwise_xor(h[i], h[i + 1:])
        dist = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)
        pares += int((dist <= umbral).sum())
    return pares


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raiz", default="data/raw")
    p.add_argument("--salida", default="results/inventario")
    p.add_argument("--datasets", nargs="*", default=["saposoa", "rocole", "uganda"])
    p.add_argument("--umbral", type=int, default=4)
    args = p.parse_args()
    os.makedirs(args.salida, exist_ok=True)

    for ds in args.datasets:
        titulo(f"ÁRBOL: {ds}")
        arbol(os.path.join(args.raiz, ds))

    df, otros = leer_imagenes(args.raiz, args.datasets)
    df.to_csv(os.path.join(args.salida, "inventario_imagenes.csv"), index=False)
    ok = df[~df["corrupta"]]

    titulo("IMÁGENES POR DATASET Y CARPETA")
    print(df.groupby(["dataset", "carpeta"]).size().to_string())

    titulo("TAMAÑO EN PÍXELES (mín / mediana / máx)")
    print(ok.groupby("dataset")[["ancho", "alto"]]
          .agg(["min", "median", "max"]).to_string())

    titulo("FORMATO Y MODO DE COLOR")
    print(ok.groupby(["dataset", "ext", "modo"]).size().to_string())

    titulo("IMÁGENES DAÑADAS")
    print(df.groupby("dataset")["corrupta"].sum().to_string())

    titulo("DUPLICADOS EXACTOS (mismo md5)")
    dup = ok[ok.duplicated("md5", keep=False)]
    print("Archivos repetidos:", len(dup))
    print("Grupos entre datasets distintos:",
          int((dup.groupby("md5")["dataset"].nunique() > 1).sum()))
    print("Grupos con carpetas distintas:",
          int((dup.groupby("md5")["carpeta"].nunique() > 1).sum()))

    titulo(f"CASI-DUPLICADOS (pHash, distancia <= {args.umbral})")
    for ds in args.datasets:
        sub = ok[ok["dataset"] == ds]
        print(f"{ds}: {len(sub)} imágenes, "
              f"{contar_casi_duplicados(sub['phash'], args.umbral)} pares")

    titulo("ARCHIVOS QUE NO SON IMÁGENES")
    print(otros.to_string() if len(otros) else "Ninguno")

    print("\nDetalle por imagen:", os.path.join(args.salida, "inventario_imagenes.csv"))


if __name__ == "__main__":
    main()
