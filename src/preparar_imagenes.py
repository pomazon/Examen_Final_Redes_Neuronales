"""Reduce las imágenes de data/splits.csv a un tamaño de trabajo y las guarda aparte.

Las fotos originales pesan ~5,5 GB y miden hasta 6 112 px. Los modelos usan 224 px.
Se reduce cada imagen para que su lado menor mida 320 px (las de Uganda, de 256 px,
no se agrandan) y se guarda como JPEG de calidad 95 con la misma ruta relativa.

Este paso no cambian las etiquetas ni la partición, solo acelera la carga y reduce el
espacio de trabajo sin cambiar la información semántica de las fotos.

Uso:
    python src/preparar_imagenes.py --splits data/splits.csv --raiz data/raw \
        --destino data/processed/imagenes
"""
import argparse
import os

import pandas as pd
from PIL import Image


def reducir(origen, destino, lado):
    """Reduce una imagen a un lado mínimo dado y la guarda como JPEG optimizado."""
    with Image.open(origen) as im:
        im.draft("RGB", (lado, lado))  # decodifica el JPEG ya reducido: mucho más rápido
        im = im.convert("RGB")
        escala = lado / min(im.size)
        if escala < 1:
            im = im.resize((round(im.width * escala), round(im.height * escala)), Image.LANCZOS)
        os.makedirs(os.path.dirname(destino), exist_ok=True)
        im.save(destino, quality=95)
        return im.size


def main():
    """Prepara una versión reducida y reutilizable de todas las imágenes del proyecto."""
    p = argparse.ArgumentParser()
    p.add_argument("--splits", default="data/splits.csv")
    p.add_argument("--raiz", default="data/raw")
    p.add_argument("--destino", default="data/processed/imagenes")
    p.add_argument("--lado", type=int, default=320, help="lado menor final en píxeles")
    args = p.parse_args()

    df = pd.read_csv(args.splits)
    tamanos = []
    for k, ruta in enumerate(df["ruta"], 1):
        tamanos.append(reducir(os.path.join(args.raiz, ruta), os.path.join(args.destino, ruta), args.lado))
        if k % 200 == 0 or k == len(df):
            print(f"\r  {k}/{len(df)}", end="")
    print()

    df["ancho"], df["alto"] = zip(*tamanos)
    print("\nTamaño final en píxeles (mín / mediana / máx):")
    print(df.groupby("dataset")[["ancho", "alto"]].agg(["min", "median", "max"]).to_string())
    total = sum(os.path.getsize(os.path.join(d, f))
                for d, _, fs in os.walk(args.destino) for f in fs)
    print(f"\nImágenes: {len(df)}   Tamaño total: {total / 1e6:.0f} MB   Carpeta: {args.destino}")


if __name__ == "__main__":
    main()
