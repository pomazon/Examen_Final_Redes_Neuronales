"""Inventario de las imágenes descargadas.

Revisa cada imagen y guarda una tabla con su dataset, clase, tamaño, huella md5 y si se
puede abrir. Imprime un resumen: imágenes por clase, tamaños, archivos que no se pueden
leer, copias exactas y archivos que no son imágenes.

Uso:
    python src/inventario.py --raiz data/raw --salida results/inventario
"""
import argparse
import glob
import hashlib
import os

import pandas as pd
from PIL import Image

EXT_IMG = {".jpg", ".jpeg", ".png"}


def titulo(texto):
    print("\n" + "=" * 70 + "\n" + texto + "\n" + "=" * 70)


def etiquetas_rocole(raiz):
    """RoCoLe guarda las fotos juntas y la clase de cada una en un Excel aparte."""
    xlsx = glob.glob(os.path.join(raiz, "rocole", "*", "Annotations", "*classes*.xlsx"))
    if not xlsx:
        return {}
    t = pd.read_excel(xlsx[0])
    return dict(zip(t["File"], t["Multiclass.Label"]))


def revisar(ruta):
    """Devuelve md5, ancho, alto y error (None si la imagen se abre completa)."""
    with open(ruta, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    try:
        with Image.open(ruta) as im:
            ancho, alto = im.size
            im.load()  # decodifica toda la imagen: detecta archivos cortados
        return md5, ancho, alto, None
    except Exception as e:
        return md5, None, None, type(e).__name__ + ": " + str(e).split("'")[0][:60]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raiz", default="data/raw")
    p.add_argument("--salida", default="results/inventario")
    p.add_argument("--datasets", nargs="*", default=["saposoa", "rocole", "uganda"])
    args = p.parse_args()
    os.makedirs(args.salida, exist_ok=True)

    rocole = etiquetas_rocole(args.raiz)
    filas, otros = [], []
    for ds in args.datasets:
        for actual, _, archivos in os.walk(os.path.join(args.raiz, ds)):
            for a in sorted(archivos):
                ruta = os.path.join(actual, a)
                if os.path.splitext(a)[1].lower() not in EXT_IMG:
                    otros.append({"dataset": ds, "archivo": os.path.relpath(ruta, args.raiz)})
                    continue
                md5, ancho, alto, error = revisar(ruta)
                clase = rocole.get(a) if ds == "rocole" else os.path.basename(actual)
                filas.append({"dataset": ds, "ruta": os.path.relpath(ruta, args.raiz),
                              "clase_original": clase, "bytes": os.path.getsize(ruta),
                              "md5": md5, "ancho": ancho, "alto": alto,
                              "legible": error is None, "error": error})
            print(f"\r{ds}: {sum(f['dataset'] == ds for f in filas)} imágenes revisadas", end="")
        print()

    df = pd.DataFrame(filas)
    df.to_csv(os.path.join(args.salida, "inventario.csv"), index=False)
    ok = df[df["legible"]]

    titulo("IMÁGENES POR DATASET Y CLASE")
    print(df.groupby(["dataset", "clase_original"], dropna=False).size().to_string())

    titulo("TAMAÑO EN PÍXELES (mín / mediana / máx)")
    print(ok.groupby("dataset")[["ancho", "alto"]].agg(["min", "median", "max"]).to_string())

    titulo("IMÁGENES QUE NO SE PUEDEN ABRIR")
    mal = df[~df["legible"]]
    if len(mal):
        print(mal.groupby(["dataset", "error"]).agg(cantidad=("ruta", "size"),
                                                   bytes_max=("bytes", "max")).to_string())
    else:
        print("Ninguna")

    titulo("COPIAS EXACTAS (mismo md5)")
    for ds, sub in ok.groupby("dataset"):
        distintas = sub["md5"].nunique()
        print(f"{ds}: {len(sub)} legibles, {distintas} distintas, {len(sub) - distintas} copias sobrantes")
    conflicto = ok.groupby("md5")["clase_original"].nunique()
    conflicto = conflicto[conflicto > 1].index
    print("\nCopias con clases distintas (misma foto en dos clases):")
    if len(conflicto):
        print(ok[ok["md5"].isin(conflicto)].groupby(["dataset", "md5", "clase_original"])
              .size().to_string())
    else:
        print("Ninguna")

    titulo("ARCHIVOS QUE NO SON IMÁGENES")
    print(pd.DataFrame(otros).to_string(index=False) if otros else "Ninguno")

    titulo("IMÁGENES UTILIZABLES (legibles, sin copias, sin clases en conflicto)")
    util = ok[~ok["md5"].isin(conflicto)].drop_duplicates("md5")
    print(pd.crosstab(util["dataset"], util["clase_original"].fillna("sin etiqueta"),
                      margins=True, margins_name="total").to_string())

    print("\nTabla por imagen:", os.path.join(args.salida, "inventario.csv"))


if __name__ == "__main__":
    main()
