"""Detalle de los problemas que encontró el inventario: imágenes dañadas, duplicados,
casi-duplicados y etiquetas de RoCoLe.

Uso:
    python src/analizar_inventario.py --raiz data/raw --inventario results/inventario
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd


def titulo(texto):
    print("\n" + "=" * 70 + "\n" + texto + "\n" + "=" * 70)


def grupos_casi_duplicados(phashes, umbral):
    """Une en un mismo grupo las imágenes a distancia <= umbral (union-find)."""
    h = np.array([int(x, 16) for x in phashes], dtype=np.uint64)
    padre = list(range(len(h)))

    def raiz(i):
        while padre[i] != i:
            padre[i] = padre[padre[i]]
            i = padre[i]
        return i

    for i in range(len(h) - 1):
        x = np.bitwise_xor(h[i], h[i + 1:])
        dist = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)
        for j in np.nonzero(dist <= umbral)[0] + i + 1:
            a, b = raiz(i), raiz(int(j))
            if a != b:
                padre[b] = a
    return [raiz(i) for i in range(len(h))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raiz", default="data/raw")
    p.add_argument("--inventario", default="results/inventario")
    p.add_argument("--umbral", type=int, default=4)
    args = p.parse_args()

    df = pd.read_csv(os.path.join(args.inventario, "inventario_imagenes.csv"))
    df["corrupta"] = df["corrupta"].astype(bool)

    titulo("IMÁGENES DAÑADAS: MOTIVO")
    dan = df[df["corrupta"]]
    print(dan.groupby(["dataset", "error"]).size().to_string())
    print("\nEjemplos:")
    print(dan.groupby("dataset").head(3)[["dataset", "ruta", "bytes"]].to_string(index=False))

    ok = df[~df["corrupta"]].copy()

    titulo("DUPLICADOS EXACTOS POR DATASET")
    tam = ok.groupby("md5")["ruta"].transform("size")
    ok["copias"] = tam
    for ds, sub in ok.groupby("dataset"):
        rep = sub[sub["copias"] > 1]
        n_grupos = rep["md5"].nunique()
        print(f"{ds}: {len(sub)} imágenes, {len(rep)} en grupos repetidos, "
              f"{n_grupos} grupos, {len(rep) - n_grupos} copias sobrantes")
        if len(rep):
            print("   tamaño de grupo:", rep.groupby("md5").size().value_counts().sort_index().to_dict())
            print("   copias sobrantes por carpeta:",
                  (rep.groupby("carpeta").size() - rep.groupby("carpeta")["md5"].nunique()).to_dict())

    titulo("DUPLICADOS EXACTOS CON ETIQUETAS DISTINTAS")
    conflicto = ok.groupby("md5")["carpeta"].nunique()
    for md5 in conflicto[conflicto > 1].index:
        print(ok.loc[ok["md5"] == md5, ["dataset", "carpeta", "ruta"]].to_string(index=False))
        print()

    titulo(f"GRUPOS DE CASI-DUPLICADOS (pHash, distancia <= {args.umbral})")
    unicas = ok.drop_duplicates("md5").copy()
    filas_grupo = []
    for ds, sub in unicas.groupby("dataset"):
        sub = sub.reset_index(drop=True)
        sub["grupo"] = grupos_casi_duplicados(sub["phash"], args.umbral)
        tam = sub.groupby("grupo").size()
        clases = sub.groupby("grupo")["carpeta"].nunique()
        print(f"{ds}: {len(sub)} imágenes distintas -> {len(tam)} grupos independientes")
        print(f"   grupos con más de una imagen: {(tam > 1).sum()} "
              f"(el mayor tiene {tam.max()} imágenes)")
        print(f"   grupos que mezclan clases: {(clases > 1).sum()}")
        multi = tam[tam > 1].index[:3]
        for g in multi:
            print("   ejemplo:", sub.loc[sub["grupo"] == g, "ruta"].tolist()[:4])
        filas_grupo.append(sub[["dataset", "ruta", "carpeta", "md5", "grupo"]])
    salida = os.path.join(args.inventario, "grupos_casi_duplicados.csv")
    pd.concat(filas_grupo).to_csv(salida, index=False)
    print("\nGrupos guardados en:", salida)

    titulo("ETIQUETAS DE ROCOLE")
    carpeta = glob.glob(os.path.join(args.raiz, "rocole", "*", "Annotations"))
    if not carpeta:
        print("No se encontró la carpeta Annotations")
        return
    carpeta = carpeta[0]
    for csv in glob.glob(os.path.join(carpeta, "*.csv")):
        print("Archivo:", os.path.basename(csv))
        t = pd.read_csv(csv)
        print("Filas:", len(t), " Columnas:", list(t.columns))
        print(t.head(5).to_string())
        for col in t.columns:
            if t[col].nunique() <= 10:
                print(f"\nValores de '{col}':")
                print(t[col].value_counts().to_string())
    for xlsx in glob.glob(os.path.join(carpeta, "*.xlsx")):
        print("\nArchivo:", os.path.basename(xlsx))
        try:
            print(pd.read_excel(xlsx).head(10).to_string())
        except Exception as e:
            print("No se pudo leer:", e)


if __name__ == "__main__":
    main()
