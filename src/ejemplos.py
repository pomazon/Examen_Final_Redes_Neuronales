"""Copia unas pocas imágenes de prueba a data/ejemplos/ para el notebook de demostración.

Una imagen por clase de Saposoa test y una sana y una con roya de RoCoLe y de Uganda.
Ninguna se usó para entrenar. Se guarda su ruta original para poder citar la fuente.

Uso:
    python src/ejemplos.py
"""
import argparse
import os
import shutil

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--splits", default="data/splits.csv")
    p.add_argument("--imagenes", default="data/processed/imagenes")
    p.add_argument("--destino", default="data/ejemplos")
    args = p.parse_args()
    os.makedirs(args.destino, exist_ok=True)

    df = pd.read_csv(args.splits)
    df = df[df["particion"].isin(["test", "externa"])]
    filas = []
    for (dataset, clase), grupo in df.groupby(["dataset", "clase"]):
        r = grupo.sample(1, random_state=0).iloc[0]
        nombre = f"{dataset}_{clase}.jpg"
        shutil.copy(os.path.join(args.imagenes, r["ruta"]), os.path.join(args.destino, nombre))
        filas.append({"archivo": nombre, "dataset": dataset, "clase": clase, "ruta_original": r["ruta"]})
    tabla = pd.DataFrame(filas)
    tabla.to_csv(os.path.join(args.destino, "fuentes.csv"), index=False)
    print(tabla.to_string(index=False))


if __name__ == "__main__":
    main()
