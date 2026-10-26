"""Limpia los datos, unifica las etiquetas y genera las particiones sin fuga.

- Saposoa se divide en entrenamiento / validación / prueba (70 / 15 / 15),
  estratificado por clase y por grupos de duplicados: las imágenes de un mismo
  grupo quedan siempre en la misma partición.
- RoCoLe y Uganda se usan solo como prueba externa, con las clases comunes
  (sana y roya).

Uso:
    python src/particiones.py --raiz data/raw --inventario results/inventario --umbral 0.97
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

# Etiqueta original -> etiqueta unificada. None = se excluye de la prueba externa.
MAPEO = {
    "saposoa": {"Sanas": "sana", "Roya": "roya", "Ojo_Gallo": "ojo_gallo"},
    "uganda": {"Health leaves": "sana", "leaf rust": "roya", "phoma": None},
    "rocole": {"healthy": "sana", "rust_level_1": "roya", "rust_level_2": "roya",
               "rust_level_3": "roya", "rust_level_4": "roya", "red_spider_mite": None},
}
PROPORCION = {"train": 0.70, "val": 0.15, "test": 0.15}


def agrupar(n, i, j):
    """Union-find: devuelve el id de grupo de cada uno de los n elementos."""
    padre = list(range(n))

    def raiz(k):
        while padre[k] != k:
            padre[k] = padre[padre[k]]
            k = padre[k]
        return k

    for a, b in zip(i, j):
        ra, rb = raiz(a), raiz(b)
        if ra != rb:
            padre[rb] = ra
    return [raiz(k) for k in range(n)]


def repartir_grupos(grupos, semilla):
    """Asigna cada grupo a la partición que más imágenes le falten, en orden aleatorio."""
    rng = np.random.default_rng(semilla)
    orden = rng.permutation(len(grupos))
    total = sum(t for _, t in grupos)
    meta = {k: v * total for k, v in PROPORCION.items()}
    actual = {k: 0 for k in PROPORCION}
    asignacion = {}
    for idx in orden:
        g, t = grupos[idx]
        destino = max(PROPORCION, key=lambda k: meta[k] - actual[k])
        asignacion[g] = destino
        actual[destino] += t
    return asignacion


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raiz", default="data/raw")
    p.add_argument("--inventario", default="results/inventario")
    p.add_argument("--umbral", type=float, default=0.97,
                   help="similitud coseno a partir de la cual dos imágenes son duplicadas")
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--salida", default="data/splits.csv")
    args = p.parse_args()

    inv = pd.read_csv(os.path.join(args.inventario, "inventario_imagenes.csv"))
    registro = []

    # 1. Imágenes dañadas
    danadas = inv["corrupta"].astype(bool)
    registro.append(("dañadas", inv[danadas].groupby("dataset").size().to_dict()))
    inv = inv[~danadas].copy()

    # 2. Etiqueta original (RoCoLe la trae en un Excel aparte)
    inv["clase_original"] = inv["carpeta"]
    xlsx = glob.glob(os.path.join(args.raiz, "rocole", "*", "Annotations", "*classes*.xlsx"))[0]
    etq = pd.read_excel(xlsx).set_index("File")["Multiclass.Label"]
    es_rc = inv["dataset"] == "rocole"
    inv.loc[es_rc, "clase_original"] = inv.loc[es_rc, "ruta"].map(lambda r: etq.get(os.path.basename(r)))
    sin_etq = inv["clase_original"].isna()
    registro.append(("sin etiqueta", inv[sin_etq].groupby("dataset").size().to_dict()))
    inv = inv[~sin_etq]

    # 3. Copias exactas con etiquetas distintas: se excluyen todas
    conflicto = inv.groupby("md5")["clase_original"].transform("nunique") > 1
    registro.append(("copias exactas con etiquetas distintas",
                     inv[conflicto].groupby("dataset").size().to_dict()))
    inv = inv[~conflicto]

    # 4. Copias exactas: se deja una
    copia = inv.duplicated("md5", keep="first")
    registro.append(("copias exactas sobrantes", inv[copia].groupby("dataset").size().to_dict()))
    inv = inv[~copia].copy()

    # 5. Etiqueta unificada
    inv["clase"] = [MAPEO[d].get(c) for d, c in zip(inv["dataset"], inv["clase_original"])]
    fuera = inv["clase"].isna()
    registro.append(("clases sin equivalente en Saposoa",
                     inv[fuera].groupby(["dataset", "clase_original"]).size().to_dict()))
    inv = inv[~fuera].copy()

    # 6. Grupos de duplicados (similitud CNN >= umbral, dentro de cada dataset)
    pares = pd.read_csv(os.path.join(args.inventario, "pares_similares_cnn.csv"))
    pares = pares[pares["similitud"] >= args.umbral]
    inv["grupo"] = ""
    for ds, sub in inv.groupby("dataset"):
        pos = {r: k for k, r in enumerate(sub["ruta"])}
        pp = pares[(pares["dataset"] == ds) & pares["ruta_a"].isin(pos) & pares["ruta_b"].isin(pos)]
        g = agrupar(len(sub), pp["ruta_a"].map(pos), pp["ruta_b"].map(pos))
        inv.loc[sub.index, "grupo"] = [f"{ds}_{x}" for x in g]

    # Grupos que mezclan clases: no se puede saber cuál etiqueta es la correcta
    mezcla = inv.groupby("grupo")["clase"].transform("nunique") > 1
    registro.append(("grupos de duplicados con clases mezcladas",
                     inv[mezcla].groupby("dataset").size().to_dict()))
    inv = inv[~mezcla].copy()

    # 7. Particiones
    inv["particion"] = "externa"
    sap = inv["dataset"] == "saposoa"
    for clase, sub in inv[sap].groupby("clase"):
        grupos = list(sub.groupby("grupo").size().items())
        asign = repartir_grupos(grupos, args.semilla)
        inv.loc[sub.index, "particion"] = sub["grupo"].map(asign)

    # Comprobación: ningún grupo está en dos particiones
    assert (inv.groupby("grupo")["particion"].nunique() == 1).all()

    cols = ["dataset", "ruta", "md5", "clase_original", "clase", "grupo", "particion"]
    inv[cols].sort_values(["dataset", "particion", "clase", "ruta"]).to_csv(args.salida, index=False)

    print(f"Umbral de duplicado: similitud >= {args.umbral}   Semilla: {args.semilla}\n")
    print("Imágenes excluidas:")
    for motivo, n in registro:
        print(f"  {motivo}: {n if n else 0}")
    print("\nImágenes por dataset, partición y clase:")
    print(pd.crosstab([inv["dataset"], inv["particion"]], inv["clase"], margins=True).to_string())
    print("\nGrupos con más de una imagen en Saposoa:",
          int((inv[sap].groupby("grupo").size() > 1).sum()))
    print("\nParticiones guardadas en:", args.salida)


if __name__ == "__main__":
    main()
