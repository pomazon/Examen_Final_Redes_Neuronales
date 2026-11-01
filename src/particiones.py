"""Limpia los datos, unifica las etiquetas y arma las particiones sin fuga.

Saposoa (entrenamiento, validación y prueba):
    Las fotos se ordenan por su orden de captura (número o fecha del nombre) y se
    agrupan en bloques de fotos consecutivas. Cada bloque va entero a una sola
    partición, para que las tomas repetidas de una misma hoja no queden repartidas.

RoCoLe y Uganda (prueba externa):
    Se unen las fotos casi iguales (similitud mayor o igual al umbral de cada
    dataset) y se deja una por grupo.

Uso:
    python src/particiones.py --inventario results/inventario/inventario.csv \
        --pares results/duplicados/pares.csv --vectores data/processed
"""
import argparse
import os
import re

import numpy as np
import pandas as pd

# Etiqueta original -> etiqueta del proyecto. None: la clase no existe en Saposoa.
MAPEO = {
    "saposoa": {"Sanas": "sana", "Roya": "roya", "Ojo_Gallo": "ojo_gallo"},
    "rocole": {"healthy": "sana", "rust_level_1": "roya", "rust_level_2": "roya",
               "rust_level_3": "roya", "rust_level_4": "roya", "red_spider_mite": None},
    "uganda": {"Health leaves": "sana", "leaf rust": "roya", "phoma": None},
}
UMBRAL_EXTERNO = {"rocole": 0.975, "uganda": 0.93}  # elegidos mirando los pares de ejemplo
PROPORCION = {"train": 0.70, "val": 0.15, "test": 0.15}


def orden_captura(ruta):
    """'Sana 257.jpg' -> 257 ; '20251129_132523.jpg' -> 20251129132523"""
    return int("".join(re.findall(r"\d+", os.path.basename(ruta))) or 0)


def agrupar(n, i, j):
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
    return np.array([raiz(k) for k in range(n)])


def repartir(unidades, semilla):
    """Asigna cada unidad (bloque o imagen) a la partición a la que más le falta."""
    rng = np.random.default_rng(semilla)
    total = sum(t for _, t in unidades)
    falta = {k: v * total for k, v in PROPORCION.items()}
    asignado = {}
    for idx in rng.permutation(len(unidades)):
        u, t = unidades[idx]
        destino = max(falta, key=falta.get)
        asignado[u] = destino
        falta[destino] -= t
    return asignado


def max_sim_con_train(df, vec):
    """Para cada imagen de val/test, la mayor similitud con una imagen de train de su clase."""
    salida = []
    for clase, sub in df.groupby("clase"):
        tr = np.stack([vec[r] for r in sub.loc[sub["particion"] == "train", "ruta"]])
        ot = sub[sub["particion"] != "train"]
        if len(ot):
            salida.append((np.stack([vec[r] for r in ot["ruta"]]) @ tr.T).max(axis=1))
    return np.concatenate(salida)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inventario", default="results/inventario/inventario.csv")
    p.add_argument("--pares", default="results/duplicados/pares.csv")
    p.add_argument("--vectores", default="data/processed")
    p.add_argument("--bloque", type=int, default=10, help="fotos consecutivas por bloque")
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--salida", default="data/splits.csv")
    p.add_argument("--reporte", default="results/particiones")
    args = p.parse_args()
    os.makedirs(args.reporte, exist_ok=True)
    excluidas = []

    inv = pd.read_csv(args.inventario)
    excluidas.append(("no se pueden abrir", inv[~inv["legible"]]))
    df = inv[inv["legible"]].copy()
    df["clase"] = [MAPEO[d].get(c) for d, c in zip(df["dataset"], df["clase_original"])]
    df["clase_txt"] = df["clase"].fillna("otra")

    # 1. Copias exactas: si todas tienen la misma clase (ya unificada) se deja una;
    #    si tienen clases distintas no se sabe cuál es la correcta y se excluyen todas.
    conflicto = df.groupby("md5")["clase_txt"].transform("nunique") > 1
    excluidas.append(("copia exacta con clases distintas", df[conflicto]))
    df = df[~conflicto]
    excluidas.append(("copia exacta sobrante", df[df.duplicated("md5")]))
    df = df.drop_duplicates("md5").copy()

    # 2. Fotos casi iguales en RoCoLe y Uganda
    pares = pd.read_csv(args.pares)
    df["grupo"] = df["ruta"]
    for ds, umbral in UMBRAL_EXTERNO.items():
        sub = df[df["dataset"] == ds]
        pos = {r: k for k, r in enumerate(sub["ruta"])}
        pp = pares[(pares["dataset"] == ds) & (pares["similitud"] >= umbral)
                   & pares["ruta_a"].isin(pos) & pares["ruta_b"].isin(pos)]
        g = agrupar(len(sub), pp["ruta_a"].map(pos), pp["ruta_b"].map(pos))
        df.loc[sub.index, "grupo"] = [f"{ds}_{x}" for x in g]
    externa = df["dataset"] != "saposoa"
    conflicto = externa & (df.groupby("grupo")["clase_txt"].transform("nunique") > 1)
    excluidas.append(("casi igual a otra foto de clase distinta", df[conflicto]))
    df = df[~conflicto]
    sobrante = (df["dataset"] != "saposoa") & df.duplicated("grupo")
    excluidas.append(("casi igual a otra foto (se deja una)", df[sobrante]))
    df = df[~sobrante]

    # 3. Clases que no existen en Saposoa
    excluidas.append(("clase sin equivalente en Saposoa", df[df["clase"].isna()]))
    df = df[df["clase"].notna()].copy()

    # 4. Particiones de Saposoa por bloques de capturas consecutivas
    df["particion"] = "externa"
    df["bloque"] = ""
    sap = df[df["dataset"] == "saposoa"].copy()
    sap["orden"] = sap["ruta"].map(orden_captura)
    for clase, sub in sap.groupby("clase"):
        sub = sub.sort_values("orden")
        bloques = [f"{clase}_{k // args.bloque:03d}" for k in range(len(sub))]
        sap.loc[sub.index, "bloque"] = bloques
        asign = repartir(list(pd.Series(bloques).value_counts().sort_index().items()), args.semilla)
        sap.loc[sub.index, "particion"] = [asign[b] for b in bloques]
    df.loc[sap.index, ["particion", "bloque"]] = sap[["particion", "bloque"]]

    # 5. Verificación con los vectores de la ResNet50
    npz = np.load(os.path.join(args.vectores, "vectores_saposoa.npz"), allow_pickle=True)
    vec = dict(zip(npz["rutas"], npz["v"]))
    sap = df[df["dataset"] == "saposoa"].copy()
    sap["orden"] = sap["ruta"].map(orden_captura)

    print("¿Las fotos consecutivas se parecen más que dos fotos al azar? (similitud media)")
    rng = np.random.default_rng(args.semilla)
    for clase, sub in sap.groupby("clase"):
        r = sub.sort_values("orden")["ruta"].values
        v = np.stack([vec[x] for x in r])
        vecinas = (v[:-1] * v[1:]).sum(axis=1)
        a, b = rng.integers(0, len(v), 2000), rng.integers(0, len(v), 2000)
        azar = (v[a] * v[b]).sum(axis=1)[a != b]
        print(f"  {clase:10s} consecutivas: {vecinas.mean():.3f}   al azar: {azar.mean():.3f}")

    aleatoria = sap.copy()
    for clase, sub in aleatoria.groupby("clase"):
        asign = repartir([(r, 1) for r in sub["ruta"]], args.semilla)
        aleatoria.loc[sub.index, "particion"] = sub["ruta"].map(asign)
    print("\nMayor similitud de cada imagen de val/test con alguna de train (misma clase):")
    for nombre, d in [("por bloques", sap), ("al azar, por imagen", aleatoria)]:
        m = max_sim_con_train(d, vec)
        print(f"  {nombre:20s} mediana {np.median(m):.3f}   ≥0,95: {(m >= 0.95).mean():5.1%}"
              f"   ≥0,96: {(m >= 0.96).mean():5.1%}")

    # 6. Guardar
    cols = ["dataset", "ruta", "md5", "clase_original", "clase", "particion", "bloque"]
    df[cols].sort_values(["dataset", "particion", "clase", "ruta"]).to_csv(args.salida, index=False)
    exc = pd.concat([e.assign(motivo=m) for m, e in excluidas if len(e)])
    exc[["dataset", "ruta", "clase_original", "motivo"]].to_csv(
        os.path.join(args.reporte, "excluidas.csv"), index=False)

    print(f"\nImágenes excluidas (bloque de {args.bloque}, semilla {args.semilla}):")
    print(pd.crosstab(exc["motivo"], exc["dataset"], margins=True, margins_name="total").to_string())
    print("\nImágenes por dataset, partición y clase:")
    tabla = pd.crosstab([df["dataset"], df["particion"]], df["clase"], margins=True, margins_name="total")
    print(tabla.to_string())
    tabla.to_csv(os.path.join(args.reporte, "conteo.csv"))
    print("\nParticiones:", args.salida)
    print("Excluidas:", os.path.join(args.reporte, "excluidas.csv"))


if __name__ == "__main__":
    main()
