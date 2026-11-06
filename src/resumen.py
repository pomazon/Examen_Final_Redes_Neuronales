"""Junta los resultados de todos los entrenamientos y de la línea base en una tabla.

Este script consolida las métricas de cada corrida en una única vista resumida para
comparar modelos y líneas base en Saposoa, RoCoLe y Uganda.

Uso:
    python src/resumen.py
"""
import argparse
import glob
import json
import os

import pandas as pd

CONJUNTOS = ["saposoa_test", "rocole", "uganda"]


def media_desv(s):
    """Formatea una serie como media ± desviación estándar para la tabla final."""
    return f"{s.mean():.3f} ± {s.std(ddof=1) if len(s) > 1 else 0:.3f}"


def main():
    """Agrupa todas las métricas por método y evaluación.

    Las métricas de modelos entrenados se leen desde `results/entrenamiento/*/semilla_*/metricas.json`;
    las de la línea base se leen desde `results/linea_base/metricas_por_semilla.csv` y se añaden a la
    misma tabla final para que quede lista para el artículo.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--entrenamiento", default="results/entrenamiento")
    p.add_argument("--linea_base", default="results/linea_base/metricas_por_semilla.csv")
    p.add_argument("--salida", default="results/resumen.csv")
    args = p.parse_args()

    filas = []
    for f in glob.glob(os.path.join(args.entrenamiento, "*", "semilla_*", "metricas.json")):
        with open(f, "r", encoding="utf-8") as fh:
            r = json.load(fh)
        for c in CONJUNTOS:
            filas.append({"metodo": r["config"], "semilla": r["semilla"], "evaluacion": c,
                          "minutos": r["minutos_entrenamiento"], **r[c]})
    if os.path.exists(args.linea_base):
        lb = pd.read_csv(args.linea_base)
        lb["metodo"] = "base_" + lb["caracteristicas"] + "_" + lb["modelo"].str.split(" ").str[0]
        filas += lb.drop(columns=["caracteristicas", "modelo"]).to_dict("records")

    df = pd.DataFrame(filas)
    tabla = {}
    for m, nombre in [("f1_macro", "F1"), ("auc", "AUC"), ("pred_ojo_gallo", "% ojo de gallo")]:
        t = df.dropna(subset=[m]).groupby(["metodo", "evaluacion"])[m].apply(media_desv).unstack()
        for c in CONJUNTOS:
            if c in t:
                tabla[(nombre, c)] = t[c]
    tabla = pd.DataFrame(tabla)
    tabla["semillas"] = df.groupby("metodo")["semilla"].nunique()
    tabla["minutos"] = df.groupby("metodo")["minutos"].mean().round(1)
    tabla.to_csv(args.salida)
    pd.set_option("display.width", 250)
    print(tabla.fillna("").to_string())
    print("\nTabla:", args.salida)


if __name__ == "__main__":
    main()
