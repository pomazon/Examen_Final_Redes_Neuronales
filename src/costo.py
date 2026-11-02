"""Costo de cómputo de los entrenamientos (requisito R6), a partir de metricas.json.

Uso:
    python src/costo.py
"""
import argparse
import glob
import json
import os

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--entrenamiento", default="results/entrenamiento")
    p.add_argument("--salida", default="results/costo.csv")
    args = p.parse_args()

    filas = [json.load(open(f)) for f in glob.glob(os.path.join(args.entrenamiento, "*", "semilla_*", "metricas.json"))]
    df = pd.DataFrame(filas)
    tabla = df.groupby(["config", "gpu"]).agg(corridas=("semilla", "count"),
                                              minutos_por_corrida=("minutos_entrenamiento", "mean"),
                                              minutos_total=("minutos_entrenamiento", "sum")).round(1)
    tabla.to_csv(args.salida)
    print(tabla.to_string())
    total = df["minutos_entrenamiento"].sum()
    print(f"\nTotal: {len(df)} corridas, {total:.0f} minutos = {total / 60:.2f} horas de GPU")
    print("No incluye: descarga, inventario, duplicados, línea base ni Grad-CAM (minutos, sobre todo en CPU)")
    print("Tabla:", args.salida)


if __name__ == "__main__":
    main()
