"""Análisis de errores a partir de las predicciones guardadas (no necesita GPU).

- Matrices de confusión por modelo y conjunto, sumando las 3 semillas. En RoCoLe y
  Uganda se muestran las 3 clases predichas, para ver cuántas fotos de campo se van a
  "ojo de gallo".
- Los errores más seguros (el modelo se equivoca con alta probabilidad) del modelo
  indicado, con una figura de ejemplos.

Uso:
    python src/errores.py --modelos resnet50_basico resnet50_fondo vit_b_16_basico vit_b_16_fondo
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from metricas import CLASES

CONJUNTOS = ["saposoa_test", "rocole", "uganda"]


def leer(nombre, carpeta):
    archivos = glob.glob(os.path.join(carpeta, nombre, "semilla_*", "predicciones.csv"))
    df = pd.concat([pd.read_csv(f).assign(semilla=f.split("semilla_")[1][0]) for f in archivos])
    df["pred"] = np.array(CLASES)[df[[f"p_{c}" for c in CLASES]].values.argmax(1)]
    df["confianza"] = df[[f"p_{c}" for c in CLASES]].max(axis=1)
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modelos", nargs="+", required=True)
    p.add_argument("--ejemplos_de", default=None, help="modelo para la figura de errores")
    p.add_argument("--semilla", default="0", help="semilla del modelo para la figura de errores")
    p.add_argument("--entrenamiento", default="results/entrenamiento")
    p.add_argument("--imagenes", default="data/processed/imagenes")
    p.add_argument("--salida", default="results/errores")
    args = p.parse_args()
    os.makedirs(args.salida, exist_ok=True)

    fig, ejes = plt.subplots(len(args.modelos), 3, figsize=(10, 2.8 * len(args.modelos)))
    ejes = np.atleast_2d(ejes)
    for fila, nombre in zip(ejes, args.modelos):
        df = leer(nombre, args.entrenamiento)
        for eje, conj in zip(fila, CONJUNTOS):
            sub = df[df["evaluacion"] == conj]
            reales = sorted(sub["clase"].unique())
            m = pd.crosstab(sub["clase"], sub["pred"]).reindex(index=reales, columns=CLASES, fill_value=0)
            m.to_csv(os.path.join(args.salida, f"confusion_{nombre}_{conj}.csv"))
            frac = m.div(m.sum(axis=1), axis=0)
            eje.imshow(frac.values, cmap="Blues", vmin=0, vmax=1)
            for (i, j), v in np.ndenumerate(m.values):
                eje.text(j, i, f"{v}\n{frac.values[i, j]:.0%}", ha="center", va="center", fontsize=7,
                         color="white" if frac.values[i, j] > 0.5 else "black")
            eje.set_xticks(range(3), [f"pred {c}" for c in CLASES], fontsize=7)
            eje.set_yticks(range(len(reales)), [f"real {c}" for c in reales], fontsize=7)
            eje.set_title(f"{nombre} · {conj}", fontsize=8)
    fig.suptitle("Matrices de confusión (suma de 3 semillas)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(args.salida, "confusion.png"), dpi=90)
    print("Figura:", os.path.join(args.salida, "confusion.png"))

    nombre = args.ejemplos_de or args.modelos[-1]
    df = leer(nombre, args.entrenamiento)
    df = df[df["semilla"] == str(args.semilla)]
    ext = df[df["evaluacion"] != "saposoa_test"].copy()
    ext["pred_2"] = np.where(ext["p_roya"] >= ext["p_sana"], "roya", "sana")
    errores = ext[ext["pred_2"] != ext["clase"]].sort_values("confianza", ascending=False)
    errores.to_csv(os.path.join(args.salida, f"errores_{nombre}.csv"), index=False)

    print(f"\n{nombre}, semilla {args.semilla}: errores entre sana y roya en la prueba externa")
    print(pd.crosstab([errores["dataset"], errores["clase"]], errores["pred_2"]).to_string())

    muestra = errores.groupby(["dataset", "clase"]).head(3)
    if muestra.empty:
        print("Sin errores para mostrar")
        return
    fig, ejes = plt.subplots(1, len(muestra), figsize=(2.2 * len(muestra), 2.6))
    for eje, (_, r) in zip(np.atleast_1d(ejes), muestra.iterrows()):
        eje.imshow(Image.open(os.path.join(args.imagenes, r["ruta"])).convert("RGB"))
        eje.set_title(f"{r['dataset']}\nreal {r['clase']} → {r['pred_2']}\n"
                      f"p_roya {r['p_roya']:.2f} p_sana {r['p_sana']:.2f}", fontsize=6)
        eje.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(args.salida, f"errores_{nombre}.png"), dpi=90)
    print("Figura:", os.path.join(args.salida, f"errores_{nombre}.png"))


if __name__ == "__main__":
    main()
