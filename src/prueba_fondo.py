"""Prueba directa de si los modelos usan el fondo de Saposoa para decidir.

Sobre las imágenes de Saposoa test se evalúa cada modelo en tres versiones:
    original    la foto tal cual
    fondo_gris  la hoja se conserva y el fondo se pinta de gris neutro
    solo_fondo  la hoja se borra (se pinta con el color del fondo) y solo queda el fondo

Lectura:
    - Si con solo_fondo el F1 queda muy por encima del azar (0,33 en tres clases),
      el modelo puede reconocer la clase mirando solo el fondo: usa un atajo.
    - Si con fondo_gris el F1 cae respecto del original, el modelo depende del fondo.

Aquí se deja documentado el objetivo del análisis para seguir sin perder el contexto
experimental de la prueba.

Uso:
    python src/prueba_fondo.py --modelos resnet50_basico resnet50_fondo vit_b_16_basico vit_b_16_fondo
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from scipy import ndimage

from aumentos import mascara_hoja, transformacion
from entrenar import crear_modelo
from metricas import CLASES, evaluar


def versiones(im):
    """Devuelve (fondo_gris, solo_fondo), o None si no hay máscara confiable."""
    m = mascara_hoja(im)
    if m is None:
        return None
    a = np.array(im, float)
    hoja = ndimage.binary_dilation(m, iterations=6)  # margen para no dejar bordes de hoja
    color_fondo = np.median(a[~hoja], axis=0)
    # Conservamos la hoja sobre gris; en la segunda versión rellenamos la zona de la hoja con el color del fondo.
    gris = np.where(m[..., None], a, 128.0)
    sin_hoja = np.where(hoja[..., None], color_fondo, a)
    return (Image.fromarray(gris.astype(np.uint8)), Image.fromarray(sin_hoja.astype(np.uint8)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modelos", nargs="+", required=True)
    p.add_argument("--semilla", type=int, default=0)
    p.add_argument("--splits", default="data/splits.csv")
    p.add_argument("--imagenes", default="data/processed/imagenes")
    p.add_argument("--carpeta_modelos", default="data/processed/modelos")
    p.add_argument("--salida", default="results/prueba_fondo")
    args = p.parse_args()
    os.makedirs(args.salida, exist_ok=True)
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    t = transformacion("evaluacion")

    df = pd.read_csv(args.splits)
    test = df[(df["dataset"] == "saposoa") & (df["particion"] == "test")]
    conjuntos = {"original": [], "fondo_gris": [], "solo_fondo": []}
    clases = []
    for r in test.itertuples():
        im = Image.open(os.path.join(args.imagenes, r.ruta)).convert("RGB")
        v = versiones(im)
        # Excluimos la imagen de las tres versiones si su máscara falla, para comparar siempre las mismas fotos.
        if v is None:
            continue
        conjuntos["original"].append(im)
        conjuntos["fondo_gris"].append(v[0])
        conjuntos["solo_fondo"].append(v[1])
        clases.append(r.clase)
    print(f"Imágenes de Saposoa test con máscara confiable: {len(clases)} de {len(test)}")

    filas = []
    for nombre in args.modelos:
        with open(f"configs/{nombre}.yaml", "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        m = crear_modelo(cfg["modelo"], preentrenado=False)
        m.load_state_dict(torch.load(os.path.join(args.carpeta_modelos, f"{nombre}_s{args.semilla}.pt"),
                                     map_location=dispositivo))
        m.eval().to(dispositivo)
        # Usamos los mismos pesos en las tres versiones para medir cómo cambia la predicción al cambiar el fondo o retirar la hoja.
        for version, imagenes in conjuntos.items():
            proba = []
            with torch.no_grad():
                for i in range(0, len(imagenes), 64):
                    x = torch.stack([t(im) for im in imagenes[i:i + 64]]).to(dispositivo)
                    proba.append(torch.softmax(m(x), 1).cpu().numpy())
            proba = np.concatenate(proba)
            r = evaluar(clases, proba, "saposoa")
            pred = pd.Series(np.array(CLASES)[proba.argmax(1)]).value_counts(normalize=True)
            filas.append({"modelo": nombre, "version": version, "f1_macro": r["f1_macro"],
                          "exactitud": r["exactitud"],
                          **{f"pred_{c}": pred.get(c, 0.0) for c in CLASES}})
        del m
        torch.cuda.empty_cache()

    tabla = pd.DataFrame(filas)
    tabla.to_csv(os.path.join(args.salida, "prueba_fondo.csv"), index=False)
    print("\nF1 macro en Saposoa test según lo que se le muestra al modelo (azar = 0,33)")
    print(tabla.pivot(index="modelo", columns="version", values="f1_macro")
          [["original", "fondo_gris", "solo_fondo"]].round(3).to_string())
    print("\nCon solo_fondo: qué clase predice cada modelo (proporción)")
    print(tabla[tabla["version"] == "solo_fondo"].set_index("modelo")
          [[f"pred_{c}" for c in CLASES]].round(2).to_string())

    idx = [clases.index(c) for c in CLASES]
    fig, ejes = plt.subplots(len(idx), 3, figsize=(6.5, 2.3 * len(idx)))
    for fila, i in zip(ejes, idx):
        for eje, (version, imagenes) in zip(fila, conjuntos.items()):
            eje.imshow(imagenes[i])
            eje.set_title(f"{version} (real: {clases[i]})", fontsize=7)
            eje.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(args.salida, "versiones.png"), dpi=80)
    print("\nFigura:", os.path.join(args.salida, "versiones.png"))


if __name__ == "__main__":
    main()
