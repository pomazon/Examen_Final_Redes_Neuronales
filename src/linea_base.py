"""Líneas base: modelos clásicos sobre características fijas.

Tres tipos de características:
    color_textura  histograma de color (HSV) y de textura (LBP) de toda la imagen
    solo_borde     histograma de color solo del borde de la imagen (el fondo en Saposoa):
                   si con esto se acierta mucho, el fondo delata la clase
    resnet50       vector de una ResNet50 preentrenada en ImageNet, sin entrenarla

Dos clasificadores: regresión logística y random forest. Se entrenan con Saposoa train,
se elige el hiperparámetro con val y se evalúa en Saposoa test, RoCoLe y Uganda.

Uso:
    python src/linea_base.py --splits data/splits.csv --imagenes data/processed/imagenes
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.feature import local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import models, transforms

from metricas import evaluar

SEMILLAS = [0, 1, 2]


def hist_hsv(pixeles):
    """Histogramas HSV normalizados para describir color de una imagen o región."""
    h, _ = np.histogramdd(pixeles.reshape(-1, 3), bins=(8, 4, 4), range=((0, 256),) * 3)
    return h.ravel() / max(h.sum(), 1)


def color_textura(im):
    """Vector de color + textura global de una hoja, útil como línea base clásica."""
    im = im.resize((224, 224))
    gris = np.array(im.convert("L"))
    lbp = local_binary_pattern(gris, 8, 1, "uniform")
    h_lbp, _ = np.histogram(lbp, bins=10, range=(0, 10))
    return np.concatenate([hist_hsv(np.array(im.convert("HSV"))), h_lbp / h_lbp.sum()])


def solo_borde(im, ancho=0.1):
    """Extrae color del borde de la imagen para comprobar si el fondo delata la clase."""
    hsv = np.array(im.resize((224, 224)).convert("HSV"))
    b = int(224 * ancho)
    mascara = np.ones((224, 224), bool)
    mascara[b:-b, b:-b] = False
    return hist_hsv(hsv[mascara])


@torch.no_grad()
def resnet50(rutas):
    """Genera embeddings fijos usando ResNet50 preentrenada en ImageNet."""
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    modelo.fc = torch.nn.Identity()
    modelo.eval().to(dispositivo)
    t = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    salida = []
    for i in range(0, len(rutas), 64):
        x = torch.stack([t(Image.open(r).convert("RGB")) for r in rutas[i:i + 64]])
        salida.append(modelo(x.to(dispositivo)).cpu().numpy())
    return np.concatenate(salida)


def caracteristicas(df, carpeta, cache):
    """Calcula y cachea las tres familias de características usadas como líneas base."""
    if os.path.exists(cache):
        return dict(np.load(cache))

    rutas = [os.path.join(carpeta, r) for r in df["ruta"]]
    ct, bo = [], []
    for k, r in enumerate(rutas, 1):
        with Image.open(r) as im:
            im = im.convert("RGB")
            ct.append(color_textura(im))
            bo.append(solo_borde(im))
        if k % 500 == 0:
            print(f"\r  color y textura: {k}/{len(rutas)}", end="")
    print("\n  resnet50...")
    feats = {"color_textura": np.array(ct), "solo_borde": np.array(bo), "resnet50": resnet50(rutas)}
    directorio = os.path.dirname(cache)
    if directorio:
        os.makedirs(directorio, exist_ok=True)
    np.savez(cache, **feats)
    return feats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--splits", default="data/splits.csv")
    p.add_argument("--imagenes", default="data/processed/imagenes")
    p.add_argument("--cache", default="data/processed/caracteristicas.npz")
    p.add_argument("--salida", default="results/linea_base")
    args = p.parse_args()
    os.makedirs(args.salida, exist_ok=True)

    df = pd.read_csv(args.splits)
    feats = caracteristicas(df, args.imagenes, args.cache)
    conjunto = df["dataset"].where(df["dataset"] != "saposoa", "saposoa_" + df["particion"])
    idx = {c: np.where(conjunto == c)[0] for c in conjunto.unique()}
    y = df["clase"].values
    evaluaciones = [("saposoa_test", "saposoa"), ("rocole", "rocole"), ("uganda", "uganda")]

    filas = []
    for nombre_f, X in feats.items():
        tr, va = idx["saposoa_train"], idx["saposoa_val"]

        # Regresión logística: C elegido con validación (macro-F1)
        mejor = None
        for C in [0.01, 0.1, 1, 10]:
            m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
            m.fit(X[tr], y[tr])
            f1 = evaluar(y[va], m.predict_proba(X[va]), "saposoa")["f1_macro"]
            if mejor is None or f1 > mejor[0]:
                mejor = (f1, C, m)
        for conj, ds in evaluaciones:
            r = evaluar(y[idx[conj]], mejor[2].predict_proba(X[idx[conj]]), ds)
            filas.append({"caracteristicas": nombre_f, "modelo": f"reg_logistica (C={mejor[1]})",
                          "semilla": 0, "evaluacion": conj, **r})

        # Random forest: 3 semillas
        for s in SEMILLAS:
            m = RandomForestClassifier(n_estimators=300, random_state=s, n_jobs=-1).fit(X[tr], y[tr])
            for conj, ds in evaluaciones:
                r = evaluar(y[idx[conj]], m.predict_proba(X[idx[conj]]), ds)
                filas.append({"caracteristicas": nombre_f, "modelo": "random_forest",
                              "semilla": s, "evaluacion": conj, **r})

    res = pd.DataFrame(filas)
    res.to_csv(os.path.join(args.salida, "metricas_por_semilla.csv"), index=False)

    metricas = ["f1_macro", "exactitud", "auc", "pred_ojo_gallo"]
    resumen = res.groupby(["caracteristicas", "modelo", "evaluacion"])[metricas].agg(["mean", "std"])
    resumen.to_csv(os.path.join(args.salida, "metricas.csv"))

    tabla = res.groupby(["caracteristicas", "modelo", "evaluacion"])["f1_macro"].agg(["mean", "std"])
    tabla = tabla.apply(lambda f: f"{f['mean']:.3f} ± {f['std'] if f['std'] == f['std'] else 0:.3f}", axis=1)
    print("\nF1 macro (media ± desviación en 3 semillas; la regresión logística es determinista)")
    print(tabla.unstack("evaluacion")[["saposoa_test", "rocole", "uganda"]].to_string())
    auc = res[res["evaluacion"] != "saposoa_test"].groupby(
        ["caracteristicas", "modelo", "evaluacion"])["auc"].mean().unstack("evaluacion")
    print("\nAUC en prueba externa (roya frente a sana)")
    print(auc.round(3).to_string())
    print("\nResultados:", args.salida)


if __name__ == "__main__":
    main()
