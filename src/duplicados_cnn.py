"""Busca duplicados reales comparando características de una ResNet50 preentrenada.

El pHash falla en Saposoa porque todas las fotos tienen el mismo fondo blanco.
Aquí cada imagen se representa con el vector de la última capa de la ResNet50
(antes del clasificador) y se compara por similitud coseno.

Uso:
    python src/duplicados_cnn.py --raiz data/raw --inventario results/inventario
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import models, transforms

TRANSFORMAR = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
TRAMOS = [(0.99, 1.01), (0.97, 0.99), (0.95, 0.97), (0.90, 0.95)]


def cargar(ruta):
    with Image.open(ruta) as im:
        im.draft("RGB", (512, 512))  # decodifica el JPEG reducido: mucho más rápido
        return TRANSFORMAR(im.convert("RGB"))


@torch.no_grad()
def caracteristicas(rutas, modelo, dispositivo, lote=32):
    vectores = []
    for i in range(0, len(rutas), lote):
        x = torch.stack([cargar(r) for r in rutas[i:i + lote]]).to(dispositivo)
        v = modelo(x)
        vectores.append(torch.nn.functional.normalize(v, dim=1).cpu())
        print(f"\r  {min(i + lote, len(rutas))}/{len(rutas)}", end="")
    print()
    return torch.cat(vectores).numpy()


def pares_similares(v, umbral):
    s = v @ v.T
    i, j = np.triu_indices(len(v), k=1)
    m = s[i, j] >= umbral
    return i[m], j[m], s[i, j][m]


def dibujar(pares, raiz, salida, titulo):
    filas = len(pares)
    if filas == 0:
        return
    fig, ejes = plt.subplots(filas, 2, figsize=(5, 2.3 * filas))
    ejes = np.atleast_2d(ejes)
    for (fila, (_, p)) in zip(ejes, pares.iterrows()):
        for eje, lado in zip(fila, ["a", "b"]):
            with Image.open(os.path.join(raiz, p[f"ruta_{lado}"])) as im:
                im.draft("RGB", (256, 256))
                im = im.convert("RGB")
                im.thumbnail((200, 200))
                eje.imshow(im)
            eje.set_title(f"{p[f'clase_{lado}']}  {os.path.basename(p[f'ruta_{lado}'])[:16]}", fontsize=7)
            eje.axis("off")
        fila[0].text(-0.1, 0.5, f"{p['similitud']:.3f}", transform=fila[0].transAxes,
                     ha="right", va="center", fontsize=9)
    fig.suptitle(titulo, fontsize=9)
    fig.tight_layout()
    fig.savefig(salida, dpi=80)
    plt.close(fig)
    print("Figura:", salida)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raiz", default="data/raw")
    p.add_argument("--inventario", default="results/inventario")
    p.add_argument("--vectores", default="data/processed")
    p.add_argument("--datasets", nargs="*", default=["saposoa", "rocole", "uganda"])
    p.add_argument("--umbral_min", type=float, default=0.90)
    p.add_argument("--semilla", type=int, default=42)
    args = p.parse_args()
    os.makedirs(args.vectores, exist_ok=True)
    rng = np.random.default_rng(args.semilla)

    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    modelo = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    modelo.fc = torch.nn.Identity()
    modelo.eval().to(dispositivo)
    print("Dispositivo:", dispositivo)

    inv = pd.read_csv(os.path.join(args.inventario, "inventario_imagenes.csv"))
    inv = inv[~inv["corrupta"].astype(bool)].drop_duplicates("md5")  # sin dañadas ni copias exactas

    todos = []
    for ds in args.datasets:
        sub = inv[inv["dataset"] == ds].reset_index(drop=True)
        print(f"\n{ds}: {len(sub)} imágenes distintas")
        v = caracteristicas([os.path.join(args.raiz, r) for r in sub["ruta"]], modelo, dispositivo)
        np.save(os.path.join(args.vectores, f"vectores_{ds}.npy"), v)

        i, j, s = pares_similares(v, args.umbral_min)
        pares = pd.DataFrame({"dataset": ds,
                              "ruta_a": sub["ruta"].values[i], "ruta_b": sub["ruta"].values[j],
                              "clase_a": sub["carpeta"].values[i], "clase_b": sub["carpeta"].values[j],
                              "similitud": s})
        todos.append(pares)

        print(f"  pares por tramo de similitud (misma clase / distinta clase):")
        for lo, hi in TRAMOS:
            t = pares[(pares["similitud"] >= lo) & (pares["similitud"] < hi)]
            misma = (t["clase_a"] == t["clase_b"]).sum()
            print(f"    {lo:.2f}-{min(hi, 1):.2f}: {len(t):6d}  ({misma} / {len(t) - misma})")

        muestras = []
        for lo, hi in TRAMOS:
            t = pares[(pares["similitud"] >= lo) & (pares["similitud"] < hi)]
            if len(t):
                muestras.append(t.iloc[rng.choice(len(t), size=min(3, len(t)), replace=False)])
        if muestras:
            dibujar(pd.concat(muestras), args.raiz,
                    os.path.join(args.inventario, f"{ds}_pares_cnn.png"),
                    f"{ds}: 3 pares al azar por tramo de similitud")

    salida = os.path.join(args.inventario, "pares_similares_cnn.csv")
    pd.concat(todos).to_csv(salida, index=False)
    print("\nPares guardados en:", salida)


if __name__ == "__main__":
    main()
