"""Ajusta (fine-tuning) una red preentrenada con Saposoa y la evalúa en todos los conjuntos.

Se entrena con Saposoa train, se elige la mejor época por F1 macro en Saposoa val y
con esa época se evalúa Saposoa test, RoCoLe y Uganda.

Uso:
    python src/entrenar.py --config configs/resnet50_basico.yaml --semilla 0
"""
import argparse
import copy
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from aumentos import FondoAleatorio, mascara_hoja, transformacion
from metricas import CLASES, evaluar


class Hojas(Dataset):
    def __init__(self, df, carpeta, transform, fondo=None):
        self.rutas = [os.path.join(carpeta, r) for r in df["ruta"]]
        self.y = [CLASES.index(c) for c in df["clase"]]
        self.transform, self.fondo = transform, fondo
        self.mascaras = {}  # cada máscara se calcula una vez y se reutiliza en las épocas siguientes

    def __len__(self):
        return len(self.rutas)

    def __getitem__(self, i):
        im = Image.open(self.rutas[i]).convert("RGB")
        if self.fondo is not None:
            if i not in self.mascaras:
                self.mascaras[i] = mascara_hoja(im)
            im = self.fondo(im, self.mascaras[i])
        return self.transform(im), self.y[i]


def crear_modelo(nombre):
    if nombre == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = torch.nn.Linear(m.fc.in_features, len(CLASES))
    elif nombre == "vit_b_16":
        m = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        m.heads.head = torch.nn.Linear(m.heads.head.in_features, len(CLASES))
    else:
        raise ValueError(nombre)
    return m


def fijar_semilla(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


@torch.no_grad()
def probabilidades(modelo, cargador, dispositivo):
    modelo.eval()
    salida = []
    for x, _ in cargador:
        with torch.autocast(dispositivo, enabled=dispositivo == "cuda"):
            salida.append(torch.softmax(modelo(x.to(dispositivo)).float(), 1).cpu())
    return torch.cat(salida).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--semilla", type=int, default=0)
    p.add_argument("--splits", default="data/splits.csv")
    p.add_argument("--imagenes", default="data/processed/imagenes")
    p.add_argument("--salida", default="results/entrenamiento")
    p.add_argument("--modelos", default="data/processed/modelos")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    nombre = os.path.splitext(os.path.basename(args.config))[0]
    carpeta = os.path.join(args.salida, nombre, f"semilla_{args.semilla}")
    os.makedirs(carpeta, exist_ok=True)
    os.makedirs(args.modelos, exist_ok=True)
    fijar_semilla(args.semilla)
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(args.splits)
    conjunto = df["dataset"].where(df["dataset"] != "saposoa", "saposoa_" + df["particion"])
    partes = {c: df[conjunto == c] for c in ["saposoa_train", "saposoa_val", "saposoa_test", "rocole", "uganda"]}

    fondo = FondoAleatorio(cfg.get("prob_fondo", 0.8), args.semilla) if cfg["aumento"] == "fondo" else None
    g = torch.Generator().manual_seed(args.semilla)
    tr = DataLoader(Hojas(partes["saposoa_train"], args.imagenes, transformacion(cfg["aumento"]), fondo),
                    batch_size=cfg["lote"], shuffle=True, num_workers=2, generator=g, drop_last=True,
                    persistent_workers=True)
    ev = {k: DataLoader(Hojas(v, args.imagenes, transformacion("evaluacion")), batch_size=64, num_workers=2)
          for k, v in partes.items() if k != "saposoa_train"}

    modelo = crear_modelo(cfg["modelo"]).to(dispositivo)
    opt = torch.optim.AdamW(modelo.parameters(), lr=cfg["lr"], weight_decay=cfg["peso_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epocas"])
    escalador = torch.amp.GradScaler(enabled=dispositivo == "cuda")
    perdida_fn = torch.nn.CrossEntropyLoss()

    inicio = time.time()
    historial, mejor = [], (np.inf, None, None)
    y_val = np.array([CLASES.index(c) for c in partes["saposoa_val"]["clase"]])
    for epoca in range(1, cfg["epocas"] + 1):
        modelo.train()
        total, n = 0.0, 0
        for x, y in tr:
            x, y = x.to(dispositivo), y.to(dispositivo)
            opt.zero_grad()
            with torch.autocast(dispositivo, enabled=dispositivo == "cuda"):
                perdida = perdida_fn(modelo(x), y)
            escalador.scale(perdida).backward()
            escalador.step(opt)
            escalador.update()
            total += perdida.item() * len(y)
            n += len(y)
        sched.step()
        pr_val = probabilidades(modelo, ev["saposoa_val"], dispositivo)
        f1_val = evaluar(partes["saposoa_val"]["clase"], pr_val, "saposoa")["f1_macro"]
        perdida_val = float(-np.log(pr_val[np.arange(len(y_val)), y_val] + 1e-12).mean())
        historial.append({"epoca": epoca, "perdida_train": total / n, "perdida_val": perdida_val, "f1_val": f1_val})
        print(f"  época {epoca:2d}  pérdida train {total / n:.4f}  pérdida val {perdida_val:.4f}  F1 val {f1_val:.3f}")
        # Se elige la época con menor pérdida en validación: el F1 llega a 1,0 en pocas
        # épocas y deja de distinguir entre ellas; la pérdida sigue siendo informativa.
        if perdida_val < mejor[0]:
            mejor = (perdida_val, epoca, copy.deepcopy(modelo.state_dict()))
    minutos = (time.time() - inicio) / 60

    modelo.load_state_dict(mejor[2])
    torch.save(mejor[2], os.path.join(args.modelos, f"{nombre}_s{args.semilla}.pt"))

    resultados = {"config": nombre, **cfg, "semilla": args.semilla, "mejor_epoca": mejor[1],
                  "minutos_entrenamiento": round(minutos, 2),
                  "gpu": torch.cuda.get_device_name(0) if dispositivo == "cuda" else "cpu"}
    predicciones = []
    for k, ds in [("saposoa_val", "saposoa"), ("saposoa_test", "saposoa"), ("rocole", "rocole"), ("uganda", "uganda")]:
        pr = probabilidades(modelo, ev[k], dispositivo)
        resultados[k] = evaluar(partes[k]["clase"], pr, ds)
        if k != "saposoa_val":
            predicciones.append(partes[k][["dataset", "ruta", "clase"]].assign(
                evaluacion=k, **{f"p_{c}": pr[:, i] for i, c in enumerate(CLASES)}))

    json.dump(resultados, open(os.path.join(carpeta, "metricas.json"), "w"), indent=2, ensure_ascii=False)
    pd.DataFrame(historial).to_csv(os.path.join(carpeta, "historial.csv"), index=False)
    pd.concat(predicciones).to_csv(os.path.join(carpeta, "predicciones.csv"), index=False)

    print(f"\n{nombre}, semilla {args.semilla}: mejor época {mejor[1]}, {minutos:.1f} min en {resultados['gpu']}")
    for k in ["saposoa_test", "rocole", "uganda"]:
        r = resultados[k]
        extra = f"  AUC {r['auc']:.3f}  pred. ojo de gallo {r['pred_ojo_gallo']:.1%}" if "auc" in r else ""
        print(f"  {k:13s} F1 {r['f1_macro']:.3f}{extra}")


if __name__ == "__main__":
    main()
