"""Figuras del artículo que se arman a partir de los resultados.

Este archivo genera los PDFs finales usados en el artículo. Las figuras se construyen
con información ya almacenada en `data/ejemplos/`, `results/entrenamiento/` y
`results/gradcam/`, por lo que no requiere re-entrenar modelos.

    ejemplos.pdf  una foto de prueba de cada conjunto y clase (data/ejemplos/)
    fondo.pdf     máscara de la hoja, fondo aleatorio y versiones de la prueba de fondo
    gradcam.pdf   recorte de results/gradcam/gradcam.png (5 fotos, 3 modelos)
    curvas.pdf    pérdida por época (promedio de 3 semillas)
    duplicados.pdf  tres pares de fotos casi iguales
    errores.pdf   errores de ViT-B/16 fuerte + fondo en fotos de campo

El diagrama de la arquitectura está en results/articulo/arquitectura.drawio
(se abre y exporta con draw.io).

Uso:
    python src/figuras_articulo.py
"""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
R = "."
sys.path.insert(0, "src")
import os
SAL = "results/articulo"
os.makedirs(SAL, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 7})

# Fig. de ejemplos de datos: una muestra representativa por conjunto y clase.
orden = [("saposoa","sana","Saposoa · sana"),("saposoa","roya","Saposoa · roya"),("saposoa","ojo_gallo","Saposoa · ojo de gallo"),
         ("rocole","sana","RoCoLe · sana"),("rocole","roya","RoCoLe · roya"),("uganda","sana","Uganda · sana"),("uganda","roya","Uganda · roya")]
fig, ax = plt.subplots(1, 7, figsize=(7.1, 1.3))
for a,(d,c,t) in zip(ax, orden):
    if (d, c) == ("uganda", "sana"):
        # La foto de data/ejemplos es un acercamiento donde no se distingue la hoja;
        # se usa otra foto de Uganda sana, tomada de la figura de Grad-CAM.
        g_ = Image.open(f"{R}/results/gradcam/gradcam.png").convert("RGB")
        rh_, cw_ = g_.size[1] / 11, g_.size[0] / 7
        im = g_.crop((int(0.07 * cw_), int(10 * rh_ + 0.16 * rh_), int(0.93 * cw_), int(11 * rh_ - 0.02 * rh_)))
    else:
        im = Image.open(f"{R}/data/ejemplos/{d}_{c}.jpg").convert("RGB")
    w,h = im.size; s=min(w,h); im = im.crop(((w-s)//2,(h-s)//2,(w-s)//2+s,(h-s)//2+s))
    a.imshow(im); a.set_title(t, fontsize=6.5); a.axis("off")
fig.tight_layout(pad=0.4); fig.savefig(f"{SAL}/ejemplos.pdf", dpi=200, bbox_inches="tight", pad_inches=0.02); plt.close(fig)

# Fig. de versiones de fondo: muestra cómo la máscara y el reemplazo aleatorio del fondo
# pueden revelar si el modelo se basó en el contexto y no en la hoja misma.
from aumentos import mascara_hoja, FondoAleatorio
from scipy import ndimage
im = Image.open(f"{R}/data/ejemplos/saposoa_roya.jpg").convert("RGB")
m = mascara_hoja(im); a = np.array(im, float)
f = FondoAleatorio(1.0, 3)
cands = [np.array(f(im, m), float) for _ in range(40)]
fondo_std = [c[~m].std() for c in cands]
alea = [Image.fromarray(cands[int(np.argmin(fondo_std))].astype(np.uint8)), Image.fromarray(cands[int(np.argmax(fondo_std))].astype(np.uint8))]
hoja = ndimage.binary_dilation(m, iterations=6); cf = np.median(a[~hoja], axis=0)
gris = Image.fromarray(np.where(m[...,None], a, 128.).astype(np.uint8))
solo = Image.fromarray(np.where(hoja[...,None], cf, a).astype(np.uint8))
paneles = [(im,"(a) original"),(Image.fromarray((m*255).astype(np.uint8)),"(b) máscara"),(alea[0],"(c) fondo aleatorio"),
           (alea[1],"(d) fondo aleatorio"),(gris,"(e) fondo gris"),(solo,"(f) solo fondo")]
fig, ax = plt.subplots(2, 3, figsize=(3.45, 2.25))
for a_,(p,t) in zip(ax.ravel(), paneles):
    a_.imshow(p, cmap="gray"); a_.set_title(t, fontsize=6.5); a_.axis("off")
fig.tight_layout(pad=0.2); fig.savefig(f"{SAL}/fondo.pdf", dpi=200); plt.close(fig)

# Fig. de Grad-CAM recortado: se aprovecha el collage ya generado para construir una
# vista compacta de varias imágenes y modelos en una sola figura.
g = np.array(Image.open(f"{R}/results/gradcam/gradcam.png").convert("RGB"))
H, W = g.shape[:2]; rh, cw = H/11, W/7
filas = [(0,"Saposoa\nojo de gallo"),(1,"Saposoa\nroya"),(2,"Saposoa\nsana"),(4,"RoCoLe\nroya"),(8,"Uganda\nroya")]
cols = [(0,"Imagen"),(1,"ResNet50\nbásico"),(3,"ResNet50\nfuerte + fondo"),(6,"ViT-B/16\nfuerte + fondo")]
fig, ax = plt.subplots(len(filas), len(cols), figsize=(3.45, 4.35))
for i,(r,rt) in enumerate(filas):
    for j,(c,ct) in enumerate(cols):
        y0=int(r*rh+0.16*rh); y1=int((r+1)*rh-0.02*rh); x0=int(c*cw+0.07*cw); x1=int((c+1)*cw-0.07*cw)
        ax[i,j].imshow(g[y0:y1,x0:x1]); ax[i,j].set_xticks([]); ax[i,j].set_yticks([])
        for s in ax[i,j].spines.values(): s.set_visible(False)
        if i==0: ax[i,j].set_title(ct, fontsize=6.5)
        if j==0: ax[i,j].set_ylabel(rt, fontsize=6.5)
fig.tight_layout(pad=0.15); fig.savefig(f"{SAL}/gradcam.pdf", dpi=250); plt.close(fig)


import glob


def cuadrado(im):
    h_, w_ = im.shape[:2]; l_ = min(h_, w_)
    return im[(h_ - l_) // 2:(h_ - l_) // 2 + l_, (w_ - l_) // 2:(w_ - l_) // 2 + l_]

def recorte(a):
    """Recorta la foto dentro de un panel: filas y columnas con poco blanco."""
    m = (a < 235).any(axis=2)
    filas = np.where(m.mean(1) > 0.6)[0]
    cols = np.where(m[filas.min():filas.max() + 1].mean(0) > 0.6)[0]
    return a[filas.min():filas.max() + 1, cols.min():cols.max() + 1]

# Curvas de entrenamiento: resumen de pérdida por época para cada configuración.
nombres = {"basico": "básico", "fuerte": "fuerte", "fondo": "fuerte + fondo"}
colores = {"basico": "#4C72B0", "fuerte": "#DD8452", "fondo": "#55A868"}
fig, ejes = plt.subplots(1, 2, figsize=(3.45, 1.75), sharey=True)
for eje, (mod, tit) in zip(ejes, [("resnet50", "ResNet50"), ("vit_b_16", "ViT-B/16")]):
    for aum in ["basico", "fuerte", "fondo"]:
        h = pd.concat([pd.read_csv(f) for f in glob.glob(f"{R}/results/entrenamiento/{mod}_{aum}/semilla_*/historial.csv")])
        g = h.groupby("epoca").mean()
        eje.plot(g.index, g["perdida_val"], color=colores[aum], lw=1.2, label=nombres[aum])
        eje.plot(g.index, g["perdida_train"], color=colores[aum], lw=0.8, ls="--")
    eje.set_title(tit, fontsize=7.5); eje.set_xlabel("época"); eje.set_yscale("log")
    eje.set_xticks([1, 5, 10, 15]); eje.grid(alpha=0.25, lw=0.4)
ejes[0].set_ylabel("pérdida (entropía cruzada)")
fig.legend(*ejes[0].get_legend_handles_labels(), fontsize=6, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.55, 1.15))
fig.tight_layout(pad=0.3); fig.savefig(f"{SAL}/curvas.pdf", bbox_inches="tight", pad_inches=0.02); plt.close(fig)

# Pares casi iguales: resalta ejemplos que parecen duplicados o imágenes muy parecidas.
def par(png, fila, nfilas):
    a = np.array(Image.open(png).convert("RGB")); H, W = a.shape[:2]; rh = H / nfilas
    banda = a[int(fila * rh + 0.12 * rh):int((fila + 1) * rh)]
    return cuadrado(recorte(banda[:, int(0.1 * W):int(0.55 * W)])), cuadrado(recorte(banda[:, int(0.55 * W):]))
casos = [(f"{R}/results/duplicados/saposoa_mas_parecidos.png", 0, 10, "Saposoa, Sana 204 y 205\nmisma hoja, similitud 0.967"),
         (f"{R}/results/duplicados/uganda_pares.png", 1, 15, "Uganda, roya\nmismo recorte, similitud 0.984"),
         (f"{R}/results/duplicados/uganda_pares.png", 6, 15, "Uganda, phoma y roya\nmismo recorte, similitud 0.958")]
fig, ejes = plt.subplots(1, 6, figsize=(7.1, 1.45))
for k, (png, fila, nf, txt) in enumerate(casos):
    i1, i2 = par(png, fila, nf)
    for j, im in enumerate([i1, i2]):
        e = ejes[2 * k + j]; e.imshow(im); e.axis("off")
    ejes[2 * k].set_title(txt, fontsize=6.5, loc="left", x=0.15)
fig.tight_layout(pad=0.2, w_pad=0.3); fig.savefig(f"{SAL}/duplicados.pdf", dpi=200, bbox_inches="tight", pad_inches=0.02); plt.close(fig)

# Ejemplos de errores del mejor modelo generalizador (ViT-B/16 fuerte + fondo, semilla 0).
a = np.array(Image.open(f"{R}/results/errores/errores_vit_b_16_fondo.png").convert("RGB"))
W = a.shape[1] / 9
e = pd.read_csv(f"{R}/results/errores/errores_vit_b_16_fondo.csv")
e = e[e["semilla"] == 0].groupby(["dataset", "clase"]).head(3).reset_index(drop=True)
nom = {"uganda": "Uganda", "rocole": "RoCoLe"}
fig, ejes = plt.subplots(1, 4, figsize=(3.45, 1.25))
for eje, k in zip(ejes, [0, 3, 4, 7]):
    r = e.iloc[k]
    im = recorte(a[30:, int(k * W):int((k + 1) * W)])
    h_, w_ = im.shape[:2]; l_ = min(h_, w_)
    im = im[(h_ - l_) // 2:(h_ - l_) // 2 + l_, (w_ - l_) // 2:(w_ - l_) // 2 + l_]
    eje.imshow(im); eje.axis("off")
    p = {"ojo de g.": r.p_ojo_gallo, "roya": r.p_roya, "sana": r.p_sana}
    top = max(p, key=p.get)
    eje.set_title(f"{nom[r.dataset]}, real {r.clase}\n{top}: {p[top]:.2f}", fontsize=6)
fig.tight_layout(pad=0.2); fig.savefig(f"{SAL}/errores.pdf", dpi=200, bbox_inches="tight", pad_inches=0.02); plt.close(fig)
print("ok")
