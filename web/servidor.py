"""Backend de la demo web del clasificador de hojas.

Uso:
    uvicorn web.servidor:app --reload --port 8000
    # Abrir http://localhost:8000/

Requiere:
    - Pesos en data/processed/modelos/{nombre}_s{semilla}.pt (los que guarda src/entrenar.py)
    - Configs en configs/{nombre}.yaml
    - pip install fastapi "uvicorn[standard]" python-multipart
    - El resto de dependencias ya están en requirements.txt
"""
from __future__ import annotations

import base64
import io
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import transforms

# Reutilizamos exactamente tu lógica de src/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from entrenar import crear_modelo, CLASES            # noqa: E402
from gradcam import GradCAM                          # noqa: E402

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
PESOS_DIR = ROOT / "data" / "processed" / "modelos"
CONFIGS_DIR = ROOT / "configs"
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

MODELOS_DISPONIBLES = [
    "resnet50_basico", "resnet50_fondo", "resnet50_fuerte",
    "vit_b_16_basico", "vit_b_16_fondo", "vit_b_16_fuerte",
]
MODELO_DEFAULT = "vit_b_16_fondo"
SEMILLA_DEFAULT = 0

# Transformación de evaluación (asumimos el estándar ImageNet que ya usas en aumentos.py)
t_eval = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# ---------------------------------------------------------------------------
# Carga del modelo (cacheado por {nombre, semilla})
# ---------------------------------------------------------------------------
@lru_cache(maxsize=16)
def cargar_modelo(nombre: str, semilla: int = 0):
    if nombre not in MODELOS_DISPONIBLES:
        raise HTTPException(404, f"Modelo no disponible: {nombre}")
    cfg_path = CONFIGS_DIR / f"{nombre}.yaml"
    if not cfg_path.exists():
        raise HTTPException(500, f"Falta configuración: {cfg_path}")
    cfg = yaml.safe_load(open(cfg_path))

    pesos_path = PESOS_DIR / f"{nombre}_s{semilla}.pt"
    if not pesos_path.exists():
        raise HTTPException(500, f"Faltan pesos: {pesos_path}. Ejecuta src/entrenar.py primero.")

    modelo = crear_modelo(cfg["modelo"], preentrenado=False)
    state = torch.load(pesos_path, map_location="cpu")
    modelo.load_state_dict(state)
    modelo.eval()
    return modelo, cfg["modelo"]

# Cache de Grad-CAM por modelo (evita acumular hooks en el modelo cacheado)
_gradcam_cache: dict[tuple[str, int], GradCAM] = {}

def obtener_gradcam(nombre: str, semilla: int = 0) -> GradCAM:
    key = (nombre, semilla)
    if key not in _gradcam_cache:
        modelo, tipo = cargar_modelo(nombre, semilla)
        _gradcam_cache[key] = GradCAM(modelo, tipo)
    return _gradcam_cache[key]

# ---------------------------------------------------------------------------
# Preprocesamiento: replica recorte() de src/gradcam.py
# ---------------------------------------------------------------------------
def recorte_central(im: Image.Image, tam: int = IMG_SIZE) -> Image.Image:
    escala = tam / min(im.size)
    if escala < 1:
        im = im.resize((round(im.width * escala), round(im.height * escala)), Image.LANCZOS)
    izq, arr = (im.width - tam) // 2, (im.height - tam) // 2
    return im.crop((izq, arr, izq + tam, arr + tam))

def preprocesar(im: Image.Image) -> torch.Tensor:
    im = recorte_central(im.convert("RGB"), IMG_SIZE)
    return t_eval(im).unsqueeze(0)

# ---------------------------------------------------------------------------
# Overlay de Grad-CAM sobre la imagen (devuelve data:image/png;base64,...)
# ---------------------------------------------------------------------------
def overlay_gradcam(im_pil: Image.Image, cam: np.ndarray) -> str:
    import matplotlib.cm as cm
    w, h = im_pil.size
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    cam_np = np.asarray(cam_img) / 255.0
    color = cm.jet(cam_np)[:, :, :3]
    base = np.asarray(im_pil, dtype=np.float32) / 255.0
    mezcla = np.clip(0.55 * base + 0.45 * color, 0, 1)
    buf = io.BytesIO()
    Image.fromarray((mezcla * 255).astype(np.uint8)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
app = FastAPI(title="Clasificador de hojas — MIA-203")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # en producción restringir
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True, "modelos": MODELOS_DISPONIBLES, "clases": CLASES}

@app.post("/predict")
async def predict(
    archivo: UploadFile = File(...),
    modelo: str = Form(MODELO_DEFAULT),
    semilla: int = Form(SEMILLA_DEFAULT),
    gradcam: bool = Form(False),
):
    if not archivo.content_type or not archivo.content_type.startswith("image/"):
        raise HTTPException(400, "El archivo debe ser una imagen.")
    contenido = await archivo.read()
    if len(contenido) > 10 * 1024 * 1024:
        raise HTTPException(413, "Imagen mayor a 10 MB.")

    try:
        im_pil = Image.open(io.BytesIO(contenido)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"No se pudo leer la imagen: {e}")

    net, tipo = cargar_modelo(modelo, semilla)
    dispositivo = next(net.parameters()).device
    x = preprocesar(im_pil).to(dispositivo)

    t0 = time.perf_counter()
    with torch.set_grad_enabled(False):
        logits = net(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    tiempo_ms = int((time.perf_counter() - t0) * 1000)

    # Predicción 3 clases (Saposoa)
    idx = int(probs.argmax())
    pred = CLASES[idx]

    # Vista binaria RoCoLe/Uganda: p_roya / (p_roya + p_sana)
    i_roya = CLASES.index("roya")
    i_sana = CLASES.index("sana")
    p_roya, p_sana = float(probs[i_roya]), float(probs[i_sana])
    bin_roya = p_roya / (p_roya + p_sana + 1e-12)

    respuesta = {
        "prediccion": pred,
        "probabilidades": {c: float(probs[i]) for i, c in enumerate(CLASES)},
        "binario": {"roya": bin_roya, "sana": 1.0 - bin_roya},
        "modelo": modelo,
        "semilla": semilla,
        "tiempo_ms": tiempo_ms,
    }

    if gradcam:
        try:
            gc = obtener_gradcam(modelo, semilla)
            with torch.set_grad_enabled(True):
                cam, _ = gc(x.clone().detach())
            im_recortada = recorte_central(im_pil, IMG_SIZE)
            respuesta["gradcam_base64"] = overlay_gradcam(im_recortada, cam[0])
        except Exception as e:
            respuesta["gradcam_error"] = str(e)

    return respuesta

# Servir los 3 archivos estáticos desde web/
app.mount("/", StaticFiles(directory=Path(__file__).parent, html=True), name="web")