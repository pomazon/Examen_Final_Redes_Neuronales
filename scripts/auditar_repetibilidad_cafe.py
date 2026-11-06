"""Audita repetibilidad local del entrenamiento del proyecto de roya del cafe.

Ejecutar desde la raiz del repositorio, con el entorno con CUDA activo:
    python auditar_repetibilidad_cafe.py

No cambia archivos del repositorio ni los resultados existentes. Ejecuta dos
entrenamientos en carpetas nuevas, con modo determinista estricto, y compara
salidas numericas. Si CUDA informa que una operacion no es determinista,
interrumpe en vez de ocultar el problema.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys


def argumentos():
    p = argparse.ArgumentParser(description="Dos entrenamientos deterministas y comparacion de resultados")
    p.add_argument("--config", default=r"C:\datos_cafe\vit_b_16_fondo_rtx5050.yaml")
    p.add_argument("--splits", default=r"data\splits.csv")
    p.add_argument("--imagenes", default=r"C:\datos_cafe\imagenes")
    p.add_argument("--destino", default=r"C:\datos_cafe\auditoria_determinista")
    p.add_argument("--semilla", type=int, default=0)
    p.add_argument("--interno", choices=["A", "B"], help=argparse.SUPPRESS)
    return p.parse_args()


def rutas(args, letra):
    base = Path(args.destino) / f"corrida_{letra}"
    config_nombre = Path(args.config).stem
    resultados = base / "resultados" / config_nombre / f"semilla_{args.semilla}"
    modelos = base / "modelos"
    return base, resultados, modelos


def ejecutar_interno(args):
    # El proceso padre fija CUBLAS_WORKSPACE_CONFIG ANTES de cargar PyTorch.
    import torch
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no disponible: activa el entorno cct20_gpu antes de ejecutar.")
    sys.path.insert(0, str(Path.cwd() / "src"))
    import entrenar

    base, _, modelos = rutas(args, args.interno)
    print(f"\n=== CORRIDA DETERMINISTA {args.interno} ===", flush=True)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    sys.argv = ["entrenar.py", "--config", args.config, "--semilla", str(args.semilla),
                "--splits", args.splits, "--imagenes", args.imagenes,
                "--salida", str(base / "resultados"), "--modelos", str(modelos)]
    entrenar.main()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for trozo in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(trozo)
    return h.hexdigest()


def diferencias(a, b, ruta="", ignorar=None):
    ignorar = ignorar or {"minutos_entrenamiento"}
    if isinstance(a, dict) and isinstance(b, dict):
        for clave in sorted(set(a) | set(b)):
            if clave not in ignorar:
                if clave not in a or clave not in b:
                    yield f"{ruta}.{clave}: clave ausente"
                else:
                    yield from diferencias(a[clave], b[clave], f"{ruta}.{clave}", ignorar)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        if not math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-8):
            yield f"{ruta}: A={a} B={b}"
    elif a != b:
        yield f"{ruta}: A={a!r} B={b!r}"


def main():
    args = argumentos()
    if args.interno:
        ejecutar_interno(args)
        return
    if not Path("src/entrenar.py").is_file():
        sys.exit("Abre Anaconda Prompt en la raiz del repositorio, donde existe src\\entrenar.py.")
    for path, etiqueta in [(args.config, "configuracion"), (args.splits, "particiones"), (args.imagenes, "imagenes")]:
        if not Path(path).exists():
            sys.exit(f"No se encontro {etiqueta}: {path}")
    for letra in ("A", "B"):
        base, _, _ = rutas(args, letra)
        if base.exists():
            sys.exit(f"La carpeta {base} YA EXISTE. No la sobrescribire. Cambia --destino o conserva/renombra esa carpeta.")

    entorno = os.environ.copy()
    entorno["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    entorno["PYTHONHASHSEED"] = str(args.semilla)
    for letra in ("A", "B"):
        base, _, _ = rutas(args, letra)
        print(f"\nIniciando corrida {letra}, salida separada: {base}", flush=True)
        cmd = [sys.executable, str(Path(__file__).resolve()), "--interno", letra,
               "--config", args.config, "--splits", args.splits,
               "--imagenes", args.imagenes, "--destino", args.destino,
               "--semilla", str(args.semilla)]
        try:
            subprocess.run(cmd, env=entorno, check=True)
        except subprocess.CalledProcessError as e:
            sys.exit(f"Fallo la corrida {letra} (codigo {e.returncode}). No se alteraron los respaldos ni GitHub. Comparte el error completo.")

    _, rA, mA = rutas(args, "A")
    _, rB, mB = rutas(args, "B")
    print("\n" + "=" * 68)
    print("AUDITORIA DE REPETIBILIDAD (MISMA GPU, CONFIGURACION Y SEMILLA)")
    a = json.loads((rA / "metricas.json").read_text(encoding="utf-8"))
    b = json.loads((rB / "metricas.json").read_text(encoding="utf-8"))
    print("Mejor epoca:", a.get("mejor_epoca"), "vs", b.get("mejor_epoca"))
    difs = list(diferencias(a, b))
    print("Metricas numericas (excluyendo tiempo):", "COINCIDEN" if not difs else "DIFIEREN")
    for d in difs[:15]:
        print("  ", d)
    if len(difs) > 15:
        print(f"  ... y {len(difs)-15} diferencias adicionales")
    for archivo in ("historial.csv", "predicciones.csv"):
        iguales = sha256(rA / archivo) == sha256(rB / archivo)
        print(archivo, "IDENTICO" if iguales else "DIFERENTE")
    nom = Path(args.config).stem
    pesoA = mA / f"{nom}_s{args.semilla}.pt"
    pesoB = mB / f"{nom}_s{args.semilla}.pt"
    if pesoA.exists() and pesoB.exists():
        print("Archivo de pesos .pt:", "IDENTICO EN BYTES" if sha256(pesoA) == sha256(pesoB) else "DIFERENTE EN BYTES")
    print("Datos conservados en:", args.destino)
    if not difs and sha256(rA / "predicciones.csv") == sha256(rB / "predicciones.csv"):
        print("CONCLUSION: Repetibilidad observada bajo este modo determinista, en este equipo.")
    else:
        print("CONCLUSION: Persisten diferencias; habra que revisar el origen sin asumir su causa.")


if __name__ == "__main__":
    main()
