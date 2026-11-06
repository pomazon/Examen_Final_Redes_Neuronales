"""Descarga los tres conjuntos de hojas de café desde Mendeley Data y los descomprime.

Este paso es el punto de partida del proyecto: si la descarga y la estructura de carpetas
quedan correctas, el resto del flujo puede ejecutarse de forma reproducible.

Uso:
    python src/descargar_datos.py --destino data/raw
    python src/descargar_datos.py --destino data/raw --solo uganda
"""
import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile

# Versiones fijas. Si cambian, cambian los resultados.
DATASETS = {
    "saposoa": "https://data.mendeley.com/public-api/zip/mfpxg4y65r/download/2",
    "rocole": "https://data.mendeley.com/public-api/zip/c5yvn32dzg/download/2",
    "uganda": "https://data.mendeley.com/public-api/zip/k36wnd6knb/download/1",
}


def descargar(url, ruta):
    """Descarga un archivo de datos y muestra progreso en MB para seguir la transferencia."""
    pedido = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(pedido) as r, open(ruta, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        leido = 0
        while True:
            bloque = r.read(1 << 20)
            if not bloque:
                break
            f.write(bloque)
            leido += len(bloque)
            if total:
                print(f"\r  {leido / 1e6:.0f} / {total / 1e6:.0f} MB", end="")
    print()


def extraer_rar(carpeta):
    # Saposoa trae un .rar dentro del zip, por eso se hace una extracción adicional.
    rars = [os.path.join(d, a) for d, _, fs in os.walk(carpeta)
            for a in fs if a.lower().endswith(".rar")]
    if not rars:
        return
    if shutil.which("unrar") is None:
        sys.exit("Falta 'unrar'. En Colab: !apt-get -qq install -y unrar")
    for rar in rars:
        subprocess.run(["unrar", "x", "-inul", "-o+", rar, os.path.dirname(rar) + "/"],
                       check=True)
        os.remove(rar)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--destino", default="data/raw")
    p.add_argument("--solo", nargs="*", choices=list(DATASETS))
    args = p.parse_args()

    for nombre in args.solo or DATASETS:
        carpeta = os.path.join(args.destino, nombre)
        if os.path.isdir(carpeta) and os.listdir(carpeta):
            print(f"{nombre}: ya está descargado, se omite")
            continue
        os.makedirs(carpeta, exist_ok=True)
        ruta_zip = carpeta + ".zip"

        print(f"{nombre}: descargando")
        descargar(DATASETS[nombre], ruta_zip)
        if not zipfile.is_zipfile(ruta_zip):
            sys.exit(f"{nombre}: lo descargado no es un zip. Revisar la URL.")

        with zipfile.ZipFile(ruta_zip) as z:
            z.extractall(carpeta)
        os.remove(ruta_zip)       # se borra para no ocupar el doble de espacio
        extraer_rar(carpeta)
        print(f"{nombre}: listo")


if __name__ == "__main__":
    main()
