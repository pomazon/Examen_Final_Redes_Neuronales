# Clasificación de enfermedades en hojas de café

Proyecto final del curso MIA-203 Redes Neuronales y Aprendizaje Profundo (UNI, FIIS).

Entrenamos modelos con fotos de hojas de café de Saposoa (San Martín, Perú) y medimos qué tan bien reconocen la roya en fotos tomadas en otros países (Ecuador y Uganda).

## Estructura

```
data/        fuentes, licencias y descarga de los datos
src/         código del proyecto
notebooks/   notebook de Colab que ejecuta el proyecto
configs/     hiperparámetros de cada experimento
results/     tablas, figuras y logs del artículo
```

## Cómo reproducir

En una máquina con Python 3 y `unrar` instalado:

```
git clone https://github.com/pomazon/Examen_Final_Redes_Neuronales.git
cd Examen_Final_Redes_Neuronales
pip install -r requirements.txt
python src/descargar_datos.py --destino data/raw
python src/inventario.py --raiz data/raw --salida results/inventario
python src/ver_muestras.py --inventario results/inventario/inventario.csv --raiz data/raw
python src/duplicados.py --inventario results/inventario/inventario.csv --raiz data/raw
python src/particiones.py --inventario results/inventario/inventario.csv --pares results/duplicados/pares.csv
```

## Datos

Tres conjuntos públicos de Mendeley Data, con licencia CC BY 4.0. Detalle y citas en [data/README.md](data/README.md).

## Resultados

Pendiente.

## Equipo

Pendiente.
