# Clasificación de enfermedades en hojas de café

Proyecto final del curso MIA-203 Redes Neuronales y Aprendizaje Profundo (UNI, FIIS).

Entrenamos modelos con fotos de hojas de café de Saposoa (San Martín, Perú) y medimos qué tan bien reconocen la roya en fotos tomadas en otros países (Ecuador y Uganda).

## Estructura

```
data/        README con fuentes, licencias y descarga de los datos
src/         código: descarga, inventario y, más adelante, entrenamiento y evaluación
notebooks/   notebook para ejecutar el proyecto en Colab
configs/     hiperparámetros de cada experimento
results/     tablas, figuras y logs del artículo
```

## Cómo reproducir

En Google Colab, abrir `notebooks/Examen_Final_Redes_Neuronales.ipynb` y ejecutar las celdas en orden.

En una máquina propia:

```
git clone https://github.com/pomazon/Examen_Final_Redes_Neuronales.git
cd Examen_Final_Redes_Neuronales
pip install -r requirements.txt
python src/descargar_datos.py --destino data/raw
python src/inventario.py --raiz data/raw --salida results/inventario
```

Para descomprimir el conjunto de Saposoa hace falta `unrar` instalado.

## Datos

Tres conjuntos públicos de Mendeley Data, con licencia CC BY 4.0. Detalle y citas en [data/README.md](data/README.md).

## Resultados

Pendiente.

## Equipo

Pendiente.
