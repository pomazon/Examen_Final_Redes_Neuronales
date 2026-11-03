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
python src/preparar_imagenes.py --splits data/splits.csv --raiz data/raw --destino data/processed/imagenes
python src/linea_base.py --splits data/splits.csv --imagenes data/processed/imagenes
python src/ver_mascaras.py --splits data/splits.csv --imagenes data/processed/imagenes
python src/entrenar.py --config configs/resnet50_basico.yaml --semilla 0   # cada config y semilla 0, 1, 2
python src/resumen.py
python src/gradcam.py --modelos resnet50_basico resnet50_fuerte resnet50_fondo vit_b_16_basico vit_b_16_fuerte vit_b_16_fondo
python src/errores.py --modelos resnet50_basico resnet50_fuerte resnet50_fondo vit_b_16_basico vit_b_16_fuerte vit_b_16_fondo
python src/costo.py
python src/ejemplos.py
python src/figuras_articulo.py
python src/prueba_fondo.py --modelos resnet50_basico resnet50_fuerte resnet50_fondo vit_b_16_basico vit_b_16_fuerte vit_b_16_fondo
```

## Datos

Tres conjuntos públicos de Mendeley Data, con licencia CC BY 4.0. Detalle y citas en [data/README.md](data/README.md).

## Resultados

Entrenamiento con Saposoa. Prueba externa en RoCoLe (Ecuador) y Uganda, solo sana contra roya. Media ± desviación en 3 semillas.

| Modelo | Aumento | Saposoa test F1 | RoCoLe AUC | Uganda AUC | Minutos por corrida (T4) |
|---|---|---|---|---|---|
| Mejor línea base | — | | 0,750 | 0,767 | |
| ResNet50 | básico | ≥ 0,98 | 0,831 ± 0,013 | 0,743 ± 0,142 | 1,6 |
| ResNet50 | fuerte | ≥ 0,98 | 0,858 ± 0,006 | 0,779 ± 0,063 | 2,6 |
| ResNet50 | fuerte + fondo | ≥ 0,98 | 0,858 ± 0,033 | 0,728 ± 0,064 | 4,8 |
| ViT-B/16 | básico | ≥ 0,98 | 0,909 ± 0,011 | 0,717 ± 0,054 | 3,6 |
| ViT-B/16 | fuerte | ≥ 0,98 | 0,920 ± 0,009 | 0,750 ± 0,052 | 4,0 |
| ViT-B/16 | fuerte + fondo | ≥ 0,98 | 0,920 ± 0,009 | 0,777 ± 0,060 | 5,8 |

Tabla completa en `results/resumen.csv`. Costo total: 1,12 horas de GPU T4 (`results/costo.csv`).

- Dentro de Saposoa el problema es casi trivial; la dificultad está fuera de él.
- ViT-B/16 generaliza mejor que ResNet50 a RoCoLe. Uganda es difícil para todos los modelos.
- Sin reemplazo de fondo, los modelos aprenden a reconocer la clase por el fondo de Saposoa (`results/prueba_fondo/`). El reemplazo de fondo elimina ese atajo, pero no mejora la prueba externa: la caída se debe a que las hojas de campo se ven distintas, no al fondo.
- En fotos de campo los modelos detectan casi toda la roya pero confunden muchas hojas sanas con enfermas (`results/errores/`).

## Pesos y demostración

Pesos de los 6 modelos de la semilla 0 (Google Drive, público): https://drive.google.com/drive/folders/1fIOa7MGDJUeVh-MrOiGy6_5snuy5TeF6?usp=sharing

Demostración en Colab con ViT-B/16 (fuerte + fondo): [notebooks/demo.ipynb](notebooks/demo.ipynb)

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pomazon/Examen_Final_Redes_Neuronales/blob/main/notebooks/demo.ipynb)

## Equipo

Grupo 2, sección A:

- Alexander André Chávez Cabana
- Alex Reynaldo Pomazón Solórzano
- Andhersson Manuel Salazar Fernández
- Christian André Urviola García
