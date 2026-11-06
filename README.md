# Clasificación de hojas de café sanas, con roya y con ojo de gallo en fotos de otros países: ResNet50 y ViT-B/16 entrenados con fotos de Saposoa (Perú)

Proyecto final del curso MIA-203 Redes Neuronales y Aprendizaje Profundo (UNI, FIIS).

Entrenamos modelos con fotos de hojas de café de Saposoa (San Martín, Perú) y medimos qué tan bien reconocen la roya en fotos tomadas en otros países (Ecuador y Uganda).

## Estructura

```
articulo/    artículo en PDF
data/        fuentes, licencias y descarga de los datos
src/         código del proyecto
notebooks/   notebook de Colab que ejecuta el proyecto y demostración
configs/     hiperparámetros de cada experimento
results/     tablas, figuras y logs del artículo
docs/        prueba de repetibilidad en una RTX 5050
scripts/     scripts de esa prueba
web/         página usada para la demostración en la exposición
```

Artículo: [articulo/Equipo2_Paper.pdf](articulo/Equipo2_Paper.pdf)

## Flujo rápido

Este repositorio está pensado para seguir un flujo reproducible y modular:

1. Descargar y revisar los datos en `data/`.
2. Generar particiones y prepararlas para entrenamiento.
3. Ejecutar una línea base para validar la dificultad del problema.
4. Entrenar los modelos configurados en `configs/`.
5. Evaluar resultados, comparar métricas y producir figuras para el artículo.

> Si solo quieres poner a andar el proyecto sin revisar cada detalle, empieza por `notebooks/Examen_Final_Redes_Neuronales.ipynb` o ejecuta los pasos en orden listados abajo.

## Cómo reproducir

### Requisitos

- Python 3.10 o más reciente (se usó 3.13 en Google Colab) y `unrar`.
- Una GPU para entrenar. Se usó una NVIDIA T4 de Colab: las 18 corridas toman 1.12 horas. Sin GPU funciona, pero mucho más lento.
- Unos 7 GB libres para los datos.

En Colab, `notebooks/Examen_Final_Redes_Neuronales.ipynb` ejecuta todo en orden.

### 1. Instalar

```
git clone https://github.com/pomazon/Examen_Final_Redes_Neuronales.git
cd Examen_Final_Redes_Neuronales
pip install -r requirements.txt
```

### 2. Datos, limpieza y particiones

```
python src/descargar_datos.py --destino data/raw
python src/inventario.py --raiz data/raw --salida results/inventario
python src/ver_muestras.py --inventario results/inventario/inventario.csv --raiz data/raw
python src/duplicados.py --inventario results/inventario/inventario.csv --raiz data/raw
python src/particiones.py --inventario results/inventario/inventario.csv --pares results/duplicados/pares.csv
python src/preparar_imagenes.py --splits data/splits.csv --raiz data/raw --destino data/processed/imagenes
```

`data/splits.csv` ya está en el repositorio: es la partición usada en el artículo (semilla 42).

### 3. Línea base

```
python src/linea_base.py --splits data/splits.csv --imagenes data/processed/imagenes
```

### 4. Entrenamiento y evaluación

Cada corrida entrena un modelo y lo evalúa en Saposoa, RoCoLe y Uganda. Guarda las métricas en `results/entrenamiento/` y los pesos en `data/processed/modelos/`.

```
python src/ver_mascaras.py --splits data/splits.csv --imagenes data/processed/imagenes
for config in resnet50_basico resnet50_fuerte resnet50_fondo vit_b_16_basico vit_b_16_fuerte vit_b_16_fondo; do
  for semilla in 0 1 2; do
    python src/entrenar.py --config configs/$config.yaml --semilla $semilla
  done
done
python src/resumen.py
python src/costo.py
```

Semillas: 0, 1 y 2 para el entrenamiento y el random forest; 42 para la partición. En la auditoría local con una RTX 5050, dos entrenamientos de ViT-B/16 con cambio de fondo y semilla 0 alcanzaron AUC de 0.936 y 0.859 en RoCoLe. Al repetir el experimento con el modo determinista activado, ambas corridas alcanzaron un AUC de 0.909 y produjeron historiales, predicciones y pesos idénticos.

### 5. Interpretación

Usa los modelos de la semilla 0. Para no reentrenar, se pueden bajar los pesos publicados:

```
gdown --folder https://drive.google.com/drive/folders/1fIOa7MGDJUeVh-MrOiGy6_5snuy5TeF6 -O data/processed/modelos
```

```
MODELOS="resnet50_basico resnet50_fuerte resnet50_fondo vit_b_16_basico vit_b_16_fuerte vit_b_16_fondo"
python src/gradcam.py --modelos $MODELOS
python src/errores.py --modelos $MODELOS
python src/prueba_fondo.py --modelos $MODELOS
```

### 6. Figuras del artículo

```
python src/ejemplos.py
python src/figuras_articulo.py
```

`figuras_articulo.py` solo necesita lo que ya está en `results/` y `data/ejemplos/`, así que funciona sin descargar los datos. El diagrama de la arquitectura está en `results/articulo/arquitectura.drawio` y se edita con draw.io.

### De cada tabla y figura del artículo a su comando

| En el artículo | Comando | Archivo |
|---|---|---|
| Tabla I (fotos por conjunto) | `particiones.py` | `results/particiones/conteo.csv` |
| Tabla II (líneas base y modelos) | `linea_base.py`, `entrenar.py` y `resumen.py` | `results/resumen.csv`, `results/linea_base/metricas.csv` |
| Tabla III (prueba de fondo y Grad-CAM) | `prueba_fondo.py` y `gradcam.py` | `results/prueba_fondo/prueba_fondo.csv`, `results/gradcam/cam_en_hoja.csv` |
| Tabla IV (clase asignada en fotos de campo) | `errores.py` | `results/errores/confusion_*.csv` |
| Fig. 1 (fotos casi iguales) | `duplicados.py` y `figuras_articulo.py` | `results/articulo/duplicados.pdf` |
| Fig. 2 (ejemplos) | `ejemplos.py` y `figuras_articulo.py` | `results/articulo/ejemplos.pdf` |
| Fig. 3 (máscara y prueba de fondo) | `figuras_articulo.py` | `results/articulo/fondo.pdf` |
| Fig. 4 (arquitectura) | draw.io | `results/articulo/arquitectura.drawio` |
| Fig. 5 (curvas de entrenamiento) | `entrenar.py` y `figuras_articulo.py` | `results/articulo/curvas.pdf` |
| Fig. 6 (Grad-CAM) | `gradcam.py` y `figuras_articulo.py` | `results/articulo/gradcam.pdf` |
| Fig. 7 (errores) | `errores.py` y `figuras_articulo.py` | `results/articulo/errores.pdf` |
| Costo de GPU | `costo.py` | `results/costo.csv` |
| Revisión de fuga (6.9 % frente a 8.2 %) | `particiones.py` | Se imprime al ejecutarlo |

## Datos

Tres conjuntos públicos de Mendeley Data, con licencia CC BY 4.0. Detalle y citas en [data/README.md](data/README.md).

## Resultados

El modelo se entrena con fotos de Saposoa y clasifica tres clases: sana, roya y ojo de gallo. En RoCoLe (Ecuador) y Uganda solo hay hojas sanas y con roya, así que ahí se mide roya frente a sana con el AUC. Media ± desviación estándar de 3 semillas. La fila "Mejor línea base" toma el mejor valor de cada columna: en Saposoa, una regresión logística sobre el vector de ResNet50 sin ajustar; en RoCoLe y Uganda, un random forest sobre ese mismo vector. Todas las líneas base están en `results/resumen.csv`.

| Modelo | Aumento | Saposoa F1 | RoCoLe AUC | Uganda AUC | Minutos por corrida (T4) |
|---|---|---|---|---|---|
| Mejor línea base | — | 1.000 | 0.750 | 0.767 | |
| ResNet50 | básico | 1.000 ± 0.000 | 0.831 ± 0.013 | 0.743 ± 0.142 | 1.6 |
| ResNet50 | fuerte | 1.000 ± 0.000 | 0.858 ± 0.006 | 0.779 ± 0.063 | 2.6 |
| ResNet50 | fuerte + fondo | 0.995 ± 0.005 | 0.858 ± 0.033 | 0.728 ± 0.064 | 4.8 |
| ViT-B/16 | básico | 1.000 ± 0.000 | 0.909 ± 0.011 | 0.717 ± 0.054 | 3.6 |
| ViT-B/16 | fuerte | 0.994 ± 0.007 | 0.920 ± 0.009 | 0.750 ± 0.052 | 4.0 |
| ViT-B/16 | fuerte + fondo | 0.990 ± 0.008 | 0.920 ± 0.009 | 0.777 ± 0.060 | 5.8 |

Tabla completa en `results/resumen.csv`. Costo total: 1.12 horas de GPU T4 (`results/costo.csv`).

- Dentro de Saposoa el problema es casi trivial; la dificultad está fuera de él.
- ViT-B/16 generaliza mejor que ResNet50 a RoCoLe. Uganda es difícil para todos los modelos.
- Sin reemplazo de fondo, los modelos aprenden a reconocer la clase por el fondo de Saposoa (`results/prueba_fondo/`). El reemplazo de fondo elimina ese atajo, pero no mejora la prueba externa: la caída se debe a que las hojas de campo se ven distintas, no al fondo.
- En fotos de campo los modelos detectan casi toda la roya pero confunden muchas hojas sanas con enfermas (`results/errores/`).
- El modelo siempre elige una de las tres clases: una hoja con otra enfermedad también terminaría como sana, roya u ojo de gallo.

### Auditoría local en RTX 5050

Christian André Urviola García reprodujo entrenamientos de ResNet50 y ViT-B/16 en una RTX 5050 y evaluó la etiqueta final que el modelo asigna a cada fotografía. En la corrida determinista de ViT-B/16 con cambio de fondo y semilla 0, el modelo acertó 362 de 1312 imágenes de RoCoLe (27.59 %) al elegir entre sana, roya y ojo de gallo. En esas mismas imágenes obtuvo un AUC de 0.909 al comparar únicamente roya frente a sana. Las dos cifras responden a preguntas distintas: qué etiqueta recibe cada foto y qué tan bien el modelo ordena las hojas con roya frente a las sanas.

El [informe de auditoría](docs/auditoria_rtx5050/README.md) incluye el entorno, el comando de ejecución y los resultados de la comparación.

## Pesos y demostración

Pesos de los 6 modelos de la semilla 0 (Google Drive, público): https://drive.google.com/drive/folders/1fIOa7MGDJUeVh-MrOiGy6_5snuy5TeF6?usp=sharing

El modelo de la demostración es `vit_b_16_fondo_s0.pt` (ViT-B/16 con aumento fuerte y cambio de fondo): tiene el mejor AUC en RoCoLe, empatado con ViT-B/16 fuerte, y no usa el fondo como atajo.

Demostración en Colab con ViT-B/16 (fuerte + fondo): [notebooks/demo.ipynb](notebooks/demo.ipynb)

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pomazon/Examen_Final_Redes_Neuronales/blob/main/notebooks/demo.ipynb)

## Uso de herramientas de IA

Se usó Claude (Anthropic) como asistente en cuatro puntos:

1. Código: escribir y depurar los scripts de `src/`, el notebook de demostración y la página de `web/`.
2. Orientación de los pasos: cómo ordenar el trabajo y qué técnicas aplicar en cada etapa.
3. Redacción y forma: ordenar la documentación del equipo y darle forma de artículo y de diapositivas.
4. Figuras: el código que las genera y el diagrama de la arquitectura.

Los integrantes ejecutaron todos los experimentos, revisaron las cifras contra `results/`, interpretaron los resultados y sacaron las conclusiones.

## Equipo

Grupo 2, sección A:

- Alexander André Chávez Cabana
- Alex Reynaldo Pomazón Solórzano
- Andhersson Manuel Salazar Fernández
- Christian André Urviola García
