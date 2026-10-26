# Datos

Este proyecto usa tres conjuntos de datos públicos de hojas de café. **Los datos no se guardan en este repositorio**; se descargan con un comando (ver "Cómo obtener los datos").

## Fuentes y licencias

Los tres conjuntos están publicados en Mendeley Data bajo licencia **Creative Commons Attribution 4.0 International (CC BY 4.0)**: https://creativecommons.org/licenses/by/4.0/

| Conjunto | Uso en el proyecto | Autores | Año | DOI |
|---|---|---|---|---|
| Coffee leaf dataset by phytosanitary class (Saposoa, San Martín, Perú) | Entrenamiento, validación y prueba interna | John Clark Santa Maria Pinedo | 2026 | https://doi.org/10.17632/mfpxg4y65r.2 |
| RoCoLe: A robusta coffee leaf images dataset | Prueba externa (generalización) | J. Parraga-Alava, K. Cusme, A. Loor, E. Santander | 2019 | https://doi.org/10.17632/c5yvn32dzg.2 |
| A Machine Learning Dataset for Classification of Common Coffee Leaf Diseases in Uganda | Prueba externa (generalización) | S. Chelangat, R. Anirwoth, K. N. Mayanja, A. Sserwadda | 2025 | https://doi.org/10.17632/k36wnd6knb.1 |

Versiones usadas (fijadas para reproducibilidad) y descarga directa:

| Conjunto | Versión | Tamaño | URL de descarga directa |
|---|---|---|---|
| Saposoa | v2 (18 sep 2026) | 3.21 GB (zip con un .rar dentro) | https://data.mendeley.com/public-api/zip/mfpxg4y65r/download/2 |
| RoCoLe | v2 (17 may 2019) | 2.09 GB (etiquetas en archivos de anotación) | https://data.mendeley.com/public-api/zip/c5yvn32dzg/download/2 |
| Uganda | v1 (7 feb 2025) | 24.8 MB (256×256, incluye imágenes aumentadas) | https://data.mendeley.com/public-api/zip/k36wnd6knb/download/1 |

Artículo asociado a RoCoLe: Parraga-Alava, J. et al. (2019). *RoCoLe: A robusta coffee leaf images dataset for evaluation of machine learning based methods in plant diseases recognition*. Data in Brief, 25, 104414.

Los titulares de los derechos no respaldan ni avalan este proyecto ni el uso que hacemos de los datos.

## Cambios realizados respecto a los originales

CC BY 4.0 exige indicar si se hicieron cambios. La versión procesada que publicamos difiere de los originales en lo siguiente:

- TODO (completar al terminar la Fase 1):
  - Redimensionado: lado mayor a 512 px, manteniendo la proporción.
  - Recodificación: JPEG, calidad TODO.
  - Imágenes corruptas eliminadas: TODO (número por conjunto).
  - Duplicados exactos y casi-duplicados: TODO (eliminados o agrupados en la misma partición).
  - Correspondencia de clases entre conjuntos: TODO (tabla de mapeo).

La versión procesada se distribuye también bajo CC BY 4.0, con la atribución de esta sección.

## Cómo obtener los datos

TODO (Fase 1). Habrá dos rutas:

- **Ruta rápida (recomendada):** `python src/descargar_datos.py --version procesada` descarga la versión ya redimensionada (cientos de MB) desde Hugging Face y verifica su md5.
- **Ruta completa:** `python src/descargar_datos.py --version original` descarga los originales desde Mendeley (~9 GB) y luego `python src/preprocesar.py` genera exactamente la misma versión procesada.

Las particiones de entrenamiento, validación y prueba están en `data/splits.csv` (ruta, md5, clase y partición), generadas con semilla fija.
