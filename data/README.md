# Datos

Los datos no se guardan en este repositorio. Se descargan con:

```
python src/descargar_datos.py --destino data/raw
```

## Fuentes y licencias

Los tres conjuntos están en Mendeley Data con licencia **Creative Commons Attribution 4.0 International (CC BY 4.0)**: https://creativecommons.org/licenses/by/4.0/

| Conjunto | Uso en el proyecto | Autores | Año | DOI |
|---|---|---|---|---|
| Coffee leaf dataset by phytosanitary class (Saposoa, San Martín, Perú) | Entrenamiento, validación y prueba | John Clark Santa Maria Pinedo | 2026 | https://doi.org/10.17632/mfpxg4y65r.2 |
| RoCoLe: A robusta coffee leaf images dataset (Ecuador) | Prueba externa | J. Parraga-Alava, K. Cusme, A. Loor, E. Santander | 2019 | https://doi.org/10.17632/c5yvn32dzg.2 |
| A Machine Learning Dataset for Classification of Common Coffee Leaf Diseases in Uganda | Prueba externa | S. Chelangat, R. Anirwoth, K. N. Mayanja, A. Sserwadda | 2025 | https://doi.org/10.17632/k36wnd6knb.1 |

Artículo asociado a RoCoLe: Parraga-Alava, J. et al. (2019). *RoCoLe: A robusta coffee leaf images dataset for evaluation of machine learning based methods in plant diseases recognition*. Data in Brief, 25, 104414. https://doi.org/10.1016/j.dib.2019.104414

Artículo de los autores de Saposoa: Santa-María, J. C. y Rodríguez, C. (2026). *Automatic detection of coffee leaf diseases through pattern recognition*. Revista Científica de Sistemas e Informática, 6(1), e1349. https://doi.org/10.51252/rcsi.v6i1.1349

Los autores de los datos no respaldan ni avalan este proyecto.

## Versiones usadas

Se fija la versión de cada conjunto: si los autores publican una nueva, los resultados podrían cambiar.

| Conjunto | Versión | Tamaño | Descarga directa |
|---|---|---|---|
| Saposoa | v2 (18 sep 2026) | 3.21 GB, un .rar dentro del zip | https://data.mendeley.com/public-api/zip/mfpxg4y65r/download/2 |
| RoCoLe | v2 (17 may 2019) | 2.09 GB | https://data.mendeley.com/public-api/zip/c5yvn32dzg/download/2 |
| Uganda | v1 (7 feb 2025) | 24.8 MB | https://data.mendeley.com/public-api/zip/k36wnd6knb/download/1 |

## Cambios respecto a los originales

CC BY 4.0 exige indicar los cambios. Los datos originales no se modifican: se descargan tal cual y los cambios se aplican sobre copias.

- **Etiquetas unificadas** en tres clases: sana, roya y ojo de gallo. RoCoLe: `healthy` → sana; `rust_level_1` a `rust_level_4` → roya. Uganda: `Health leaves` → sana; `leaf rust` → roya.
- **Clases excluidas** por no existir en Saposoa: `red_spider_mite` (RoCoLe) y `phoma` (Uganda).
- **Imágenes excluidas:** las que no se pueden abrir, las copias exactas (mismo md5) y las casi iguales (similitud ≥ umbral con vectores de ResNet50), dejando una de cada grupo; si las copias tenían clases distintas, se excluyen todas. Lista y motivo en `results/particiones/excluidas.csv`.
- **Tamaño:** se reducen para que el lado menor mida 320 px (sin agrandar las pequeñas) y se guardan como JPEG calidad 95.
- **`data/ejemplos/`:** siete de esas imágenes reducidas, para el notebook de demostración. `fuentes.csv` indica de qué archivo original sale cada una.
