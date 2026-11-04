# Resultados

Tablas, figuras y registros que se usan en el artículo. Cada carpeta la genera un script de `src/` (ver la tabla "De cada tabla y figura del artículo a su comando" en el README principal).

| Carpeta o archivo | Qué contiene | Script |
|---|---|---|
| `inventario/` | Lista de todas las fotos descargadas (tamaño, md5, si se pueden abrir, clase original) y muestras | `inventario.py`, `ver_muestras.py` |
| `duplicados/` | Pares de fotos casi iguales y figuras por tramo de similitud | `duplicados.py` |
| `particiones/` | Fotos por conjunto y partición (`conteo.csv`) y fotos excluidas con su motivo (`excluidas.csv`) | `particiones.py` |
| `linea_base/` | Métricas de las líneas base, por semilla y promediadas | `linea_base.py` |
| `aumentos/` | Revisión de la máscara de la hoja | `ver_mascaras.py` |
| `entrenamiento/<config>/semilla_<s>/` | Por cada corrida: métricas (`metricas.json`), pérdida por época (`historial.csv`) y probabilidades de cada foto (`predicciones.csv`) | `entrenar.py` |
| `resumen.csv` | Tabla final: media ± desviación de las 3 semillas, con las líneas base | `resumen.py` |
| `costo.csv` | Minutos de GPU por configuración | `costo.py` |
| `gradcam/` | Mapas Grad-CAM y fracción del mapa dentro de la hoja | `gradcam.py` |
| `errores/` | Matrices de confusión y errores con mucha seguridad | `errores.py` |
| `prueba_fondo/` | F1 con la foto original, con fondo gris y solo con el fondo | `prueba_fondo.py` |
| `articulo/` | Figuras del artículo y diagrama de la arquitectura | `figuras_articulo.py` |
