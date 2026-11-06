\# Auditoría de reproducibilidad y generalización — RTX 5050



\## Objetivo



Reproducir y validar los experimentos del proyecto en una NVIDIA RTX 5050 Laptop GPU, utilizando los datasets originales Saposoa, RoCoLe y Uganda.



\## Entorno



\- GPU: NVIDIA RTX 5050 Laptop GPU (8 GB).

\- PyTorch: 2.8.0+cu128.

\- CUDA: 12.8.

\- Sistema operativo: Windows.

\- Semilla: 0.

\- Épocas: 15.



\## Contribución realizada



\- Reproducción local de los entrenamientos ResNet50 y ViT-B/16.

\- Adaptación de configuraciones ViT a batch 8.

\- Evaluación de la exactitud final de tres clases.

\- Análisis de matrices de confusión en datasets externos.

\- Auditoría determinista mediante dos entrenamientos independientes.

\- Documentación de los hallazgos y limitaciones.



\## Resultados experimentales



| Modelo | Configuración | Exactitud RoCoLe | Exactitud Uganda |

|---|---|---:|---:|

| ResNet50 | Básico | 11.36 % | 2.69 % |

| ViT-B/16 | Básico | 43.60 % | 48.97 % |

| ViT-B/16 | Fuerte | 30.95 % | 28.92 % |

| ViT-B/16 | Fondo determinista | 27.59 % | 26.50 % |



La exactitud corresponde a la predicción final entre las tres clases: ojo de gallo, roya y sana. Es distinta del AUC binario condicionado presentado en los experimentos originales.



\## Auditoría determinista



Se ejecutaron dos entrenamientos independientes de ViT-B/16 con modificación de fondo, utilizando la misma semilla y configuración.



\*\*Resultados:\*\*



\- Mejor época: 15 en ambas ejecuciones.

\- AUC RoCoLe: 0.909 en ambas.

\- AUC Uganda: 0.802 en ambas.

\- Historiales CSV idénticos.

\- Predicciones CSV idénticas.

\- Pesos del modelo idénticos byte por byte.



Esto demuestra repetibilidad local bajo el protocolo determinista empleado. No garantiza resultados idénticos entre GPUs diferentes.



\## Conclusión



Se comprobó que los modelos pueden entrenarse y evaluarse en un equipo diferente al originalmente utilizado.



Los resultados también evidencian limitaciones importantes de generalización: una AUC binaria condicionada elevada no garantiza una clasificación final correcta cuando el modelo dispone de tres clases de salida.



Los resultados comparativos son exploratorios porque corresponden a una sola semilla por configuración.



\## Uso de IA



Se utilizó asistencia de IA para elaborar y revisar scripts auxiliares y documentación. La ejecución de los experimentos y la verificación de sus resultados se realizaron en el equipo local.

