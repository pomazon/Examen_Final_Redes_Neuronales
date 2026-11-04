# Configuraciones

Un archivo por experimento: modelo, tipo de aumento de datos e hiperparámetros. Se usan con `python src/entrenar.py --config configs/<archivo>.yaml --semilla <s>`.

| Archivo | Modelo | Aumento |
|---|---|---|
| resnet50_basico.yaml | ResNet50 | recorte del 70 al 100 % y volteo horizontal |
| resnet50_fuerte.yaml | ResNet50 | recorte del 40 al 100 %, volteos, giros, color y desenfoque |
| resnet50_fondo.yaml | ResNet50 | fuerte + cambio de fondo en 8 de cada 10 fotos |
| vit_b_16_basico.yaml | ViT-B/16 | recorte del 70 al 100 % y volteo horizontal |
| vit_b_16_fuerte.yaml | ViT-B/16 | recorte del 40 al 100 %, volteos, giros, color y desenfoque |
| vit_b_16_fondo.yaml | ViT-B/16 | fuerte + cambio de fondo en 8 de cada 10 fotos |

Hiperparámetros comunes: AdamW, decaimiento de pesos 0.05, 15 épocas, lotes de 32. Tasa de aprendizaje inicial: 1e-4 en ResNet50 y 3e-5 en ViT-B/16, con programa coseno.
