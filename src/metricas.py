"""Métricas comunes para la línea base y los modelos.

Saposoa tiene tres clases. RoCoLe y Uganda solo tienen sana y roya: para ellas el
modelo se evalúa como un problema binario usando la probabilidad de roya frente a
sana, y además se reporta qué proporción de fotos habría sido asignada a ojo de gallo.
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score

CLASES = ["ojo_gallo", "roya", "sana"]  # orden alfabético, el mismo que usa scikit-learn


def evaluar(y, proba, dataset):
    """Calcula métricas de clasificación para cada conjunto.

    Args:
        y: etiquetas de texto.
        proba: matriz de probabilidades con forma (n_muestras, 3), en el orden de CLASES.
        dataset: nombre del conjunto; usa "saposoa" para tres clases y "rocole"/"uganda"
                 para el caso binario de roya vs sana.
    """
    y = np.asarray(y)
    proba = np.asarray(proba, dtype=np.float64)

    # En Saposoa la predicción es la clase de mayor probabilidad entre las tres.
    if dataset == "saposoa":
        pred = np.array(CLASES)[proba.argmax(axis=1)]
        return {
            "exactitud": accuracy_score(y, pred),
            "f1_macro": f1_score(y, pred, average="macro"),
            "mcc": matthews_corrcoef(y, pred),
        }

    p_roya, p_sana = proba[:, 1], proba[:, 2]
    # En los conjuntos externos comparamos roya y sana con su probabilidad relativa.
    puntaje = p_roya / (p_roya + p_sana + 1e-12)
    # El umbral da la etiqueta binaria; el AUC usa el puntaje continuo antes del corte.
    pred = np.where(puntaje >= 0.5, "roya", "sana")
    return {
        "exactitud": accuracy_score(y, pred),
        "f1_macro": f1_score(y, pred, average="macro"),
        "mcc": matthews_corrcoef(y, pred),
        "auc": roc_auc_score(y == "roya", puntaje),
        "pred_ojo_gallo": float((proba.argmax(axis=1) == 0).mean()),
    }
