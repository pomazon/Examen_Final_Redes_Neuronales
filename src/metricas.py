"""Métricas comunes para la línea base y los modelos.

Saposoa tiene tres clases. RoCoLe y Uganda solo tienen sana y roya: para ellas el
modelo elige entre esas dos (se ignora la probabilidad de ojo de gallo) y además se
reporta qué parte de las fotos habría clasificado como ojo de gallo.
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score

CLASES = ["ojo_gallo", "roya", "sana"]  # orden alfabético, el mismo que usa scikit-learn


def evaluar(y, proba, dataset):
    """y: etiquetas de texto. proba: matriz (n, 3) con columnas en el orden de CLASES."""
    y = np.asarray(y)
    proba = np.asarray(proba)
    if dataset == "saposoa":
        pred = np.array(CLASES)[proba.argmax(axis=1)]
        return {"exactitud": accuracy_score(y, pred),
                "f1_macro": f1_score(y, pred, average="macro"),
                "mcc": matthews_corrcoef(y, pred)}
    p_roya, p_sana = proba[:, 1], proba[:, 2]
    puntaje = p_roya / (p_roya + p_sana + 1e-12)
    pred = np.where(puntaje >= 0.5, "roya", "sana")
    return {"exactitud": accuracy_score(y, pred),
            "f1_macro": f1_score(y, pred, average="macro"),
            "mcc": matthews_corrcoef(y, pred),
            "auc": roc_auc_score(y == "roya", puntaje),
            "pred_ojo_gallo": float((proba.argmax(axis=1) == 0).mean())}
