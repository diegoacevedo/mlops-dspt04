# src/model_evaluation.py
import os
import io

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from cargar_datos import cargarDatos
from ft_engineering import (
    TARGET,
    COLUMNAS_NUMERIC,
    COLUMNAS_CATEGORIC,
    COLUMNAS_ORDINAL,
    limpiar_categoricas_con_ruido_numerico,
)

# Debe coincidir con la ruta que usan model_training.py (joblib.dump) y model_monitoring.py
MODEL_PATH = os.path.join(os.path.dirname(__file__), "modelo_pipeline.pkl")

# Mismos parámetros de split usados en model_training.build_model, para reproducir
# el mismo conjunto de test sobre el que quedó evaluado el modelo entrenado.
TEST_FRAC = 0.2
RANDOM_STATE = 1234

# Clase de interés del negocio: "no pago a tiempo" (la clase minoritaria)
POS_LABEL = 0

CUSTOM_TITLES = {
    "precision": "Precisión no pago a tiempo",
    "recall": "Recall no pago a tiempo",
    "f1_score": "F1 no pago a tiempo",
    "accuracy": "Accuracy General",
    "roc_auc": "ROC AUC",
    "casosNoPagoAtiempo": "Conteo Casos No Pago a Tiempo",
}
METRICS_TO_PLOT = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "casosNoPagoAtiempo"]


def _load_evaluation_data():
    """Carga los datos crudos y reproduce el mismo split usado en entrenamiento."""
    df = cargarDatos()
    df = limpiar_categoricas_con_ruido_numerico(
        df, "tendencia_ingresos", ["Creciente", "Decreciente", "Estable"]
    )

    feature_cols = COLUMNAS_NUMERIC + COLUMNAS_CATEGORIC + COLUMNAS_ORDINAL
    X = df[feature_cols]
    y = df[TARGET]

    return train_test_split(X, y, test_size=TEST_FRAC, random_state=RANDOM_STATE)


def summarize_split(y_true, y_pred, y_proba):
    """Métricas de clasificación tomando 'no pago a tiempo' (0) como clase positiva."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "casosNoPagoAtiempo": int((y_true == POS_LABEL).sum()),
    }


def build_results_df(model_name, model, x_train, x_test, y_train, y_test):
    y_pred_train = model.predict(x_train)
    y_proba_train = model.predict_proba(x_train)[:, 1]
    y_pred_test = model.predict(x_test)
    y_proba_test = model.predict_proba(x_test)[:, 1]

    train_metrics = summarize_split(y_train, y_pred_train, y_proba_train)
    test_metrics = summarize_split(y_test, y_pred_test, y_proba_test)

    records = []
    for data_set, metrics in [("train", train_metrics), ("test", test_metrics)]:
        for metric_name, score in metrics.items():
            records.append({
                "Model": model_name,
                "Data Set": data_set,
                "Metric": metric_name,
                "Score": score,
            })
    return pd.DataFrame(records)


def evaluation():
    """
    Carga el pipeline entrenado (preprocesador + modelo), reproduce el split
    train/test usado en entrenamiento, calcula las métricas de clasificación
    reales sobre ambos conjuntos y devuelve un PNG (BytesIO) comparándolas.
    """
    pipeline = joblib.load(MODEL_PATH)
    x_train, x_test, y_train, y_test = _load_evaluation_data()

    model_name = pipeline.named_steps["model"].__class__.__name__
    results_df = build_results_df(model_name, pipeline, x_train, x_test, y_train, y_test)

    fig, axes = plt.subplots(3, 2, figsize=(30, 18))
    axes = axes.flatten()
    fig.suptitle(f"Evaluaciones de modelo {model_name}", fontsize=30)

    for i, metric in enumerate(METRICS_TO_PLOT):
        ax = axes[i]
        metric_df = results_df[results_df["Metric"] == metric]

        sns.barplot(data=metric_df, x="Model", y="Score", hue="Data Set", ax=ax, palette="cividis")
        ax.legend(fontsize=18)
        title = CUSTOM_TITLES.get(metric, metric.replace("_", " ").title())
        ax.set_title(title, fontsize=24)
        ax.set_xticks([])
        ax.set_ylabel("Puntuación", fontsize=18)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45, labelsize=18)
        ax.tick_params(axis="y", labelsize=18)

        if metric == "roc_auc":
            ax.set_ylim(0.45, 1.05)
        elif metric == "casosNoPagoAtiempo":
            max_no_pago = results_df[results_df["Metric"] == "casosNoPagoAtiempo"]["Score"].max()
            ax.set_ylim(0, max_no_pago + 5)
        else:
            ax.set_ylim(0, 1.05)

    if len(METRICS_TO_PLOT) < len(axes):
        axes[-1].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    return buf


if __name__ == "__main__":
    buffer = evaluation()
    preview_path = os.path.join(os.path.dirname(__file__), "evaluation_preview.png")
    with open(preview_path, "wb") as f:
        f.write(buffer.getvalue())
    print(f"Gráfico de evaluación guardado en {preview_path}")
