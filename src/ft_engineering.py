import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

TARGET = "Pago_atiempo"

# Numéricas: solo imputación
COLUMNAS_NUMERIC = [
    "capital_prestado",
    "plazo_meses",
    "edad_cliente",
    "salario_cliente",
    "total_otros_prestamos",
    "cuota_pactada",
    "puntaje_datacredito",
    "cant_creditosvigentes",
    "huella_consulta",
    "saldo_mora",
    "saldo_total",
    "saldo_principal",
    "saldo_mora_codeudor",
    "creditos_sectorFinanciero",
    "creditos_sectorCooperativo",
    "creditos_sectorReal",
    "promedio_ingresos_datacredito"
]

# Categóricas nominales: imputación + OneHotEncoder
COLUMNAS_CATEGORIC = ["tipo_laboral", "tendencia_ingresos"]

# Categóricas ordinales: imputación + OrdinalEncoder (orden inferido en fit)
COLUMNAS_ORDINAL = ["tipo_credito"]

EXCLUIR_DE_X = ["fecha_prestamo","puntaje"]

def limpiar_categoricas_con_ruido_numerico(df, columna, categorias_validas):
    """Reemplaza valores fuera de las categorías válidas por NaN."""
    df[columna] = df[columna].where(df[columna].isin(categorias_validas), np.nan)
    return df

def _get_feature_columns(df: pd.DataFrame) -> tuple[list, list, list]:
    """Obtiene listas de columnas por tipo presentes en df."""
    numeric = [c for c in COLUMNAS_NUMERIC if c in df.columns]
    cat_ = [c for c in COLUMNAS_CATEGORIC if c in df.columns]
    ord_ = [c for c in COLUMNAS_ORDINAL if c in df.columns]
    return numeric, cat_, ord_


def build_preprocessor(numeric_cols, categorical_cols, ordinal_cols):
    """
    Construye el ColumnTransformer según el diagrama:
    - numeric: SimpleImputer
    - categorical: SimpleImputer -> OneHotEncoder
    - ordinal: SimpleImputer -> OrdinalEncoder
    """

    transformers = []

    if numeric_cols:
        transformers.append(
            ("numeric", SimpleImputer(strategy="median"), numeric_cols),
        )

    if categorical_cols:
        transformers.append(
            (
                "categoric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)),
                ]),
                categorical_cols,
            )
        )

    if ordinal_cols:
        transformers.append(
            (
                "ordinal",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                ]),
                ordinal_cols,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )