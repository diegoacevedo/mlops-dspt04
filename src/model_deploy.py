# src/model_deploy.py

import io
import os
from typing import List, Optional

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, File, Response, UploadFile
from pydantic import BaseModel

from ft_engineering import COLUMNAS_NUMERIC, COLUMNAS_CATEGORIC, COLUMNAS_ORDINAL
from model_evaluation import evaluation

# --- 1. Definición de los modelos de datos con Pydantic ---

# Los nombres de los campos coinciden con las columnas CRUDAS del dataset de
# entrenamiento (ver ft_engineering.py: COLUMNAS_NUMERIC/CATEGORIC/ORDINAL).
# El pipeline cargado (modelo_pipeline.pkl) ya incluye el preprocesador
# (imputación + one-hot/ordinal encoding), así que la API recibe los datos
# tal cual vienen del negocio y NO espera que el cliente los codifique.
# Todos los campos son opcionales porque el propio pipeline sabe imputar
# valores faltantes (así se entrenó).
class PredictionInput(BaseModel):
    # Numéricas
    capital_prestado: Optional[float] = None
    plazo_meses: Optional[float] = None
    edad_cliente: Optional[float] = None
    salario_cliente: Optional[float] = None
    total_otros_prestamos: Optional[float] = None
    cuota_pactada: Optional[float] = None
    puntaje_datacredito: Optional[float] = None
    cant_creditosvigentes: Optional[float] = None
    huella_consulta: Optional[float] = None
    saldo_mora: Optional[float] = None
    saldo_total: Optional[float] = None
    saldo_principal: Optional[float] = None
    saldo_mora_codeudor: Optional[float] = None
    creditos_sectorFinanciero: Optional[float] = None
    creditos_sectorCooperativo: Optional[float] = None
    creditos_sectorReal: Optional[float] = None
    promedio_ingresos_datacredito: Optional[float] = None

    # Categóricas nominales
    tipo_laboral: Optional[str] = None
    tendencia_ingresos: Optional[str] = None

    # Categórica ordinal
    tipo_credito: Optional[float] = None


# Define la estructura para una solicitud por batch (lote).
class BatchPredictionInput(BaseModel):
    data: List[PredictionInput]


# Columnas que espera el preprocesador del pipeline, en el mismo orden que usa
# run_training.py. El ColumnTransformer selecciona por nombre, así que el
# orden real no importa, pero reindexar asegura que estén todas presentes.
FEATURE_COLUMNS = COLUMNAS_NUMERIC + COLUMNAS_CATEGORIC + COLUMNAS_ORDINAL


# --- 2. Inicialización de la aplicación y carga del modelo ---

app = FastAPI(
    title="API de Predicción de Pago a Tiempo",
    description="Despliega el pipeline (preprocesador + XGBoost) para predicciones por batch.",
    version="1.0.0"
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "modelo_pipeline.pkl")

# Cargar el pipeline entrenado al iniciar la aplicación.
# 'modelo_pipeline.pkl' es el mismo artefacto que genera model_training.py
# (joblib.dump) y que usa model_monitoring.py.
try:
    pipeline = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Error cargando el modelo: {e}")
    pipeline = None


# --- 3. Definición del endpoint de predicción ---

def _predict_dataframe(df: pd.DataFrame) -> dict:
    """
    Lógica de predicción compartida por /predict y /predict/csv:
    reindexa contra las columnas que espera el preprocesador (lo que falte
    queda como NaN y el imputador del pipeline se encarga, así se entrenó
    el modelo) y devuelve clase predicha + probabilidad por registro.
    """
    df = df.reindex(columns=FEATURE_COLUMNS)

    # predict_proba porque también devolvemos la probabilidad, no solo la clase
    probabilidades = pipeline.predict_proba(df)[:, 1]  # P(Pago_atiempo = 1)

    threshold = 0.5
    final_predictions = [1 if prob >= threshold else 0 for prob in probabilidades]

    return {
        "n_registros": len(df),
        "predictions": final_predictions,
        "probabilities": probabilidades.tolist(),
    }


@app.post('/predict')
async def predict_batch(input_data: BatchPredictionInput):
    """
    Endpoint para predicciones por batch a partir de JSON.
    Recibe una lista de registros (columnas crudas) y devuelve, por cada uno,
    la clase predicha (1 = paga a tiempo, 0 = no paga a tiempo) y la
    probabilidad de que el cliente pague a tiempo.
    """
    if pipeline is None:
        return {"error": "El modelo no pudo ser cargado. Revisa los logs del servidor."}

    if not input_data.data:
        return {"error": "El payload no contiene datos para predecir."}

    try:
        # Convierte la lista de objetos Pydantic a un DataFrame
        input_list = [item.model_dump() for item in input_data.data]
        df = pd.DataFrame(input_list)
        return _predict_dataframe(df)

    except Exception as e:
        return {'error': f"Ocurrió un error durante la predicción: {str(e)}"}


@app.post('/predict/csv')
async def predict_csv(file: UploadFile = File(...)):
    """
    Endpoint para predicciones por batch a partir de un archivo CSV.
    El CSV debe tener como columnas (al menos) las que usa el modelo -ver
    ft_engineering.COLUMNAS_NUMERIC/CATEGORIC/ORDINAL-; las que falten se
    imputan igual que en /predict.
    """
    if pipeline is None:
        return {"error": "El modelo no pudo ser cargado. Revisa los logs del servidor."}

    if not file.filename.lower().endswith(".csv"):
        return {"error": "El archivo debe tener extensión .csv"}

    try:
        contenido = await file.read()
        df = pd.read_csv(io.BytesIO(contenido))
    except Exception as e:
        return {"error": f"No se pudo leer el CSV: {str(e)}"}

    if df.empty:
        return {"error": "El CSV no contiene filas para predecir."}

    try:
        return _predict_dataframe(df)
    except Exception as e:
        return {'error': f"Ocurrió un error durante la predicción: {str(e)}"}


@app.get("/")
def read_root():
    return {"status": "OK", "message": "API de predicción está en línea. Usa el endpoint /predict."}


@app.get("/evaluation",
         responses={200: {"content": {"image/png": {}}}},
         description="Retorna una imagen PNG con la visualización de las métricas de evaluación del modelo."
)
async def serve_evaluation_plot():
    """
    Endpoint para visualizar la evaluación del modelo.
    Llama a la función importada para generar el gráfico y lo devuelve.
    """
    image_buffer = evaluation()  # <-- ¡Aquí usamos la función importada!
    if image_buffer:
        return Response(content=image_buffer.getvalue(), media_type="image/png")
    else:
        return {"error": "No se pudo generar el gráfico de evaluación."}


# Esto permite ejecutar el script directamente para pruebas locales
if __name__ == '__main__':
    uvicorn.run("model_deploy:app", host="0.0.0.0", port=8000, reload=True)
