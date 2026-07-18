import os
import time
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from evidently import Report
from evidently.presets import DataDriftPreset
from sklearn.model_selection import train_test_split

# ================================
# 1. Configuración
# ================================
MODEL_PATH = "modelo_pipeline.pkl"   # pipeline completo (preprocesador + modelo)
DATASET_PATH = "../Base_de_datos_drift.csv"   # dataset crudo
MONITOR_LOG = "monitoring_log.csv"  # dataset para monitorear

# ================================
# 2. Cargar pipeline entrenado
# ================================
@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)

pipeline = load_pipeline()

# ================================
# 3. Cargar dataset y dividir
# ================================
@st.cache_data
def load_data():
    df = pd.read_csv(DATASET_PATH)
    target = "Pago_atiempo"
    X = df.drop(columns=[target])
    y = df[target]
    X_ref, X_new, y_ref, y_new = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_ref, X_new, y_ref, y_new

X_ref, X_new, y_ref, y_new = load_data()

# ================================
# 4. Predicciones usando el pipeline local
# ================================
def get_predictions(X_batch: pd.DataFrame):
    try:
        # predict_proba porque el dashboard trata "prediction" como continua
        # (mean, std, tasa > 0.5). Con .predict() solo tendrías 0/1.
        preds = pipeline.predict_proba(X_batch)[:, 1]
        return preds.tolist()
    except Exception as e:
        st.error(f"❌ Error generando predicciones: {e}")
        return None

# ================================
# 5. Guardar logs con timestamp
# ================================
def log_predictions(X_batch, preds):
    log_df = X_batch.copy()
    log_df["prediction"] = preds
    log_df["timestamp"] = pd.Timestamp.now()

    if os.path.exists(MONITOR_LOG):
        log_df.to_csv(MONITOR_LOG, mode="a", header=False, index=False)
    else:
        log_df.to_csv(MONITOR_LOG, index=False)

# ================================
# 6. Reporte Evidently
# ================================
def generate_drift_report(ref_data, new_data):
    report = Report(metrics=[DataDriftPreset()])
    result = report.run(reference_data=ref_data, current_data=new_data)
    return result

# ================================
# 7. Streamlit UI con gráficas
# ================================
st.set_page_config(page_title="Monitoreo del Modelo", layout="wide")
st.title("📊 Monitoreo del Modelo en Producción")

# Métricas principales en la parte superior
if os.path.exists(MONITOR_LOG):
    logged_data = pd.read_csv(MONITOR_LOG)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Predicciones", len(logged_data))
    with col2:
        st.metric("Predicción Promedio", f"{logged_data['prediction'].mean():.3f}")
    with col3:
        st.metric("Desviación Estándar", f"{logged_data['prediction'].std():.3f}")
    with col4:
        positive_rate = (logged_data['prediction'] > 0.5).mean() * 100
        st.metric("Tasa Positiva (%)", f"{positive_rate:.1f}%")

st.sidebar.header("Opciones")
sample_size = st.sidebar.slider("Tamaño de muestra para monitoreo:", 50, 500, 200)

if st.button("🔄 Generar nuevas predicciones y actualizar log"):
    sample = X_new.sample(n=sample_size, random_state=int(time.time()))
    preds = get_predictions(sample)
    if preds:
        log_predictions(sample, preds)
        st.success("✅ Nuevas predicciones agregadas al log.")
        st.rerun()

# Mostrar datos y gráficas
if os.path.exists(MONITOR_LOG):
    logged_data = pd.read_csv(MONITOR_LOG)

    # Crear tabs para organizar mejor
    tab1, tab2, tab3 = st.tabs(["📈 Gráficas", "📊 Data Drift", "📂 Logs"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            # Histograma de predicciones
            fig_hist = px.histogram(
                logged_data,
                x='prediction',
                nbins=20,
                title="Distribución de Predicciones",
                color_discrete_sequence=['#1f77b4']
            )
            st.plotly_chart(fig_hist, width="stretch")

        with col2:
            # Gráfico de línea temporal (si hay timestamp)
            if 'timestamp' in logged_data.columns:
                logged_data['timestamp'] = pd.to_datetime(logged_data['timestamp'])
                # Agrupar por minuto para mejor visualización
                temporal_data = logged_data.groupby(
                    logged_data['timestamp'].dt.floor('min')
                )['prediction'].mean().reset_index()

                fig_time = px.line(
                    temporal_data,
                    x='timestamp',
                    y='prediction',
                    title="Evolución Temporal de Predicciones",
                    color_discrete_sequence=['#ff7f0e']
                )
                st.plotly_chart(fig_time, width="stretch")
            else:
                # Box plot como alternativa
                fig_box = px.box(
                    logged_data,
                    y='prediction',
                    title="Distribución de Predicciones (Box Plot)"
                )
                st.plotly_chart(fig_box, width="stretch")

        # Gráfico de comparación con datos de referencia
        st.subheader("🔍 Comparación con Datos de Referencia")

        # Seleccionar algunas columnas numéricas para comparar
        numeric_cols = logged_data.select_dtypes(include=['float64', 'int64']).columns
        numeric_cols = [col for col in numeric_cols if col != 'prediction'][2:8]  # Solo las primeras 4

        if len(numeric_cols) > 0:
            cols_layout = st.columns(len(numeric_cols))

            for i, col in enumerate(numeric_cols):
                if col in X_ref.columns:
                    fig_individual = go.Figure()
                    fig_individual.add_trace(go.Bar(
                        x=['Referencia'],
                        y=[X_ref[col].mean()],
                        marker_color='lightblue',
                        name='Referencia'
                    ))
                    fig_individual.add_trace(go.Bar(
                        x=['Actual'],
                        y=[logged_data[col].mean()],
                        marker_color='orange',
                        name='Actual'
                    ))
                    fig_individual.update_layout(
                        title=col,
                        showlegend=False,
                        height=350
                    )
                    with cols_layout[i]:
                        st.plotly_chart(fig_individual, width="stretch")

    with tab2:
        st.subheader("📊 Reporte de Data Drift")
        drift_report = generate_drift_report(
            X_ref, logged_data.drop(columns=["prediction", "timestamp"], errors="ignore")
        )

        # Mostrar reporte
        try:
            drift_report.save_html("drift_report.html")
            with open("drift_report.html", "r", encoding="utf-8") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=1000, scrolling=True)
        except Exception:
            st.write("✅ Reporte de Drift generado exitosamente")
            st.write(f"📊 Datos de referencia: {X_ref.shape}, Datos actuales: {logged_data.drop(columns=['prediction', 'timestamp'], errors='ignore').shape}")

            # Mostrar métricas básicas de drift
            try:
                drift_data = drift_report.as_dict()
                if 'metrics' in drift_data and len(drift_data['metrics']) > 0:
                    dataset_drift = drift_data['metrics'][0].get('result', {}).get('dataset_drift', 'No disponible')

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Dataset Drift Detectado", "Sí" if dataset_drift else "No")
                    with col2:
                        # Contar cuántas features tienen drift
                        feature_drifts = drift_data['metrics'][0].get('result', {}).get('drift_by_columns', {})
                        drift_count = sum(1 for v in feature_drifts.values() if v) if feature_drifts else 0
                        st.metric("Features con Drift", f"{drift_count}/{len(feature_drifts)}" if feature_drifts else "0")
            except Exception:
                pass

    with tab3:
        st.subheader("📂 Log de Monitoreo")

        # Filtro para mostrar más o menos filas
        show_rows = st.selectbox("Mostrar últimas:", [10, 25, 50, 100], index=0)
        st.dataframe(logged_data.tail(show_rows), width="stretch")

        # Botón de descarga
        csv = logged_data.to_csv(index=False)
        st.download_button(
            label="📥 Descargar CSV completo",
            data=csv,
            file_name=f"monitoring_log_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

else:
    st.warning("⚠️ No hay datos de monitoreo aún. Presiona el botón para iniciar.")