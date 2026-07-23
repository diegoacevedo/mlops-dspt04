# Imagen para disponibilizar el modelo de riesgo crediticio como API (FastAPI + Uvicorn).
FROM python:3.11-slim

# Evita archivos .pyc y logs bufferizados (para ver los logs de uvicorn en tiempo real)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Instala las dependencias primero para aprovechar el cache de capas de Docker:
# si el código cambia pero requirements.txt no, esta capa no se reconstruye.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto del código fuente y los datos necesarios (ver .dockerignore
# para lo que queda afuera: entorno virtual, notebooks, logs generados, etc.)
COPY . .

# model_deploy.py y el resto de los módulos (ft_engineering, cargar_datos,
# model_evaluation, modelo_pipeline.pkl) viven en src/ y resuelven sus rutas
# relativas a esa carpeta.
WORKDIR /app/src

EXPOSE 8000

CMD ["uvicorn", "model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]
