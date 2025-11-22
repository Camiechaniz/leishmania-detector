# ---- Imagen base con Python estable (3.10) ----
FROM python:3.10-slim

# ---- Instalar dependencias del sistema necesarias ----
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# ---- Crear directorio de trabajo ----
WORKDIR /app

# ---- Copiar requisitos ----
COPY requirements.txt .

# ---- Instalar dependencias de Python ----
RUN pip install --no-cache-dir -r requirements.txt

# ---- Copiar tu aplicación ----
COPY . .

# ---- Exponer puerto del servidor ----
EXPOSE 10000

# ---- Comando para iniciar Streamlit ----
CMD ["streamlit", "run", "app.py", "--server.port=10000", "--server.address=0.0.0.0"]
