# 1. Imagen base oficial de Python ligera y segura
FROM python:3.10-slim as base

# Evita que Python escriba archivos .pyc y fuerza la salida de logs sin buffer
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 2. Crear un usuario no raíz para seguir buenas prácticas de DevSecOps
RUN useradd --create-home appuser
WORKDIR /app

# 3. Copiar e instalar dependencias primero (aprovecha la caché de capas de Docker)
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir .

# 4. Copiar el resto del código del repositorio
COPY cancer_ga/ ./cancer_ga/
COPY main.py ./

# Cambiar permisos al usuario no privilegiado
RUN chown -R appuser:appuser /app
USER appuser

# 5. Comando por defecto al ejecutar el contenedor
CMD ["python", "main.py"]