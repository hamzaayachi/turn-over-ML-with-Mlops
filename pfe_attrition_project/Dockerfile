# Dockerfile pour l'application de prédiction d'attrition
# Version unique et complète

FROM python:3.9-slim

# Métadonnées
LABEL maintainer="PFE Bachelor 2026"
LABEL description="Attrition Prediction Application"
LABEL version="1.0.0"

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODE=all

# Installation des dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Création de l'utilisateur non-root
RUN adduser --system --uid 1001 appuser

# Répertoire de travail
WORKDIR /app

# Copie des requirements
COPY requirements.txt .

# Installation des dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copie de tout le projet
COPY --chown=appuser:appuser . .

# Création des dossiers
RUN mkdir -p data/raw data/processed data/new models reports monitoring_reports mlruns

# Exposition des ports
EXPOSE 8501 8000 5000

# Utilisateur non-root
USER appuser

# Script d'entrée
CMD ["sh", "-c", "case \"$MODE\" in streamlit) streamlit run app.py --server.port=8501 --server.address=0.0.0.0 ;; api) uvicorn api:app --host 0.0.0.0 --port 8000 ;; mlflow) mlflow server --host 0.0.0.0 --port 5000 ;; training) python prefect_pipeline.py --mode train --force ;; monitoring) python monitoring.py --mode once ;; *) echo \"Modes: streamlit, api, mlflow, training, monitoring\" && streamlit run app.py --server.port=8501 --server.address=0.0.0.0 ;; esac"]