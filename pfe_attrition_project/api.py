"""
API FastAPI pour la prédiction d'attrition des employés.
Expose des endpoints REST pour les prédictions individuelles et par lot.
"""

import os
import pickle
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, validator
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import uvicorn
import io

# ============================================================================
# CONFIGURATION
# ============================================================================

MODELS_DIR = "models"
PRODUCTION_DIR = "production"
API_VERSION = "1.0.0"
API_TITLE = "Attrition Prediction API"
API_DESCRIPTION = """
## API de Prédiction du Risque d'Attrition des Employés

Cette API permet de prédire la probabilité qu'un employé quitte l'entreprise
en utilisant un modèle de Machine Learning (Random Forest).

### Niveaux de risque:
- 🟢 **Faible** : Probabilité < 40%
- 🟡 **Modéré** : Probabilité entre 40% et 60%
- 🔴 **Élevé** : Probabilité > 60%
"""

# ============================================================================
# MODÈLES PYDANTIC
# ============================================================================

class EmployeeData(BaseModel):
    """Modèle de données pour un employé."""
    
    # Démographie
    Age: int = Field(..., ge=18, le=70, description="Âge de l'employé")
    Gender: str = Field(..., pattern="^(Male|Female)$", description="Genre")
    MaritalStatus: str = Field(..., pattern="^(Single|Married|Divorced)$", description="Situation familiale")
    
    # Carrière
    Department: str = Field(..., description="Département")
    JobRole: str = Field(..., description="Poste occupé")
    JobLevel: int = Field(..., ge=1, le=5, description="Niveau hiérarchique")
    YearsAtCompany: int = Field(..., ge=0, le=40, description="Années dans l'entreprise")
    YearsInCurrentRole: int = Field(..., ge=0, le=20, description="Années dans le poste actuel")
    YearsSinceLastPromotion: int = Field(..., ge=0, le=15, description="Années depuis dernière promotion")
    YearsWithCurrManager: int = Field(..., ge=0, le=20, description="Années avec manager actuel")
    NumCompaniesWorked: int = Field(..., ge=0, le=20, description="Nombre d'entreprises précédentes")
    
    # Rémunération
    MonthlyIncome: float = Field(..., gt=0, le=25000, description="Revenu mensuel ($)")
    PercentSalaryHike: int = Field(..., ge=0, le=50, description="Pourcentage d'augmentation")
    StockOptionLevel: int = Field(..., ge=0, le=3, description="Niveau de stock-options")
    
    # Satisfaction
    JobSatisfaction: int = Field(..., ge=1, le=4, description="Satisfaction au travail")
    EnvironmentSatisfaction: int = Field(..., ge=1, le=4, description="Satisfaction environnement")
    RelationshipSatisfaction: int = Field(..., ge=1, le=4, description="Satisfaction relations")
    WorkLifeBalance: int = Field(..., ge=1, le=4, description="Équilibre vie pro/perso")
    JobInvolvement: int = Field(..., ge=1, le=4, description="Implication au travail")
    
    # Conditions de travail
    OverTime: str = Field(..., pattern="^(Yes|No)$", description="Heures supplémentaires")
    BusinessTravel: str = Field(..., pattern="^(Non-Travel|Travel_Rarely|Travel_Frequently)$", description="Fréquence des voyages")
    DistanceFromHome: int = Field(..., ge=0, le=50, description="Distance domicile-travail (km)")
    TrainingTimesLastYear: int = Field(..., ge=0, le=6, description="Formations suivies")
    
    # Éducation
    Education: int = Field(..., ge=1, le=5, description="Niveau d'éducation")
    EducationField: str = Field(..., description="Domaine d'éducation")
    
    # Performance
    PerformanceRating: int = Field(..., ge=1, le=4, description="Note de performance")
    
    # Autres
    HourlyRate: int = Field(..., ge=30, le=100, description="Taux horaire")
    MonthlyRate: int = Field(..., ge=2000, le=27000, description="Taux mensuel")
    TotalWorkingYears: int = Field(..., ge=0, le=40, description="Années d'expérience totales")
    
    class Config:
        schema_extra = {
            "example": {
                "Age": 35,
                "Gender": "Male",
                "MaritalStatus": "Single",
                "Department": "Sales",
                "JobRole": "Sales Executive",
                "JobLevel": 2,
                "YearsAtCompany": 5,
                "YearsInCurrentRole": 3,
                "YearsSinceLastPromotion": 2,
                "YearsWithCurrManager": 3,
                "NumCompaniesWorked": 2,
                "MonthlyIncome": 5000,
                "PercentSalaryHike": 15,
                "StockOptionLevel": 1,
                "JobSatisfaction": 3,
                "EnvironmentSatisfaction": 3,
                "RelationshipSatisfaction": 3,
                "WorkLifeBalance": 3,
                "JobInvolvement": 3,
                "OverTime": "No",
                "BusinessTravel": "Travel_Rarely",
                "DistanceFromHome": 5,
                "TrainingTimesLastYear": 2,
                "Education": 3,
                "EducationField": "Life Sciences",
                "PerformanceRating": 3,
                "HourlyRate": 65,
                "MonthlyRate": 15000,
                "TotalWorkingYears": 10
            }
        }


class PredictionResponse(BaseModel):
    """Réponse de prédiction."""
    attrition_probability: float
    risk_level: str
    prediction: int
    prediction_label: str
    confidence: float
    risk_factors: List[str]
    recommendations: List[str]
    model_version: str
    timestamp: str


class BatchPredictionResponse(BaseModel):
    """Réponse de prédiction par lot."""
    results: List[PredictionResponse]
    total_processed: int
    timestamp: str


class HealthResponse(BaseModel):
    """Réponse de health check."""
    status: str
    model_version: str
    model_loaded: bool
    api_version: str
    timestamp: str


# ============================================================================
# CHARGEMENT DU MODÈLE
# ============================================================================

class ModelLoader:
    """Chargeur de modèle avec cache."""
    
    _instance = None
    _model = None
    _scaler = None
    _feature_columns = None
    _model_version = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load(self):
        """Charge le modèle et les artefacts."""
        if self._model is not None:
            return self
        
        print("📦 Chargement du modèle...")
        
        # Recherche du modèle
        for dir_path in [PRODUCTION_DIR, MODELS_DIR, '.']:
            model_path = os.path.join(dir_path, 'production_model.pkl')
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self._model = pickle.load(f)
                print(f"✅ Modèle chargé depuis {model_path}")
                break
        
        if self._model is None:
            print("⚠️ Modèle non trouvé, création d'un modèle par défaut")
            self._model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Chargement du scaler
        for dir_path in [PRODUCTION_DIR, MODELS_DIR, '.']:
            scaler_path = os.path.join(dir_path, 'scaler.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self._scaler = pickle.load(f)
                break
        
        # Chargement des colonnes
        for dir_path in [PRODUCTION_DIR, MODELS_DIR, '.']:
            cols_path = os.path.join(dir_path, 'feature_columns.pkl')
            if os.path.exists(cols_path):
                with open(cols_path, 'rb') as f:
                    self._feature_columns = pickle.load(f)
                break
        
        # Chargement de la version
        for dir_path in [PRODUCTION_DIR, MODELS_DIR, '.']:
            version_path = os.path.join(dir_path, 'version.txt')
            if os.path.exists(version_path):
                with open(version_path, 'r') as f:
                    for line in f:
                        if line.startswith('MODEL_VERSION='):
                            self._model_version = line.split('=')[1].strip()
                            break
                break
        
        if self._model_version is None:
            self._model_version = "1.0.0"
        
        return self
    
    @property
    def model(self):
        if self._model is None:
            self.load()
        return self._model
    
    @property
    def scaler(self):
        if self._scaler is None:
            self.load()
        return self._scaler
    
    @property
    def feature_columns(self):
        if self._feature_columns is None:
            self.load()
        return self._feature_columns
    
    @property
    def model_version(self):
        if self._model_version is None:
            self.load()
        return self._model_version


# ============================================================================
# SERVICE DE PRÉDICTION
# ============================================================================

class PredictionService:
    """Service de prédiction."""
    
    def __init__(self):
        self.loader = ModelLoader()
        self.loader.load()
    
    def preprocess_input(self, employee_data: Dict) -> np.ndarray:
        """Prétraite les données d'entrée."""
        input_df = pd.DataFrame([employee_data])
        input_encoded = pd.get_dummies(input_df)
        
        feature_cols = self.loader.feature_columns
        if feature_cols:
            for col in feature_cols:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            input_encoded = input_encoded[feature_cols]
        
        if self.loader.scaler:
            input_scaled = self.loader.scaler.transform(input_encoded)
        else:
            input_scaled = input_encoded.values
        
        return input_scaled
    
    def identify_risk_factors(self, employee: Dict) -> List[str]:
        """Identifie les facteurs de risque."""
        risk_factors = []
        
        if employee.get('OverTime') == 'Yes':
            risk_factors.append("⚠️ Heures supplémentaires (multiplie le risque par 3)")
        
        age = employee.get('Age', 0)
        if age < 25:
            risk_factors.append("⚠️ Jeune âge (<25 ans) - mobilité élevée")
        elif age > 55:
            risk_factors.append("✅ Âge expérimenté (>55 ans) - facteur de stabilité")
        
        years = employee.get('YearsAtCompany', 0)
        if years < 2:
            risk_factors.append("⚠️ Faible ancienneté (<2 ans) - période vulnérable")
        
        satisfaction = employee.get('JobSatisfaction', 0)
        if satisfaction <= 2:
            risk_factors.append("⚠️ Faible satisfaction au travail")
        
        wlb = employee.get('WorkLifeBalance', 0)
        if wlb <= 2:
            risk_factors.append("⚠️ Mauvais équilibre vie pro/perso")
        
        job_role = employee.get('JobRole', '')
        if 'Sales' in job_role:
            risk_factors.append("⚠️ Poste commercial - turnover historiquement élevé")
        
        distance = employee.get('DistanceFromHome', 0)
        if distance > 20:
            risk_factors.append("⚠️ Longue distance domicile-travail")
        
        return risk_factors
    
    def generate_recommendations(self, risk_factors: List[str], probability: float) -> List[str]:
        """Génère des recommandations."""
        recommendations = []
        
        if probability > 0.6:
            recommendations.append("🔴 ACTION PRIORITAIRE: Plan de rétention immédiat")
        
        for factor in risk_factors:
            if "Heures supplémentaires" in factor:
                recommendations.append("✓ Réévaluer la charge de travail")
                recommendations.append("✓ Envisager un recrutement supplémentaire")
            if "Jeune âge" in factor:
                recommendations.append("✓ Mettre en place un programme de mentorat")
            if "Faible ancienneté" in factor:
                recommendations.append("✓ Renforcer le programme d'intégration")
            if "Faible satisfaction" in factor or "Mauvais équilibre" in factor:
                recommendations.append("✓ Organiser un entretien individuel")
            if "Poste commercial" in factor:
                recommendations.append("✓ Revoir le package de rémunération variable")
            if "Longue distance" in factor:
                recommendations.append("✓ Proposer du télétravail")
        
        if not recommendations:
            recommendations.append("✓ Profil stable - continuer le suivi annuel")
        
        return list(dict.fromkeys(recommendations))[:5]
    
    def predict(self, employee_data: Dict) -> Dict:
        """Effectue une prédiction."""
        input_scaled = self.preprocess_input(employee_data)
        proba = self.loader.model.predict_proba(input_scaled)[0, 1]
        pred = 1 if proba >= 0.5 else 0
        
        if proba >= 0.6:
            risk_level = "élevé 🔴"
        elif proba >= 0.4:
            risk_level = "modéré 🟡"
        else:
            risk_level = "faible 🟢"
        
        confidence = max(proba, 1 - proba)
        risk_factors = self.identify_risk_factors(employee_data)
        recommendations = self.generate_recommendations(risk_factors, proba)
        
        return {
            'attrition_probability': round(proba, 4),
            'risk_level': risk_level,
            'prediction': pred,
            'prediction_label': 'Quittera' if pred == 1 else 'Restera',
            'confidence': round(confidence, 4),
            'risk_factors': risk_factors,
            'recommendations': recommendations,
            'model_version': self.loader.model_version,
            'timestamp': datetime.now().isoformat()
        }


# ============================================================================
# CRÉATION DE L'API
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie."""
    print("🚀 Démarrage de l'API...")
    yield
    print("🛑 Arrêt de l'API...")


app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service
service = PredictionService()


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Page d'accueil."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Attrition Prediction API</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .container {
                background: rgba(255,255,255,0.1);
                border-radius: 20px;
                padding: 30px;
                backdrop-filter: blur(10px);
            }
            h1 { color: white; }
            a { color: #ffd700; text-decoration: none; }
            .endpoints {
                margin-top: 20px;
                padding: 20px;
                background: rgba(0,0,0,0.2);
                border-radius: 10px;
            }
            code {
                background: rgba(0,0,0,0.3);
                padding: 2px 6px;
                border-radius: 4px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 Attrition Prediction API</h1>
            <p>API de prédiction du risque de départ des employés.</p>
            <div class="endpoints">
                <h2>📌 Endpoints:</h2>
                <ul>
                    <li><a href="/docs">📚 Documentation interactive (Swagger)</a></li>
                    <li><a href="/health">💚 Health Check</a></li>
                    <li><code>POST /predict</code> - Prédiction individuelle</li>
                    <li><code>POST /predict/batch</code> - Prédiction par lot</li>
                    <li><code>POST /predict/csv</code> - Prédiction via CSV</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Vérification de santé."""
    return HealthResponse(
        status="healthy",
        model_version=service.loader.model_version,
        model_loaded=service.loader._model is not None,
        api_version=API_VERSION,
        timestamp=datetime.now().isoformat()
    )


@app.get("/version")
async def get_version():
    """Retourne la version."""
    return {
        "api_version": API_VERSION,
        "model_version": service.loader.model_version,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(employee: EmployeeData):
    """Prédiction individuelle."""
    try:
        result = service.predict(employee.dict())
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(employees: List[EmployeeData]):
    """Prédiction par lot."""
    try:
        results = []
        for employee in employees:
            result = service.predict(employee.dict())
            results.append(PredictionResponse(**result))
        
        return BatchPredictionResponse(
            results=results,
            total_processed=len(results),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    """Prédiction depuis un fichier CSV."""
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        results = []
        for _, row in df.iterrows():
            employee_dict = row.to_dict()
            result = service.predict(employee_dict)
            results.append(result)
        
        results_df = pd.DataFrame(results)
        output = io.StringIO()
        results_df.to_csv(output, index=False)
        
        return JSONResponse(content={
            "status": "success",
            "total_processed": len(results),
            "csv_output": output.getvalue()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# EXÉCUTION
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )