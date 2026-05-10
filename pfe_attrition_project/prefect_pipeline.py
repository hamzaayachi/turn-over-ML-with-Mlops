"""
Pipeline Prefect pour l'orchestration du projet de prédiction d'attrition.
Automatise l'entraînement, l'évaluation et le déploiement du modèle.
Version compatible avec Prefect 2.x
"""

import os
import sys
import pickle
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import warnings

warnings.filterwarnings('ignore')

# Prefect imports - Version compatible Prefect 2.x
from prefect import flow, task
from prefect.logging import get_run_logger

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# ============================================================================
# CONFIGURATION
# ============================================================================

# Répertoire racine du projet basé sur l'emplacement du script.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Chemins robustes multi-environnements (local, Docker, Streamlit Cloud).
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "WA_Fn-UseC_-HR-Employee-Attrition.csv")
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed")
MODELS_PATH = os.path.join(BASE_DIR, "models")
REPORTS_PATH = os.path.join(BASE_DIR, "reports")

# Création des dossiers
for path in [PROCESSED_PATH, MODELS_PATH, REPORTS_PATH]:
    Path(path).mkdir(parents=True, exist_ok=True)

# ============================================================================
# TÂCHES DE CHARGEMENT DES DONNÉES
# ============================================================================

@task(name="load_data", retries=2, retry_delay_seconds=30)
def load_data() -> pd.DataFrame:
    """
    Charge les données brutes.
    """
    logger = get_run_logger()
    logger.info(f"📂 Chargement des données depuis {DATA_PATH}")
    
    try:
        df = pd.read_csv(DATA_PATH)
        logger.info(f"✅ Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")
        return df
    except FileNotFoundError:
        logger.error(f"❌ Fichier non trouvé: {DATA_PATH}")
        raise
    except Exception as e:
        logger.error(f"❌ Erreur de chargement: {e}")
        raise

@task(name="clean_data")
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les données.
    """
    logger = get_run_logger()
    logger.info("🧹 Nettoyage des données...")
    
    # Encodage de la cible
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
    
    # Suppression des colonnes inutiles
    columns_to_drop = ['EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours']
    existing_cols = [col for col in columns_to_drop if col in df.columns]
    
    if existing_cols:
        df = df.drop(existing_cols, axis=1)
        logger.info(f"   - Colonnes supprimées: {existing_cols}")
    
    # Vérification des valeurs manquantes
    missing = df.isnull().sum()
    if missing.sum() > 0:
        logger.warning(f"   - Valeurs manquantes: {missing[missing > 0].to_dict()}")
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].fillna(df[col].median())
    else:
        logger.info("   ✅ Aucune valeur manquante")
    
    logger.info(f"✅ Nettoyage terminé: {df.shape}")
    return df

@task(name="encode_categorical")
def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode les variables catégorielles.
    """
    logger = get_run_logger()
    logger.info("🔢 Encodage des variables catégorielles...")
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'Attrition' in categorical_cols:
        categorical_cols.remove('Attrition')
    
    logger.info(f"   - Variables catégorielles: {categorical_cols}")
    
    if categorical_cols:
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        logger.info(f"   - Nouvelles colonnes: {df_encoded.shape[1]}")
    else:
        df_encoded = df.copy()
    
    logger.info(f"✅ Encodage terminé: {df_encoded.shape}")
    return df_encoded

@task(name="split_data")
def split_data(df_encoded: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Divise les données en train/test.
    """
    logger = get_run_logger()
    logger.info("📊 Division des données...")
    
    X = df_encoded.drop('Attrition', axis=1)
    y = df_encoded['Attrition']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"   - Train: {X_train.shape[0]} lignes")
    logger.info(f"   - Test: {X_test.shape[0]} lignes")
    logger.info(f"   - Distribution target (train): {dict(y_train.value_counts(normalize=True))}")
    
    return X_train, X_test, y_train, y_test, X.columns

@task(name="normalize_data")
def normalize_data(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Normalise les données avec StandardScaler.
    """
    logger = get_run_logger()
    logger.info("📏 Normalisation des données...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"✅ Normalisation terminée")
    
    return X_train_scaled, X_test_scaled, scaler

@task(name="save_processed_data")
def save_processed_data(X_train, X_test, y_train, y_test, feature_columns, scaler):
    """
    Sauvegarde les données traitées.
    """
    logger = get_run_logger()
    logger.info("💾 Sauvegarde des données traitées...")
    
    # Création des dataframes
    train_df = pd.DataFrame(X_train, columns=feature_columns)
    train_df['Attrition'] = y_train.values
    test_df = pd.DataFrame(X_test, columns=feature_columns)
    test_df['Attrition'] = y_test.values
    
    # Sauvegarde
    train_df.to_csv(os.path.join(PROCESSED_PATH, "train_data.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_PATH, "test_data.csv"), index=False)
    
    # Sauvegarde du scaler
    with open(os.path.join(MODELS_PATH, "scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)
    
    # Sauvegarde des colonnes
    with open(os.path.join(MODELS_PATH, "feature_columns.pkl"), 'wb') as f:
        pickle.dump(list(feature_columns), f)
    
    logger.info(f"✅ Données sauvegardées dans {PROCESSED_PATH}")
    logger.info(f"✅ Scaler sauvegardé dans {MODELS_PATH}")
    
    return True

# ============================================================================
# TÂCHES D'ENTRAÎNEMENT
# ============================================================================

@task(name="train_model")
def train_model(X_train, y_train, config: Dict = None) -> RandomForestClassifier:
    """
    Entraîne le modèle Random Forest.
    """
    logger = get_run_logger()
    logger.info("🤖 Entraînement du modèle...")
    
    if config is None:
        config = {
            'n_estimators': 300,
            'max_depth': 15,
            'min_samples_split': 8,
            'min_samples_leaf': 4,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }
    
    logger.info(f"   - Paramètres: {config}")
    
    model = RandomForestClassifier(**config)
    model.fit(X_train, y_train)
    
    logger.info("✅ Modèle entraîné avec succès")
    return model

@task(name="evaluate_model")
def evaluate_model(model: RandomForestClassifier, X_test, y_test, feature_names) -> Dict:
    """
    Évalue le modèle et retourne les métriques.
    """
    logger = get_run_logger()
    logger.info("📊 Évaluation du modèle...")
    
    # Prédictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Métriques
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    metrics['top_features'] = feature_importance.head(10).to_dict('records')
    
    logger.info(f"   - Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"   - Precision: {metrics['precision']:.4f}")
    logger.info(f"   - Recall: {metrics['recall']:.4f}")
    logger.info(f"   - F1-Score: {metrics['f1_score']:.4f}")
    logger.info(f"   - ROC-AUC: {metrics['roc_auc']:.4f}")
    
    return metrics

@task(name="save_model")
def save_model(model: RandomForestClassifier, metrics: Dict) -> str:
    """
    Sauvegarde le modèle et ses métriques.
    """
    logger = get_run_logger()
    logger.info("💾 Sauvegarde du modèle...")
    
    # Sauvegarde du modèle
    model_path = os.path.join(MODELS_PATH, "production_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Sauvegarde des métriques
    metrics_path = os.path.join(MODELS_PATH, "model_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Sauvegarde de la version
    version_content = f"""MODEL_VERSION=1.0.0
CREATED_AT={datetime.now().isoformat()}
MODEL_TYPE=RandomForestClassifier
ACCURACY={metrics['accuracy']:.4f}
ROC_AUC={metrics['roc_auc']:.4f}
STATUS=production
"""
    with open(os.path.join(MODELS_PATH, "version.txt"), 'w') as f:
        f.write(version_content)
    
    logger.info(f"✅ Modèle sauvegardé: {model_path}")
    logger.info(f"✅ Métriques sauvegardées: {metrics_path}")
    
    return model_path

# ============================================================================
# TÂCHES DE VALIDATION
# ============================================================================

@task(name="validate_model")
def validate_model(metrics: Dict, min_accuracy: float = 0.75, min_auc: float = 0.70) -> bool:
    """
    Valide les performances du modèle.
    """
    logger = get_run_logger()
    logger.info("🔍 Validation du modèle...")
    
    accuracy = metrics['accuracy']
    roc_auc = metrics['roc_auc']
    
    logger.info(f"   - Accuracy: {accuracy:.4f} (requis: >{min_accuracy})")
    logger.info(f"   - ROC-AUC: {roc_auc:.4f} (requis: >{min_auc})")
    
    if accuracy >= min_accuracy and roc_auc >= min_auc:
        logger.info("✅ Modèle validé avec succès!")
        return True
    else:
        logger.warning(f"⚠️ Modèle ne répond pas aux critères")
        return False

@task(name="generate_report")
def generate_report(metrics: Dict, model_path: str) -> str:
    """
    Génère un rapport d'entraînement.
    """
    logger = get_run_logger()
    logger.info("📄 Génération du rapport...")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'model': {
            'type': 'RandomForestClassifier',
            'path': model_path
        },
        'metrics': {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'roc_auc': metrics['roc_auc']
        },
        'top_features': metrics.get('top_features', [])[:5]
    }
    
    report_path = os.path.join(REPORTS_PATH, f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"✅ Rapport généré: {report_path}")
    
    # Génération d'un rapport markdown
    md_report = f"""# Rapport d'Entraînement du Modèle

**Date**: {report['timestamp']}

## Métriques du Modèle

| Métrique | Valeur |
|----------|--------|
| Accuracy | {metrics['accuracy']:.4f} |
| Precision | {metrics['precision']:.4f} |
| Recall | {metrics['recall']:.4f} |
| F1-Score | {metrics['f1_score']:.4f} |
| ROC-AUC | {metrics['roc_auc']:.4f} |

## Top 5 Features Importantes

"""
    for i, feat in enumerate(report['top_features'][:5]):
        md_report += f"{i+1}. **{feat['feature']}**: {feat['importance']:.4f}\n"
    
    md_path = report_path.replace('.json', '.md')
    with open(md_path, 'w') as f:
        f.write(md_report)
    
    logger.info(f"✅ Rapport Markdown: {md_path}")
    
    return report_path

# ============================================================================
# FLOW PRINCIPAL
# ============================================================================

@flow(name="attrition_training_pipeline",
      log_prints=True)
def training_pipeline(force_retrain: bool = False) -> Dict[str, Any]:
    """
    Pipeline complet d'entraînement du modèle d'attrition.
    
    Args:
        force_retrain: Forcer l'entraînement même si un modèle existe
        
    Returns:
        Dictionnaire avec les résultats
    """
    logger = get_run_logger()
    logger.info("="*60)
    logger.info("🚀 DÉMARRAGE DU PIPELINE D'ENTRAÎNEMENT")
    logger.info("="*60)
    
    # Vérifier si un modèle existe déjà
    existing_model_path = os.path.join(MODELS_PATH, "production_model.pkl")
    if os.path.exists(existing_model_path) and not force_retrain:
        logger.info("📦 Un modèle existe déjà. Utilisez force_retrain=True pour le remplacer.")
        model_metrics_path = os.path.join(MODELS_PATH, "model_metrics.json")
        if os.path.exists(model_metrics_path):
            with open(model_metrics_path, 'r') as f:
                existing_metrics = json.load(f)
            return {
                'status': 'skipped',
                'message': 'Model already exists',
                'metrics': existing_metrics
            }
        else:
            return {
                'status': 'skipped',
                'message': 'Model already exists'
            }
    
    try:
        # 1. Chargement et prétraitement
        df_raw = load_data()
        df_clean = clean_data(df_raw)
        df_encoded = encode_categorical(df_clean)
        
        # 2. Division et normalisation
        X_train, X_test, y_train, y_test, feature_names = split_data(df_encoded)
        X_train_scaled, X_test_scaled, scaler = normalize_data(X_train, X_test)
        
        # 3. Sauvegarde des données traitées
        save_processed_data(X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler)
        
        # 4. Entraînement
        model = train_model(X_train_scaled, y_train)
        
        # 5. Évaluation
        metrics = evaluate_model(model, X_test_scaled, y_test, feature_names)
        
        # 6. Validation
        is_valid = validate_model(metrics)
        
        if not is_valid:
            logger.warning("⚠️ Modèle non validé, mais poursuite de l'exécution")
        
        # 7. Sauvegarde
        model_path = save_model(model, metrics)
        
        # 8. Génération du rapport
        report_path = generate_report(metrics, model_path)
        
        logger.info("="*60)
        logger.info("✅ PIPELINE TERMINÉ AVEC SUCCÈS")
        logger.info("="*60)
        
        return {
            'status': 'success',
            'model_path': model_path,
            'metrics': metrics,
            'is_valid': is_valid,
            'report_path': report_path
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur dans le pipeline: {e}")
        return {
            'status': 'failed',
            'error': str(e)
        }

# ============================================================================
# FLOW DE PRÉDICTION
# ============================================================================

@flow(name="prediction_flow")
def prediction_flow(employee_data: Dict) -> Dict:
    """
    Pipeline de prédiction pour un employé.
    
    Args:
        employee_data: Dictionnaire avec les caractéristiques de l'employé
        
    Returns:
        Résultat de la prédiction
    """
    logger = get_run_logger()
    logger.info("🎯 Prédiction pour un employé...")
    
    try:
        # Chargement du modèle et du scaler
        with open(os.path.join(MODELS_PATH, "production_model.pkl"), 'rb') as f:
            model = pickle.load(f)
        
        with open(os.path.join(MODELS_PATH, "scaler.pkl"), 'rb') as f:
            scaler = pickle.load(f)
        
        with open(os.path.join(MODELS_PATH, "feature_columns.pkl"), 'rb') as f:
            feature_columns = pickle.load(f)
        
        # Prétraitement
        input_df = pd.DataFrame([employee_data])
        input_encoded = pd.get_dummies(input_df)
        
        for col in feature_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        
        input_encoded = input_encoded[feature_columns]
        input_scaled = scaler.transform(input_encoded)
        
        # Prédiction
        proba = model.predict_proba(input_scaled)[0, 1]
        pred = 1 if proba >= 0.5 else 0
        
        # Niveau de risque
        if proba >= 0.6:
            risk = "élevé"
        elif proba >= 0.4:
            risk = "modéré"
        else:
            risk = "faible"
        
        result = {
            'status': 'success',
            'attrition_probability': float(proba),
            'prediction': int(pred),
            'prediction_label': 'Quittera' if pred == 1 else 'Restera',
            'risk_level': risk,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Prédiction: {result['prediction_label']} (probabilité: {proba:.2%})")
        return result
        
    except Exception as e:
        logger.error(f"❌ Erreur de prédiction: {e}")
        return {
            'status': 'failed',
            'error': str(e)
        }

# ============================================================================
# FLOW DE PRÉDICTION PAR LOT
# ============================================================================

@flow(name="batch_prediction_flow")
def batch_prediction_flow(employees_file: str) -> Dict:
    """
    Pipeline de prédiction pour un fichier CSV d'employés.
    
    Args:
        employees_file: Chemin du fichier CSV
        
    Returns:
        Résultats des prédictions
    """
    logger = get_run_logger()
    logger.info(f"📊 Prédiction par lot depuis {employees_file}...")
    
    try:
        df = pd.read_csv(employees_file)
        results = []
        
        for idx, row in df.iterrows():
            result = prediction_flow(row.to_dict())
            results.append(result)
            logger.info(f"   - Employé {idx+1}: {result['prediction_label']} ({result['attrition_probability']:.2%})")
        
        # Statistiques
        high_risk = sum(1 for r in results if r.get('risk_level') == 'élevé')
        moderate_risk = sum(1 for r in results if r.get('risk_level') == 'modéré')
        low_risk = sum(1 for r in results if r.get('risk_level') == 'faible')
        
        summary = {
            'status': 'success',
            'total_processed': len(results),
            'high_risk_count': high_risk,
            'moderate_risk_count': moderate_risk,
            'low_risk_count': low_risk,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Batch terminé: {len(results)} prédictions")
        logger.info(f"   - Risque élevé: {high_risk}")
        logger.info(f"   - Risque modéré: {moderate_risk}")
        logger.info(f"   - Risque faible: {low_risk}")
        
        return summary
        
    except Exception as e:
        logger.error(f"❌ Erreur batch: {e}")
        return {
            'status': 'failed',
            'error': str(e)
        }

# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    """
    Fonction principale.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Pipeline Prefect pour l'attrition")
    parser.add_argument("--mode", type=str, default="train",
                       choices=["train", "predict", "batch", "interactive"],
                       help="Mode d'exécution")
    parser.add_argument("--file", type=str, help="Fichier CSV pour batch prediction")
    parser.add_argument("--force", action="store_true", help="Forcer l'entraînement")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        print("🚀 Lancement du pipeline d'entraînement...")
        result = training_pipeline(force_retrain=args.force)
        print("\n📊 Résultat:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    elif args.mode == "interactive":
        print("Mode interactif - Entrez les données de l'employé:")
        print("-" * 50)
        
        try:
            employee_data = {
                'Age': int(input("Âge: ")),
                'Gender': input("Genre (Male/Female): "),
                'MaritalStatus': input("Situation familiale (Single/Married/Divorced): "),
                'Department': input("Département (Sales/Research & Development/Human Resources): "),
                'JobRole': input("Poste: "),
                'JobLevel': int(input("Niveau hiérarchique (1-5): ")),
                'YearsAtCompany': int(input("Années dans l'entreprise: ")),
                'MonthlyIncome': float(input("Revenu mensuel: ")),
                'JobSatisfaction': int(input("Satisfaction au travail (1-4): ")),
                'WorkLifeBalance': int(input("Équilibre vie pro/perso (1-4): ")),
                'OverTime': input("Heures supplémentaires (Yes/No): "),
                'DistanceFromHome': int(input("Distance domicile-travail (km): ")),
                'Education': 3,
                'EnvironmentSatisfaction': 3,
                'HourlyRate': 65,
                'JobInvolvement': 3,
                'MonthlyRate': 15000,
                'NumCompaniesWorked': 2,
                'PercentSalaryHike': 15,
                'PerformanceRating': 3,
                'RelationshipSatisfaction': 3,
                'StockOptionLevel': 1,
                'TotalWorkingYears': 10,
                'TrainingTimesLastYear': 2,
                'YearsInCurrentRole': 3,
                'YearsSinceLastPromotion': 2,
                'YearsWithCurrManager': 3,
                'BusinessTravel': 'Travel_Rarely',
                'EducationField': 'Life Sciences'
            }
            
            result = prediction_flow(employee_data)
            print("\n📊 Résultat de la prédiction:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
        except Exception as e:
            print(f"❌ Erreur: {e}")
        
    elif args.mode == "batch":
        if not args.file:
            print("❌ Veuillez spécifier --file pour le batch prediction")
            sys.exit(1)
        if not os.path.exists(args.file):
            print(f"❌ Fichier non trouvé: {args.file}")
            sys.exit(1)
        result = batch_prediction_flow(args.file)
        print("\n📊 Résultat du batch:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    elif args.mode == "predict":
        # Mode prédiction rapide avec exemple
        example_data = {
            'Age': 35,
            'Gender': 'Male',
            'MaritalStatus': 'Single',
            'Department': 'Sales',
            'JobRole': 'Sales Executive',
            'JobLevel': 2,
            'YearsAtCompany': 5,
            'MonthlyIncome': 5000,
            'JobSatisfaction': 3,
            'WorkLifeBalance': 3,
            'OverTime': 'No',
            'DistanceFromHome': 5,
            'Education': 3,
            'EnvironmentSatisfaction': 3,
            'HourlyRate': 65,
            'JobInvolvement': 3,
            'MonthlyRate': 15000,
            'NumCompaniesWorked': 2,
            'PercentSalaryHike': 15,
            'PerformanceRating': 3,
            'RelationshipSatisfaction': 3,
            'StockOptionLevel': 1,
            'TotalWorkingYears': 10,
            'TrainingTimesLastYear': 2,
            'YearsInCurrentRole': 3,
            'YearsSinceLastPromotion': 2,
            'YearsWithCurrManager': 3,
            'BusinessTravel': 'Travel_Rarely',
            'EducationField': 'Life Sciences'
        }
        print("🔮 Prédiction pour un employé exemple...")
        result = prediction_flow(example_data)
        print(json.dumps(result, indent=2, ensure_ascii=False))

# ============================================================================
# EXÉCUTION
# ============================================================================

if __name__ == "__main__":
    main()