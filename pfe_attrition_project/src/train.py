"""
Script d'entraînement du modèle Random Forest pour la prédiction d'attrition.
"""

import pandas as pd
import numpy as np
import pickle
import os
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """
    Classe pour l'entraînement du modèle d'attrition.
    """
    
    def __init__(self, config: dict = None):
        """
        Initialise le trainer.
        
        Args:
            config: Configuration du modèle
        """
        self.config = config or {
            'n_estimators': 300,
            'max_depth': 15,
            'min_samples_split': 8,
            'min_samples_leaf': 4,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }
        self.model = None
        self.metrics = {}
        
    def load_preprocessed_data(self, train_path: str = 'data/processed/train_data.csv',
                                test_path: str = 'data/processed/test_data.csv'):
        """
        Charge les données prétraitées.
        
        Args:
            train_path: Chemin des données d'entraînement
            test_path: Chemin des données de test
            
        Returns:
            Tuple (X_train, X_test, y_train, y_test)
        """
        print("📂 Chargement des données prétraitées...")
        
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        
        # Séparation features/target
        X_train = train_data.drop('Attrition', axis=1)
        y_train = train_data['Attrition']
        X_test = test_data.drop('Attrition', axis=1)
        y_test = test_data['Attrition']
        
        print(f"✅ Données chargées:")
        print(f"   - Train: {X_train.shape}")
        print(f"   - Test: {X_test.shape}")
        print(f"   - Distribution target (train): {y_train.value_counts(normalize=True).to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train, y_train):
        """
        Entraîne le modèle Random Forest.
        
        Args:
            X_train: Features d'entraînement
            y_train: Target d'entraînement
            
        Returns:
            Modèle entraîné
        """
        print("🤖 Entraînement du modèle Random Forest...")
        print(f"   - Paramètres: {self.config}")
        
        self.model = RandomForestClassifier(**self.config)
        self.model.fit(X_train, y_train)
        
        print("✅ Modèle entraîné avec succès!")
        return self.model
    
    def evaluate(self, X_test, y_test):
        """
        Évalue le modèle sur les données de test.
        
        Args:
            X_test: Features de test
            y_test: Target de test
            
        Returns:
            Dictionnaire des métriques
        """
        print("📊 Évaluation du modèle...")
        
        # Prédictions
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Métriques
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
        
        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        self.metrics['confusion_matrix'] = cm.tolist()
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.metrics['feature_importance'] = feature_importance.head(10).to_dict('records')
        
        print(f"   - Accuracy: {self.metrics['accuracy']:.4f}")
        print(f"   - Precision: {self.metrics['precision']:.4f}")
        print(f"   - Recall: {self.metrics['recall']:.4f}")
        print(f"   - F1-Score: {self.metrics['f1_score']:.4f}")
        print(f"   - ROC-AUC: {self.metrics['roc_auc']:.4f}")
        
        return self.metrics
    
    def save_model(self, output_dir: str = 'models'):
        """
        Sauvegarde le modèle entraîné.
        
        Args:
            output_dir: Répertoire de sauvegarde
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Sauvegarde du modèle
        model_path = f'{output_dir}/production_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"💾 Modèle sauvegardé: {model_path}")
        
        # Sauvegarde des métriques
        metrics_path = f'{output_dir}/model_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"💾 Métriques sauvegardées: {metrics_path}")
        
        # Sauvegarde de la configuration
        config_path = f'{output_dir}/model_config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"💾 Configuration sauvegardée: {config_path}")
    
    def log_to_mlflow(self, experiment_name: str = "employee_attrition"):
        """
        Enregistre les résultats dans MLflow.
        
        Args:
            experiment_name: Nom de l'expérience MLflow
        """
        try:
            mlflow.set_experiment(experiment_name)
            
            with mlflow.start_run() as run:
                # Log des paramètres
                mlflow.log_params(self.config)
                
                # Log des métriques
                for metric_name, metric_value in self.metrics.items():
                    if metric_name not in ['confusion_matrix', 'feature_importance']:
                        mlflow.log_metric(metric_name, metric_value)
                
                # Log du modèle
                mlflow.sklearn.log_model(self.model, "random_forest_model")
                
                # Log des artefacts
                if os.path.exists('models/model_metrics.json'):
                    mlflow.log_artifact('models/model_metrics.json')
                
                print(f"✅ Logged to MLflow - Run ID: {run.info.run_id}")
                return run.info.run_id
                
        except Exception as e:
            print(f"⚠️ MLflow logging failed: {e}")
            return None
    
    def run_training_pipeline(self, use_mlflow: bool = True):
        """
        Exécute le pipeline complet d'entraînement.
        
        Args:
            use_mlflow: Si True, enregistre dans MLflow
            
        Returns:
            Dictionnaire avec les résultats
        """
        print("="*60)
        print("🚀 DÉMARRAGE DU PIPELINE D'ENTRAÎNEMENT")
        print("="*60)
        
        # 1. Chargement des données
        X_train, X_test, y_train, y_test = self.load_preprocessed_data()
        
        # 2. Entraînement
        self.train(X_train, y_train)
        
        # 3. Évaluation
        metrics = self.evaluate(X_test, y_test)
        
        # 4. Sauvegarde
        self.save_model()
        
        # 5. MLflow
        run_id = None
        if use_mlflow:
            run_id = self.log_to_mlflow()
        
        # 6. Versioning
        self._create_version_file(run_id)
        
        print("="*60)
        print("✅ PIPELINE D'ENTRAÎNEMENT TERMINÉ")
        print("="*60)
        
        return {
            'model': self.model,
            'metrics': metrics,
            'config': self.config,
            'run_id': run_id
        }
    
    def _create_version_file(self, run_id: str = None):
        """
        Crée le fichier de version.
        """
        version_content = f"""MODEL_VERSION=1.0.0
CREATED_AT={datetime.now().isoformat()}
MODEL_TYPE=RandomForestClassifier
ACCURACY={self.metrics.get('accuracy', 0):.4f}
ROC_AUC={self.metrics.get('roc_auc', 0):.4f}
RUN_ID={run_id if run_id else 'local'}
STATUS=production
"""
        with open('models/version.txt', 'w') as f:
            f.write(version_content)
        print("💾 Fichier version.txt créé")


if __name__ == "__main__":
    # Entraînement du modèle
    trainer = ModelTrainer()
    result = trainer.run_training_pipeline(use_mlflow=True)
    
    print(f"\n📊 Résultats finaux:")
    print(f"   - Accuracy: {result['metrics']['accuracy']:.4f}")
    print(f"   - ROC-AUC: {result['metrics']['roc_auc']:.4f}")
    print(f"   - Run ID: {result['run_id']}")