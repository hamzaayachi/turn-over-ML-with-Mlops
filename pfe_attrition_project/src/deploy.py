"""
Script de déploiement du modèle d'attrition en production.
"""

import os
import sys
import json
import pickle
import shutil
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional
import requests
import yaml

class ModelDeployer:
    """
    Classe pour le déploiement du modèle en production.
    """
    
    def __init__(self, models_dir: str = 'models',
                 production_dir: str = 'production'):
        """
        Initialise le déployeur.
        
        Args:
            models_dir: Répertoire des modèles
            production_dir: Répertoire de production
        """
        self.models_dir = models_dir
        self.production_dir = production_dir
        self.current_version = None
        
    def check_model_artifacts(self) -> bool:
        """
        Vérifie que tous les artefacts nécessaires sont présents.
        
        Returns:
            True si tous les fichiers sont présents
        """
        required_files = [
            f'{self.models_dir}/production_model.pkl',
            f'{self.models_dir}/scaler.pkl',
            f'{self.models_dir}/feature_columns.pkl',
            f'{self.models_dir}/version.txt',
            f'{self.models_dir}/model_metrics.json'
        ]
        
        missing = []
        for file in required_files:
            if not os.path.exists(file):
                missing.append(file)
        
        if missing:
            print(f"❌ Fichiers manquants: {missing}")
            return False
        
        print("✅ Tous les artefacts sont présents")
        return True
    
    def validate_model_performance(self, min_accuracy: float = 0.80) -> bool:
        """
        Valide les performances du modèle avant déploiement.
        
        Args:
            min_accuracy: Accuracy minimale requise
            
        Returns:
            True si le modèle est valide
        """
        with open(f'{self.models_dir}/model_metrics.json', 'r') as f:
            metrics = json.load(f)
        
        accuracy = metrics.get('accuracy', 0)
        roc_auc = metrics.get('roc_auc', 0)
        
        print(f"📊 Validation du modèle:")
        print(f"   - Accuracy: {accuracy:.4f} (requis: >{min_accuracy})")
        print(f"   - ROC-AUC: {roc_auc:.4f} (requis: >0.75)")
        
        if accuracy >= min_accuracy and roc_auc >= 0.75:
            print("✅ Modèle validé avec succès!")
            return True
        else:
            print("❌ Modèle ne répond pas aux critères de performance")
            return False
    
    def backup_current_production(self):
        """
        Sauvegarde le modèle actuellement en production.
        """
        if os.path.exists(self.production_dir):
            backup_dir = f'{self.production_dir}_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            shutil.copytree(self.production_dir, backup_dir)
            print(f"💾 Backup créé: {backup_dir}")
            return backup_dir
        return None
    
    def deploy_to_production(self):
        """
        Déploie le modèle en production.
        """
        print("🚀 Déploiement du modèle en production...")
        
        # 1. Création du dossier production
        os.makedirs(self.production_dir, exist_ok=True)
        
        # 2. Backup de l'ancienne version
        self.backup_current_production()
        
        # 3. Copie des fichiers
        files_to_copy = [
            'production_model.pkl',
            'scaler.pkl',
            'feature_columns.pkl',
            'version.txt',
            'model_metrics.json',
            'model_config.json'
        ]
        
        for file in files_to_copy:
            src = f'{self.models_dir}/{file}'
            if os.path.exists(src):
                shutil.copy2(src, f'{self.production_dir}/{file}')
                print(f"   - Copié: {file}")
        
        # 4. Lecture de la version
        with open(f'{self.production_dir}/version.txt', 'r') as f:
            for line in f:
                if line.startswith('MODEL_VERSION='):
                    self.current_version = line.split('=')[1].strip()
                    break
        
        # 5. Création du manifeste de déploiement
        manifest = {
            'deployed_version': self.current_version,
            'deployed_at': datetime.now().isoformat(),
            'environment': 'production',
            'deployed_by': os.environ.get('USER', 'unknown'),
            'files': files_to_copy
        }
        
        with open(f'{self.production_dir}/deployment_manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"✅ Modèle déployé avec succès! Version: {self.current_version}")
        return manifest
    
    def test_production_model(self, test_input: Dict = None):
        """
        Teste le modèle déployé avec des exemples.
        
        Args:
            test_input: Données de test (optionnel)
            
        Returns:
            Résultats des tests
        """
        print("🧪 Test du modèle en production...")
        
        # Chargement du modèle déployé
        with open(f'{self.production_dir}/production_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open(f'{self.production_dir}/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open(f'{self.production_dir}/feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        
        # Exemple de test par défaut
        if test_input is None:
            import numpy as np
            test_input = np.random.randn(1, len(feature_columns))
            test_scaled = scaler.transform(test_input)
        else:
            test_scaled = scaler.transform(test_input)
        
        # Prédiction
        proba = model.predict_proba(test_scaled)[0, 1]
        pred = 1 if proba >= 0.5 else 0
        
        print(f"   - Prédiction: {'Quittera' if pred == 1 else 'Restera'}")
        print(f"   - Probabilité: {proba:.2%}")
        
        return {'prediction': pred, 'probability': proba}
    
    def rollback(self, backup_path: str = None):
        """
        Effectue un rollback vers une version précédente.
        
        Args:
            backup_path: Chemin du backup à restaurer
        """
        if backup_path is None:
            # Trouver le dernier backup
            backups = [d for d in os.listdir('.') if d.startswith(f'{self.production_dir}_backup_')]
            if not backups:
                print("❌ Aucun backup trouvé")
                return False
            
            backup_path = max(backups, key=lambda x: x)
        
        print(f"🔄 Rollback vers: {backup_path}")
        
        # Restauration
        if os.path.exists(self.production_dir):
            shutil.rmtree(self.production_dir)
        
        shutil.copytree(backup_path, self.production_dir)
        
        print("✅ Rollback effectué avec succès!")
        return True
    
    def deploy_fastapi(self, host: str = '0.0.0.0', port: int = 8000, background: bool = False):
        """
        Déploie l'API FastAPI.
        
        Args:
            host: Hôte
            port: Port
            background: Exécuter en arrière-plan
        """
        print(f"🌐 Déploiement de l'API FastAPI sur {host}:{port}...")
        
        cmd = f"uvicorn api:app --host {host} --port {port}"
        
        if background:
            subprocess.Popen(cmd, shell=True)
            print(f"✅ API démarrée en arrière-plan sur http://{host}:{port}")
        else:
            subprocess.run(cmd, shell=True)
    
    def deploy_streamlit(self, host: str = '0.0.0.0', port: int = 8501, background: bool = False):
        """
        Déploie le dashboard Streamlit.
        
        Args:
            host: Hôte
            port: Port
            background: Exécuter en arrière-plan
        """
        print(f"📊 Déploiement du dashboard Streamlit sur {host}:{port}...")
        
        cmd = f"streamlit run app.py --server.port {port} --server.address {host}"
        
        if background:
            subprocess.Popen(cmd, shell=True)
            print(f"✅ Dashboard démarré en arrière-plan sur http://{host}:{port}")
        else:
            subprocess.run(cmd, shell=True)
    
    def deploy_with_docker(self):
        """
        Déploie l'application avec Docker Compose.
        """
        print("🐳 Déploiement avec Docker Compose...")
        
        # Vérification de Docker
        try:
            subprocess.run(['docker', '--version'], check=True, capture_output=True)
            subprocess.run(['docker-compose', '--version'], check=True, capture_output=True)
        except:
            print("❌ Docker ou Docker Compose non trouvé")
            return False
        
        # Lancement des services
        subprocess.run(['docker-compose', 'up', '-d', '--build'])
        
        print("✅ Services Docker démarrés!")
        print("   - API: http://localhost:8000")
        print("   - Dashboard: http://localhost:8501")
        print("   - MLflow: http://localhost:5000")
        
        return True
    
    def create_deployment_report(self):
        """
        Crée un rapport de déploiement.
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'version': self.current_version,
            'status': 'deployed',
            'production_path': self.production_dir,
            'model_files': os.listdir(self.production_dir) if os.path.exists(self.production_dir) else []
        }
        
        # Ajout des métriques
        if os.path.exists(f'{self.production_dir}/model_metrics.json'):
            with open(f'{self.production_dir}/model_metrics.json', 'r') as f:
                report['metrics'] = json.load(f)
        
        # Sauvegarde du rapport
        os.makedirs('reports', exist_ok=True)
        with open('reports/deployment_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Rapport de déploiement: reports/deployment_report.json")
        return report
    
    def run_full_deployment(self, deploy_api: bool = True, 
                           deploy_dashboard: bool = True,
                           use_docker: bool = False):
        """
        Exécute le déploiement complet.
        
        Args:
            deploy_api: Déployer l'API
            deploy_dashboard: Déployer le dashboard
            use_docker: Utiliser Docker
        """
        print("="*60)
        print("🚀 DÉMARRAGE DU DÉPLOIEMENT COMPLET")
        print("="*60)
        
        # 1. Vérification des artefacts
        if not self.check_model_artifacts():
            print("❌ Déploiement annulé: artefacts manquants")
            return False
        
        # 2. Validation des performances
        if not self.validate_model_performance():
            print("⚠️ Performance insuffisante, continuer quand même?")
            response = input("Continuer? (y/n): ")
            if response.lower() != 'y':
                return False
        
        # 3. Déploiement en production
        manifest = self.deploy_to_production()
        
        # 4. Test du modèle déployé
        self.test_production_model()
        
        # 5. Déploiement des services
        if use_docker:
            self.deploy_with_docker()
        else:
            if deploy_api:
                self.deploy_fastapi(background=True)
            if deploy_dashboard:
                self.deploy_streamlit(background=True)
        
        # 6. Rapport de déploiement
        report = self.create_deployment_report()
        
        print("="*60)
        print("✅ DÉPLOIEMENT COMPLETÉ AVEC SUCCÈS!")
        print("="*60)
        print(f"📦 Version: {self.current_version}")
        print(f"📁 Production path: {self.production_dir}")
        
        return report


class KubernetesDeployer(ModelDeployer):
    """
    Déploiement sur Kubernetes (extension).
    """
    
    def create_k8s_manifest(self):
        """Crée les manifests Kubernetes."""
        
        deployment = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {'name': 'attrition-predictor', 'namespace': 'production'},
            'spec': {
                'replicas': 3,
                'selector': {'matchLabels': {'app': 'attrition-predictor'}},
                'template': {
                    'metadata': {'labels': {'app': 'attrition-predictor'}},
                    'spec': {
                        'containers': [{
                            'name': 'api',
                            'image': f'attrition-predictor:{self.current_version}',
                            'ports': [{'containerPort': 8000}],
                            'env': [
                                {'name': 'MODEL_VERSION', 'value': self.current_version}
                            ]
                        }]
                    }
                }
            }
        }
        
        service = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {'name': 'attrition-predictor-service'},
            'spec': {
                'selector': {'app': 'attrition-predictor'},
                'ports': [{'port': 80, 'targetPort': 8000}],
                'type': 'LoadBalancer'
            }
        }
        
        os.makedirs('k8s', exist_ok=True)
        
        with open('k8s/deployment.yaml', 'w') as f:
            yaml.dump(deployment, f)
        
        with open('k8s/service.yaml', 'w') as f:
            yaml.dump(service, f)
        
        print("✅ Manifests Kubernetes créés dans le dossier 'k8s/'")
        
    def deploy_to_kubernetes(self):
        """Déploie sur Kubernetes."""
        self.create_k8s_manifest()
        
        subprocess.run(['kubectl', 'apply', '-f', 'k8s/deployment.yaml'])
        subprocess.run(['kubectl', 'apply', '-f', 'k8s/service.yaml'])
        
        print("✅ Déploiement Kubernetes effectué!")


if __name__ == "__main__":
    # Déploiement standard
    deployer = ModelDeployer()
    result = deployer.run_full_deployment(deploy_api=True, deploy_dashboard=True)
    
    # Pour déployer sur Kubernetes (optionnel)
    # k8s_deployer = KubernetesDeployer()
    # k8s_deployer.deploy_to_kubernetes()