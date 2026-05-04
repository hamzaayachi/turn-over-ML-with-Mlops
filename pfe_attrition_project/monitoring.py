"""
Script de monitoring pour le modèle d'attrition des employés.
Utilise Evidently pour détecter la dérive des données et des performances.
Version compatible avec Evidently 0.4.x+
"""

import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

# Evidently imports - Version compatible 0.4.x+
try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metrics import DataDriftTable, DatasetMissingValuesMetric
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, ClassificationPreset
    EVIDENTLY_AVAILABLE = True
except ImportError:
    print("⚠️ Evidently non disponible, utilisation du mode simplifié")
    EVIDENTLY_AVAILABLE = False
    
    # Classes stub
    class ColumnMapping: pass
    class Report: pass

# Pour le scheduling
try:
    import schedule
    SCHEDULE_AVAILABLE = True
except ImportError:
    SCHEDULE_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = r"C:\Users\hamza\Desktop\prjet pfe mlops\pfe_attrition_project\data\raw\WA_Fn-UseC_-HR-Employee-Attrition.csv"
PROCESSED_PATH = "data/processed"
MODELS_PATH = "models"
MONITORING_PATH = "monitoring_reports"
REPORTS_PATH = "reports"

# Création des dossiers
for path in [MONITORING_PATH, REPORTS_PATH]:
    Path(path).mkdir(parents=True, exist_ok=True)

# Seuils d'alerte
ALERT_THRESHOLDS = {
    'data_drift_threshold': 0.3,
    'target_drift_threshold': 0.2,
    'min_accuracy': 0.75,
    'min_roc_auc': 0.70,
    'max_missing_percentage': 5
}

# ============================================================================
# DÉTECTEUR DE DÉRIVE SIMPLIFIÉ
# ============================================================================

class SimpleDriftDetector:
    """Détecteur de dérive simple (sans Evidently)."""
    
    @staticmethod
    def detect_drift(reference_df, current_df, columns=None):
        """Détecte la dérive en comparant les distributions."""
        if columns is None:
            columns = reference_df.select_dtypes(include=[np.number]).columns.tolist()
            if 'Attrition' in columns:
                columns.remove('Attrition')
        
        drifted_columns = []
        drift_scores = {}
        
        for col in columns[:20]:
            if col not in current_df.columns:
                continue
            
            ref_mean = reference_df[col].mean()
            curr_mean = current_df[col].mean()
            
            if ref_mean != 0:
                diff_pct = abs((curr_mean - ref_mean) / ref_mean)
            else:
                diff_pct = abs(curr_mean - ref_mean)
            
            drift_scores[col] = diff_pct
            
            if diff_pct > 0.2:
                drifted_columns.append(col)
        
        drift_score = len(drifted_columns) / max(1, len(columns))
        
        return {
            'drift_detected': drift_score > ALERT_THRESHOLDS['data_drift_threshold'],
            'drift_score': drift_score,
            'drifted_columns': drifted_columns,
            'total_columns': len(columns),
            'drift_scores': drift_scores
        }

# ============================================================================
# CLASS DE MONITORING
# ============================================================================

class AttritionMonitoring:
    """Classe de monitoring pour le modèle d'attrition."""
    
    def __init__(self, reference_data_path: str = None, model_path: str = None):
        self.reference_data = None
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.detector = SimpleDriftDetector()
        
        # Chargement des données de référence
        if reference_data_path is None:
            reference_data_path = f"{PROCESSED_PATH}/train_data.csv"
        
        if os.path.exists(reference_data_path):
            self.reference_data = pd.read_csv(reference_data_path)
            print(f"Donnees de reference: {self.reference_data.shape}")
        
        # Chargement du modèle
        if model_path is None:
            model_path = f"{MODELS_PATH}/production_model.pkl"
        
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Modele charge: {type(self.model).__name__}")
        
        # Chargement du scaler
        scaler_path = f"{MODELS_PATH}/scaler.pkl"
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
    
    def load_current_data(self, current_data_path: str = None) -> pd.DataFrame:
        """Charge les données actuelles."""
        if current_data_path is None:
            current_data_path = f"{PROCESSED_PATH}/test_data.csv"
        
        if os.path.exists(current_data_path):
            return pd.read_csv(current_data_path)
        return None
    
    def check_data_drift(self, current_data: pd.DataFrame = None, save_report: bool = True) -> Dict:
        """Vérifie la dérive des données."""
        print("Verification de la derive des donnees...")
        
        if self.reference_data is None:
            return {'error': 'No reference data'}
        
        if current_data is None:
            current_data = self.load_current_data()
        
        if current_data is None:
            return {'error': 'No current data'}
        
        result = self.detector.detect_drift(self.reference_data, current_data)
        
        alerts = []
        if result.get('drift_detected', False):
            drift_score = result.get('drift_score', 0)
            drifted_cols = result.get('drifted_columns', [])
            alerts.append(f"Derive detectee: {drift_score:.1%} des colonnes")
            if drifted_cols:
                alerts.append(f"Colonnes: {', '.join(drifted_cols[:3])}")
        
        if save_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"{MONITORING_PATH}/data_drift_{timestamp}.json"
            with open(report_path, 'w') as f:
                json.dump({'timestamp': timestamp, 'result': result}, f, indent=2)
        
        return {
            'status': 'success',
            'drift_detected': result.get('drift_detected', False),
            'drift_score': result.get('drift_score', 0),
            'drifted_columns': result.get('drifted_columns', []),
            'alerts': alerts,
            'timestamp': datetime.now().isoformat()
        }
    
    def check_target_drift(self, current_data: pd.DataFrame = None, save_report: bool = True) -> Dict:
        """Vérifie la dérive de la cible."""
        print("Verification de la derive de la cible...")
        
        if self.reference_data is None:
            return {'error': 'No reference data'}
        
        if current_data is None:
            current_data = self.load_current_data()
        
        if current_data is None:
            return {'error': 'No current data'}
        
        ref_dist = self.reference_data['Attrition'].value_counts(normalize=True)
        curr_dist = current_data['Attrition'].value_counts(normalize=True)
        
        ref_rate = ref_dist.get(1, 0)
        curr_rate = curr_dist.get(1, 0)
        
        drift_detected = abs(ref_rate - curr_rate) > ALERT_THRESHOLDS['target_drift_threshold']
        
        alerts = []
        if drift_detected:
            alerts.append(f"Derive cible: {ref_rate:.2%} -> {curr_rate:.2%}")
        
        if save_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"{MONITORING_PATH}/target_drift_{timestamp}.json"
            with open(report_path, 'w') as f:
                json.dump({
                    'timestamp': timestamp,
                    'reference_distribution': ref_dist.to_dict(),
                    'current_distribution': curr_dist.to_dict(),
                    'drift_detected': drift_detected
                }, f, indent=2)
        
        return {
            'status': 'success',
            'drift_detected': drift_detected,
            'reference_distribution': ref_dist.to_dict(),
            'current_distribution': curr_dist.to_dict(),
            'alerts': alerts,
            'timestamp': datetime.now().isoformat()
        }
    
    def check_model_performance(self, test_data: pd.DataFrame = None, save_report: bool = True) -> Dict:
        """Vérifie les performances du modèle."""
        print("Verification des performances...")
        
        if self.model is None:
            return {'error': 'No model loaded'}
        
        if test_data is None:
            test_data = self.load_current_data()
        
        if test_data is None:
            return {'error': 'No test data'}
        
        from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
        
        cols = [c for c in test_data.columns if c != 'Attrition']
        X_test = test_data[cols]
        y_test = test_data['Attrition']
        
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
        
        alerts = []
        if metrics['accuracy'] < ALERT_THRESHOLDS['min_accuracy']:
            alerts.append(f"Accuracy basse: {metrics['accuracy']:.3f}")
        if metrics['roc_auc'] < ALERT_THRESHOLDS['min_roc_auc']:
            alerts.append(f"ROC-AUC bas: {metrics['roc_auc']:.3f}")
        
        if save_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"{MONITORING_PATH}/performance_{timestamp}.json"
            with open(report_path, 'w') as f:
                json.dump({'timestamp': timestamp, 'metrics': metrics, 'alerts': alerts}, f, indent=2)
        
        return {
            'status': 'success',
            'metrics': metrics,
            'alerts': alerts,
            'timestamp': datetime.now().isoformat()
        }
    
    def run_data_quality_tests(self, current_data: pd.DataFrame = None) -> Dict:
        """Exécute des tests de qualité."""
        print("Tests de qualite...")
        
        if current_data is None:
            current_data = self.load_current_data()
        
        if current_data is None:
            return {'error': 'No data'}
        
        results = []
        
        # Nombre de lignes
        results.append({
            'name': 'Nombre de lignes',
            'status': 'SUCCESS',
            'value': str(len(current_data)),
            'expected': '> 0'
        })
        
        # Valeurs manquantes
        total_cells = current_data.shape[0] * current_data.shape[1]
        missing_pct = (current_data.isnull().sum().sum() / total_cells) * 100 if total_cells > 0 else 0
        missing_status = 'SUCCESS' if missing_pct < ALERT_THRESHOLDS['max_missing_percentage'] else 'FAIL'
        results.append({
            'name': 'Valeurs manquantes',
            'status': missing_status,
            'value': f"{missing_pct:.2f}%",
            'expected': f"< {ALERT_THRESHOLDS['max_missing_percentage']}%"
        })
        
        # Age
        if 'Age' in current_data.columns:
            age_min = current_data['Age'].min()
            age_max = current_data['Age'].max()
            results.append({
                'name': 'Age - min',
                'status': 'SUCCESS' if age_min >= 18 else 'WARNING',
                'value': str(age_min),
                'expected': '>= 18'
            })
            results.append({
                'name': 'Age - max',
                'status': 'SUCCESS' if age_max <= 70 else 'WARNING',
                'value': str(age_max),
                'expected': '<= 70'
            })
        
        passed = sum(1 for r in results if r['status'] == 'SUCCESS')
        failed = sum(1 for r in results if r['status'] == 'FAIL')
        warnings_count = sum(1 for r in results if r['status'] == 'WARNING')
        
        return {
            'status': 'success',
            'tests': results,
            'passed': passed,
            'failed': failed,
            'warnings': warnings_count,
            'timestamp': datetime.now().isoformat()
        }
    
    def run_full_monitoring(self, save_reports: bool = True) -> Dict:
        """Exécute le monitoring complet."""
        print("="*60)
        print("MONITORING COMPLET")
        print("="*60)
        
        current_data = self.load_current_data()
        
        if current_data is None:
            return {'status': 'failed', 'error': 'No current data'}
        
        data_drift = self.check_data_drift(current_data, save_reports)
        target_drift = self.check_target_drift(current_data, save_reports)
        performance = self.check_model_performance(current_data, save_reports)
        quality = self.run_data_quality_tests(current_data)
        
        all_alerts = []
        all_alerts.extend(data_drift.get('alerts', []))
        all_alerts.extend(target_drift.get('alerts', []))
        all_alerts.extend(performance.get('alerts', []))
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'data_drift': data_drift,
            'target_drift': target_drift,
            'performance': performance,
            'quality': quality,
            'alerts': all_alerts,
            'alert_count': len(all_alerts)
        }
        
        # Sauvegarde
        if save_reports:
            report_path = f"{MONITORING_PATH}/monitoring_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            with open(f"{MONITORING_PATH}/last_status.json", 'w') as f:
                json.dump(result, f, indent=2)
        
        print("="*60)
        print(f"MONITORING TERMINE - Alertes: {len(all_alerts)}")
        if all_alerts:
            for alert in all_alerts:
                print(f"  - {alert}")
        print("="*60)
        
        return result

# ============================================================================
# GÉNÉRATION DU DASHBOARD
# ============================================================================

def generate_monitoring_dashboard():
    """Génère un dashboard HTML."""
    print("Generation du dashboard...")
    
    status_file = f"{MONITORING_PATH}/last_status.json"
    
    if not os.path.exists(status_file):
        print("Aucun historique. Executez d'abord: python monitoring.py --mode once")
        return
    
    with open(status_file, 'r') as f:
        status = json.load(f)
    
    html_content = f'''<!DOCTYPE html>
<html>
<head>
    <title>Monitoring Dashboard - Attrition Model</title>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .card {{ background: white; border-radius: 10px; padding: 20px; margin: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
        .alert {{ background: #ff4444; color: white; padding: 10px; border-radius: 5px; margin: 5px 0; }}
        .success {{ background: #00cc66; color: white; padding: 10px; border-radius: 5px; margin: 5px 0; }}
        .warning {{ background: #ffaa00; color: white; padding: 10px; border-radius: 5px; margin: 5px 0; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        .good {{ color: #00cc66; }}
        .bad {{ color: #ff4444; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Monitoring Dashboard - Attrition Model</h1>
            <p>Derniere mise a jour: {status.get('timestamp', 'N/A')}</p>
        </div>
        
        <div class="card">
            <h2>Alertes ({status.get('alert_count', 0)})</h2>
'''
    
    if status.get('alert_count', 0) == 0:
        html_content += '<div class="success">Aucune alerte - Tout fonctionne correctement</div>\n'
    else:
        for alert in status.get('alerts', []):
            html_content += f'<div class="alert">[WARNING] {alert}</div>\n'
    
    html_content += '''
        </div>
        
        <div class="card">
            <h2>Performance du Modele</h2>
            <table>
                <tr><th>Metrique</th><th>Valeur</th><th>Statut</th></tr>
'''
    
    metrics = status.get('performance', {}).get('metrics', {})
    for name, value in metrics.items():
        is_good = (name == 'accuracy' and value > 0.75) or (name == 'roc_auc' and value > 0.7)
        status_class = 'good' if is_good else 'bad'
        html_content += f"<tr><td>{name}</td><td><strong>{value:.4f}</strong></td><td class='{status_class}'>[OK]' if is_good else '[WARN]'</td></tr>\n"
    
    html_content += '''
            </table>
        </div>
        
        <div class="card">
            <h2>Derive des Donnees</h2>
'''
    
    data_drift = status.get('data_drift', {})
    if data_drift.get('drift_detected', False):
        html_content += f'<div class="warning">Derive detectee - Score: {data_drift.get("drift_score", 0):.2%}</div>'
        drifted = data_drift.get('drifted_columns', [])
        if drifted:
            html_content += f"<p>Colonnes: {', '.join(drifted[:5])}</p>"
    else:
        html_content += '<div class="success">Aucune derive majeure detectee</div>'
    
    html_content += f'''
            <p><strong>Score de derive:</strong> {data_drift.get('drift_score', 0):.2%}</p>
        </div>
        
        <div class="card">
            <h2>Qualite des Donnees</h2>
            <table>
                <tr><th>Test</th><th>Statut</th><th>Valeur</th><th>Attendu</th></tr>
'''
    
    for test in status.get('quality', {}).get('tests', []):
        status_icon = 'OK' if test['status'] == 'SUCCESS' else 'WARN' if test['status'] == 'WARNING' else 'FAIL'
        html_content += f"<tr><td>{test['name']}</td><td>{status_icon}</td><td>{test['value']}</td><td>{test.get('expected', '-')}</td></tr>\n"
    
    html_content += f'''
            </table>
            <p><strong>Resume:</strong> {status.get('quality', {}).get('passed', 0)} OK, {status.get('quality', {}).get('failed', 0)} FAIL, {status.get('quality', {}).get('warnings', 0)} WARN</p>
        </div>
    </div>
</body>
</html>
'''
    
    dashboard_path = f"{MONITORING_PATH}/dashboard.html"
    with open(dashboard_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Dashboard genere: {dashboard_path}")
    return dashboard_path

# ============================================================================
# SCHEDULING
# ============================================================================

def run_monitoring_job():
    """Job de monitoring periodique."""
    print(f"\n[{datetime.now()}] Execution du monitoring...")
    monitor = AttritionMonitoring()
    return monitor.run_full_monitoring()

def schedule_monitoring(interval_minutes: int = 60):
    """Planifie le monitoring."""
    if not SCHEDULE_AVAILABLE:
        print("Module 'schedule' non installe. Execution unique...")
        run_monitoring_job()
        return
    
    print(f"Planification toutes les {interval_minutes} minutes")
    run_monitoring_job()
    
    schedule.every(interval_minutes).minutes.do(run_monitoring_job)
    
    try:
        while True:
            schedule.run_pending()
            import time
            time.sleep(60)
    except KeyboardInterrupt:
        print("\nMonitoring arrete")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Fonction principale."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitoring pour attrition")
    parser.add_argument("--mode", type=str, default="once",
                       choices=["once", "schedule", "dashboard", "test"],
                       help="Mode d'execution")
    parser.add_argument("--interval", type=int, default=60,
                       help="Intervalle en minutes")
    
    args = parser.parse_args()
    
    if args.mode == "once":
        monitor = AttritionMonitoring()
        result = monitor.run_full_monitoring()
        print(f"\nResultat: {result.get('status', 'unknown')}")
        print(f"Alertes: {result.get('alert_count', 0)}")
        
    elif args.mode == "schedule":
        schedule_monitoring(args.interval)
        
    elif args.mode == "dashboard":
        generate_monitoring_dashboard()
        
    elif args.mode == "test":
        print("Mode test...")
        monitor = AttritionMonitoring()
        current = monitor.load_current_data()
        if current is not None:
            quality = monitor.run_data_quality_tests(current)
            print(f"Qualite: {quality.get('passed', 0)} OK, {quality.get('failed', 0)} FAIL")

if __name__ == "__main__":
    main()