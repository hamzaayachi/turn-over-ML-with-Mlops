"""
Script d'évaluation avancée du modèle d'attrition.
"""

import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve, 
                             confusion_matrix, classification_report)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """
    Classe pour l'évaluation avancée du modèle.
    """
    
    def __init__(self, model_path: str = 'models/production_model.pkl',
                 scaler_path: str = 'models/scaler.pkl'):
        """
        Initialise l'évaluateur.
        
        Args:
            model_path: Chemin du modèle
            scaler_path: Chemin du scaler
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self.metrics = {}
        
    def load_model(self):
        """Charge le modèle et le scaler."""
        print("📦 Chargement du modèle...")
        
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(self.scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        print(f"✅ Modèle chargé: {type(self.model).__name__}")
        return self
    
    def load_test_data(self, test_path: str = 'data/processed/test_data.csv'):
        """
        Charge les données de test.
        
        Args:
            test_path: Chemin des données de test
            
        Returns:
            Tuple (X_test, y_test)
        """
        test_data = pd.read_csv(test_path)
        X_test = test_data.drop('Attrition', axis=1)
        y_test = test_data['Attrition']
        
        print(f"✅ Données de test chargées: {X_test.shape}")
        return X_test, y_test
    
    def evaluate_basic_metrics(self, y_true, y_pred, y_proba):
        """
        Calcule les métriques de base.
        
        Returns:
            Dictionnaire des métriques
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_proba)
        }
        
        print("\n📊 Métriques de base:")
        for name, value in metrics.items():
            print(f"   - {name}: {value:.4f}")
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path: str = 'reports/confusion_matrix.png'):
        """
        Génère et sauvegarde la matrice de confusion.
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Resté', 'Quitté'],
                    yticklabels=['Resté', 'Quitté'])
        plt.title('Matrice de Confusion - Modèle d\'Attrition')
        plt.xlabel('Prédiction')
        plt.ylabel('Réel')
        
        pip install docker docker-compose.makedirs('reports', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Matrice de confusion sauvegardée: {save_path}")
        return cm
    
    def plot_roc_curve(self, y_true, y_proba, save_path: str = 'reports/roc_curve.png'):
        """
        Génère et sauvegarde la courbe ROC.
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, 'b-', label=f'Random Forest (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'r--', label='Modèle aléatoire')
        plt.xlabel('Taux de faux positifs (FPR)')
        plt.ylabel('Taux de vrais positifs (TPR)')
        plt.title('Courbe ROC - Modèle d\'Attrition')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        os.makedirs('reports', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Courbe ROC sauvegardée: {save_path}")
        return fpr, tpr, auc
    
    def plot_calibration_curve(self, y_true, y_proba, save_path: str = 'reports/calibration_curve.png'):
        """
        Génère la courbe de calibration.
        """
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
        
        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, 'b-', marker='o', label='Modèle')
        plt.plot([0, 1], [0, 1], 'r--', label='Parfaitement calibré')
        plt.xlabel('Probabilité prédite moyenne')
        plt.ylabel('Fréquence réelle positive')
        plt.title('Courbe de Calibration')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Courbe de calibration sauvegardée: {save_path}")
    
    def plot_feature_importance(self, feature_names, top_n: int = 20, 
                                save_path: str = 'reports/feature_importance.png'):
        """
        Génère le graphique d'importance des features.
        """
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(top_n), importances[indices][::-1])
        plt.yticks(range(top_n), [feature_names[i] for i in indices[::-1]])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Features les Plus Importantes')
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Feature importance sauvegardée: {save_path}")
        
        # Affichage du top 10
        print("\n🏆 Top 10 Features les plus importantes:")
        for i in range(min(10, top_n)):
            print(f"   {i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
    def generate_classification_report(self, y_true, y_pred, save_path: str = 'reports/classification_report.txt'):
        """
        Génère le rapport de classification détaillé.
        """
        report = classification_report(y_true, y_pred, 
                                      target_names=['Resté (0)', 'Quitté (1)'],
                                      output_dict=True)
        
        # Sauvegarde en JSON
        with open(save_path.replace('.txt', '.json'), 'w') as f:
            json.dump(report, f, indent=2)
        
        # Sauvegarde en texte
        with open(save_path, 'w') as f:
            f.write(classification_report(y_true, y_pred, 
                                         target_names=['Resté (0)', 'Quitté (1)']))
        
        print(f"📊 Rapport de classification sauvegardé: {save_path}")
        return report
    
    def calculate_threshold_metrics(self, y_true, y_proba):
        """
        Calcule les métriques pour différents seuils.
        
        Returns:
            DataFrame avec les métriques par seuil
        """
        thresholds = np.arange(0.1, 0.9, 0.05)
        results = []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            results.append({
                'threshold': threshold,
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred),
                'recall': recall_score(y_true, y_pred),
                'f1': f1_score(y_true, y_pred)
            })
        
        df_results = pd.DataFrame(results)
        
        # Trouver le meilleur seuil (max F1)
        best_threshold = df_results.loc[df_results['f1'].idxmax(), 'threshold']
        
        print(f"\n🎯 Meilleur seuil (max F1): {best_threshold:.2f}")
        
        # Graphique
        plt.figure(figsize=(10, 6))
        plt.plot(df_results['threshold'], df_results['accuracy'], 'b-', label='Accuracy')
        plt.plot(df_results['threshold'], df_results['precision'], 'g-', label='Precision')
        plt.plot(df_results['threshold'], df_results['recall'], 'r-', label='Recall')
        plt.plot(df_results['threshold'], df_results['f1'], 'm-', label='F1-Score')
        plt.axvline(x=best_threshold, color='k', linestyle='--', label=f'Best threshold = {best_threshold:.2f}')
        plt.xlabel('Seuil de décision')
        plt.ylabel('Score')
        plt.title('Métriques par Seuil de Décision')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        os.makedirs('reports', exist_ok=True)
        plt.savefig('reports/threshold_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        return df_results, best_threshold
    
    def run_full_evaluation(self):
        """
        Exécute l'évaluation complète du modèle.
        """
        print("="*60)
        print("🚀 DÉMARRAGE DE L'ÉVALUATION DU MODÈLE")
        print("="*60)
        
        # 1. Chargement
        self.load_model()
        X_test, y_test = self.load_test_data()
        
        # 2. Prédictions
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        # 3. Métriques de base
        basic_metrics = self.evaluate_basic_metrics(y_test, y_pred, y_proba)
        
        # 4. Visualisations
        self.plot_confusion_matrix(y_test, y_pred)
        self.plot_roc_curve(y_test, y_proba)
        self.plot_calibration_curve(y_test, y_proba)
        self.plot_feature_importance(X_test.columns)
        self.generate_classification_report(y_test, y_pred)
        
        # 5. Analyse des seuils
        threshold_df, best_threshold = self.calculate_threshold_metrics(y_test, y_proba)
        
        # 6. Résumé
        print("\n" + "="*60)
        print("📊 RÉSUMÉ DE L'ÉVALUATION")
        print("="*60)
        print(f"✅ Accuracy: {basic_metrics['accuracy']:.4f}")
        print(f"✅ Precision: {basic_metrics['precision']:.4f}")
        print(f"✅ Recall: {basic_metrics['recall']:.4f}")
        print(f"✅ F1-Score: {basic_metrics['f1_score']:.4f}")
        print(f"✅ ROC-AUC: {basic_metrics['roc_auc']:.4f}")
        print(f"🎯 Meilleur seuil: {best_threshold:.2f}")
        
        # Sauvegarde du résumé
        summary = {
            'metrics': basic_metrics,
            'best_threshold': float(best_threshold),
            'model_type': type(self.model).__name__,
            'n_features': X_test.shape[1],
            'n_test_samples': len(y_test)
        }
        
        with open('reports/evaluation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n✅ Évaluation terminée! Les rapports sont dans le dossier 'reports/'")
        
        return summary


if __name__ == "__main__":
    evaluator = ModelEvaluator()
    results = evaluator.run_full_evaluation()