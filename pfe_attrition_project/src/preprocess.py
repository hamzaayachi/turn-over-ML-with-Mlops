"""
Script de prétraitement des données pour le modèle d'attrition.
Gère le nettoyage, l'encodage et la normalisation des données.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os
import json
from typing import Tuple, Dict, Any

class DataPreprocessor:
    """
    Classe pour le prétraitement des données d'attrition.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialise le préprocesseur.
        
        Args:
            config: Configuration optionnelle
        """
        self.config = config or {}
        self.scaler = None
        self.feature_columns = None
        self.encoded_columns = None
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Charge les données depuis un fichier CSV.
        
        Args:
            filepath: Chemin vers le fichier CSV
            
        Returns:
            DataFrame chargé
        """
        print(f"📂 Chargement des données depuis {filepath}...")
        df = pd.read_csv(filepath)
        print(f"✅ Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoie les données (valeurs manquantes, colonnes inutiles).
        
        Args:
            df: DataFrame à nettoyer
            
        Returns:
            DataFrame nettoyé
        """
        print("🧹 Nettoyage des données...")
        
        # Encodage de la variable cible
        if 'Attrition' in df.columns:
            df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
        
        # Suppression des colonnes redondantes
        columns_to_drop = ['EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours']
        existing_cols = [col for col in columns_to_drop if col in df.columns]
        
        if existing_cols:
            df = df.drop(existing_cols, axis=1)
            print(f"   - Colonnes supprimées: {existing_cols}")
        
        # Vérification des valeurs manquantes
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"   ⚠️ Valeurs manquantes détectées: {missing[missing > 0].to_dict()}")
            # Imputation par la médiane pour les numériques
            for col in df.select_dtypes(include=[np.number]).columns:
                df[col] = df[col].fillna(df[col].median())
        else:
            print("   ✅ Aucune valeur manquante")
        
        print(f"✅ Nettoyage terminé: {df.shape}")
        return df
    
    def encode_categorical(self, df: pd.DataFrame, drop_first: bool = True) -> pd.DataFrame:
        """
        Encode les variables catégorielles en one-hot encoding.
        
        Args:
            df: DataFrame à encoder
            drop_first: Si True, drop la première colonne pour éviter la multicolinéarité
            
        Returns:
            DataFrame encodé
        """
        print("🔢 Encodage des variables catégorielles...")
        
        # Identification des colonnes catégorielles
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if 'Attrition' in categorical_cols:
            categorical_cols.remove('Attrition')
        
        print(f"   - Variables catégorielles: {categorical_cols}")
        
        if categorical_cols:
            # One-hot encoding
            df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=drop_first)
            
            # Sauvegarde des noms des colonnes encodées
            self.encoded_columns = [col for col in df_encoded.columns if any(cat in col for cat in categorical_cols)]
            print(f"   - Colonnes créées: {len(self.encoded_columns)}")
        else:
            df_encoded = df.copy()
        
        # Séparation features et target
        if 'Attrition' in df_encoded.columns:
            self.feature_columns = [col for col in df_encoded.columns if col != 'Attrition']
        
        print(f"✅ Encodage terminé: {df_encoded.shape}")
        return df_encoded
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.2, 
                   random_state: int = 42) -> Tuple:
        """
        Divise les données en train/test.
        
        Args:
            X: Features
            y: Target
            test_size: Proportion pour le test
            random_state: Seed pour reproductibilité
            
        Returns:
            Tuple (X_train, X_test, y_train, y_test)
        """
        print("📊 Division des données...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y
        )
        
        print(f"   - Train: {X_train.shape[0]} lignes")
        print(f"   - Test: {X_test.shape[0]} lignes")
        print(f"   - Distribution target (train): {y_train.value_counts(normalize=True).to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def normalize_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None) -> Tuple:
        """
        Normalise les données avec StandardScaler.
        
        Args:
            X_train: Données d'entraînement
            X_test: Données de test (optionnel)
            
        Returns:
            Tuple (X_train_scaled, X_test_scaled) ou X_train_scaled seul
        """
        print("📏 Normalisation des données...")
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        result = [X_train_scaled]
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            result.append(X_test_scaled)
        
        print(f"✅ Normalisation terminée")
        return tuple(result)
    
    def save_artifacts(self, output_dir: str = 'models'):
        """
        Sauvegarde les artefacts du préprocesseur.
        
        Args:
            output_dir: Répertoire de sauvegarde
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Sauvegarde du scaler
        if self.scaler:
            with open(f'{output_dir}/scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"💾 Scaler sauvegardé: {output_dir}/scaler.pkl")
        
        # Sauvegarde des colonnes
        if self.feature_columns:
            with open(f'{output_dir}/feature_columns.pkl', 'wb') as f:
                pickle.dump(self.feature_columns, f)
            print(f"💾 Colonnes sauvegardées: {output_dir}/feature_columns.pkl")
        
        # Sauvegarde de la configuration
        config_info = {
            'feature_columns': self.feature_columns,
            'encoded_columns': self.encoded_columns,
            'n_features': len(self.feature_columns) if self.feature_columns else 0
        }
        
        with open(f'{output_dir}/preprocess_config.json', 'w') as f:
            json.dump(config_info, f, indent=2)
    
    def run_pipeline(self, input_file: str, output_dir: str = 'models') -> Dict:
        """
        Exécute le pipeline complet de prétraitement.
        
        Args:
            input_file: Chemin du fichier d'entrée
            output_dir: Répertoire de sortie
            
        Returns:
            Dictionnaire avec les données traitées
        """
        print("="*60)
        print("🚀 DÉMARRAGE DU PIPELINE DE PRÉTRAITEMENT")
        print("="*60)
        
        # 1. Chargement
        df = self.load_data(input_file)
        
        # 2. Nettoyage
        df_clean = self.clean_data(df)
        
        # 3. Encodage
        df_encoded = self.encode_categorical(df_clean)
        
        # 4. Séparation features/target
        X = df_encoded.drop('Attrition', axis=1)
        y = df_encoded['Attrition']
        
        # 5. Division train/test
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # 6. Normalisation
        X_train_scaled, X_test_scaled = self.normalize_data(X_train, X_test)
        
        # 7. Sauvegarde des artefacts
        self.save_artifacts(output_dir)
        
        # 8. Sauvegarde des données traitées
        os.makedirs('data/processed', exist_ok=True)
        
        train_data = pd.DataFrame(X_train_scaled, columns=self.feature_columns)
        train_data['Attrition'] = y_train.values
        train_data.to_csv('data/processed/train_data.csv', index=False)
        
        test_data = pd.DataFrame(X_test_scaled, columns=self.feature_columns)
        test_data['Attrition'] = y_test.values
        test_data.to_csv('data/processed/test_data.csv', index=False)
        
        print("="*60)
        print("✅ PIPELINE DE PRÉTRAITEMENT TERMINÉ")
        print("="*60)
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feature_columns': self.feature_columns,
            'n_features': len(self.feature_columns)
        }


if __name__ == "__main__":
    # Test du préprocesseur
    preprocessor = DataPreprocessor()
    result = preprocessor.run_pipeline("WA_Fn-UseC_-HR-Employee-Attrition.csv")
    
    print(f"\n📊 Résumé:")
    print(f"   - Features: {result['n_features']}")
    print(f"   - Train shape: {result['X_train'].shape}")
    print(f"   - Test shape: {result['X_test'].shape}")