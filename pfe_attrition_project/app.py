"""
Application Streamlit pour la prédiction d'attrition des employés.
Dashboard interactif pour l'analyse et la prédiction du risque de départ.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
from datetime import datetime
import os

warnings.filterwarnings('ignore')

# Base du projet calculée depuis ce fichier pour rester portable
# (local Windows/Linux, Streamlit Cloud et conteneur Docker).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE_PATH = os.path.join(
    BASE_DIR, "data", "raw", "WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

# ============================================================================
# CONFIGURATION DE LA PAGE
# ============================================================================

st.set_page_config(
    page_title="Prédiction d'Attrition des Employés",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# STYLE CSS PERSONNALISÉ
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #00d4ff;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        border: 1px solid #444;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-card h3 {
        color: #aaa;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    
    .metric-card h2 {
        color: #00d4ff;
        font-size: 2rem;
        margin: 0;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 1rem 0;
        animation: pulse 1.5s infinite;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #ffaa00 0%, #ff6600 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #00cc66 0%, #008844 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    .footer {
        text-align: center;
        padding: 1rem;
        margin-top: 2rem;
        color: #666;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CHARGEMENT DES DONNÉES
# ============================================================================

@st.cache_data
def load_data():
    """
    Charge et prépare les données.
    """
    # Chemin construit de manière robuste depuis la racine du projet.
    file_path = DATA_FILE_PATH
    
    try:
        df = pd.read_csv(file_path)
        df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
        
        # Suppression des colonnes inutiles
        columns_to_drop = ['EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours']
        df = df.drop([col for col in columns_to_drop if col in df.columns], axis=1)
        
        return df
    except FileNotFoundError:
        st.error("❌ Fichier de données non trouvé. Veuillez vérifier le chemin du fichier.")
        return None
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement: {e}")
        return None

# ============================================================================
# ENTRAÎNEMENT DU MODÈLE
# ============================================================================

@st.cache_resource
def train_model():
    """
    Entraîne le modèle Random Forest.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
    
    df = load_data()
    if df is None:
        return None, None, None, None, None, None, None
    
    # Encodage
    df_encoded = pd.get_dummies(df, drop_first=True)
    X = df_encoded.drop('Attrition', axis=1)
    y = df_encoded['Attrition']
    
    # Division
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Modèle
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=8,
        min_samples_leaf=4,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # Prédictions
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
    cm = confusion_matrix(y_test, y_pred)
    
    # Importance des features
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return model, scaler, X.columns, accuracy, roc_auc, cm, feature_importance

# ============================================================================
# FONCTION DE PRÉDICTION
# ============================================================================

def predict_attrition(employee_data, model, scaler, feature_columns):
    """
    Effectue la prédiction pour un employé.
    """
    input_df = pd.DataFrame([employee_data])
    input_encoded = pd.get_dummies(input_df)
    
    # Alignement des colonnes
    for col in feature_columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    
    input_encoded = input_encoded[feature_columns]
    input_scaled = scaler.transform(input_encoded)
    
    proba = model.predict_proba(input_scaled)[0, 1]
    return proba

# ============================================================================
# ANALYSE EXPLORATOIRE DES DONNÉES
# ============================================================================

def create_eda_section(df):
    """
    Crée la section d'analyse exploratoire.
    """
    st.markdown("<h1 class='main-header'>📈 Analyse Exploratoire</h1>", unsafe_allow_html=True)
    
    # Statistiques générales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("👥 Total Employés", f"{df.shape[0]:,}")
    
    with col2:
        attrition_rate = (df['Attrition'].sum() / df.shape[0]) * 100
        st.metric("⚠️ Taux d'Attrition", f"{attrition_rate:.1f}%")
    
    with col3:
        avg_age = df['Age'].mean()
        st.metric("📅 Âge Moyen", f"{avg_age:.0f} ans")
    
    with col4:
        avg_income = df['MonthlyIncome'].mean()
        st.metric("💰 Salaire Moyen", f"${avg_income:,.0f}")
    
    st.markdown("---")
    
    # Graphiques EDA
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Distribution", "📈 Corrélations", "👥 Démographie", "💼 Carrière"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ['#51cf66', '#ff6b6b']
            labels = ['Restés (0)', 'Quittés (1)']
            sizes = [df['Attrition'].value_counts()[0], df['Attrition'].value_counts()[1]]
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90, explode=(0, 0.05))
            ax.set_title("Distribution de l'Attrition", fontsize=14, fontweight='bold')
            st.pyplot(fig)
            plt.close()
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(df['Age'], bins=20, color='#00d4ff', edgecolor='black', alpha=0.7)
            ax.axvline(df['Age'].mean(), color='red', linestyle='--', label=f'Moyenne: {df["Age"].mean():.0f}')
            ax.set_xlabel('Âge')
            ax.set_ylabel('Fréquence')
            ax.set_title("Distribution des Âges", fontsize=14, fontweight='bold')
            ax.legend()
            st.pyplot(fig)
            plt.close()
    
    with tab2:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        ax.set_title("Matrice de Corrélation", fontsize=14, fontweight='bold')
        st.pyplot(fig)
        plt.close()
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            gender_attrition = df.groupby('Gender')['Attrition'].mean() * 100
            fig = px.bar(x=gender_attrition.values, y=gender_attrition.index, 
                        orientation='h', color=gender_attrition.values,
                        color_continuous_scale='RdYlGn_r',
                        title="Taux d'Attrition par Genre")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            marital_attrition = df.groupby('MaritalStatus')['Attrition'].mean() * 100
            fig = px.pie(values=marital_attrition.values, names=marital_attrition.index,
                        title="Taux d'Attrition par Situation Familiale")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            dept_attrition = df.groupby('Department')['Attrition'].mean() * 100
            fig = px.bar(x=dept_attrition.index, y=dept_attrition.values,
                        color=dept_attrition.values,
                        color_continuous_scale='RdYlGn_r',
                        title="Taux d'Attrition par Département")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            job_attrition = df.groupby('JobRole')['Attrition'].mean() * 100
            job_attrition = job_attrition.sort_values(ascending=True)
            fig = px.bar(x=job_attrition.values[-10:], y=job_attrition.index[-10:],
                        orientation='h', color=job_attrition.values[-10:],
                        color_continuous_scale='RdYlGn_r',
                        title="Top 10 Postes avec Taux d'Attrition")
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# SECTION PRÉDICTION - VERSION CORRIGÉE
# ============================================================================

def create_prediction_section(model, scaler, feature_columns):
    """
    Crée la section de prédiction interactive.
    """
    st.markdown("<h1 class='main-header'>🎯 Prédiction du Risque d'Attrition</h1>", unsafe_allow_html=True)
    
    st.info("💡 Renseignez les informations de l'employé pour évaluer son risque de départ.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 👤 Informations personnelles")
        age = st.slider("Âge", 18, 60, 30)
        gender = st.selectbox("Genre", ["Male", "Female"])
        marital_status = st.selectbox("Situation familiale", ["Single", "Married", "Divorced"])
        education = st.selectbox("Niveau d'éducation", [1, 2, 3, 4, 5])
        education_field = st.selectbox("Domaine d'éducation", 
                                       ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"])
    
    with col2:
        st.markdown("### 💼 Informations professionnelles")
        department = st.selectbox("Département", ["Sales", "Research & Development", "Human Resources"])
        job_role = st.selectbox("Poste", [
            "Sales Executive", "Research Scientist", "Laboratory Technician", 
            "Manufacturing Director", "Healthcare Representative", "Manager", 
            "Sales Representative", "Research Director", "Human Resources"
        ])
        job_level = st.slider("Niveau hiérarchique", 1, 5, 2)
        years_at_company = st.slider("Années dans l'entreprise", 0, 40, 5)
        years_in_role = st.slider("Années dans le poste actuel", 0, 20, 3)
        years_since_promotion = st.slider("Années depuis dernière promotion", 0, 15, 2)
        num_companies = st.slider("Nombre d'entreprises précédentes", 0, 20, 2)
    
    with col3:
        st.markdown("### 💰 Rémunération & Satisfaction")
        monthly_income = st.number_input("Revenu mensuel ($)", 1000, 20000, 5000, step=500)
        
        # CORRECTION: Utilisation de selectbox au lieu de slider avec format_func
        job_satisfaction_options = {1: "Très insatisfait", 2: "Insatisfait", 3: "Satisfait", 4: "Très satisfait"}
        job_satisfaction = st.selectbox(
            "Satisfaction au travail",
            options=list(job_satisfaction_options.keys()),
            format_func=lambda x: job_satisfaction_options[x],
            index=2
        )
        
        work_life_options = {1: "Mauvais", 2: "Moyen", 3: "Bon", 4: "Excellent"}
        work_life_balance = st.selectbox(
            "Équilibre vie pro/perso",
            options=list(work_life_options.keys()),
            format_func=lambda x: work_life_options[x],
            index=2
        )
        
        overtime = st.selectbox("Heures supplémentaires", ["No", "Yes"])
        business_travel = st.selectbox("Fréquence des voyages", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
        training_times = st.slider("Formations par an", 0, 6, 2)
        distance = st.slider("Distance domicile-travail (km)", 0, 50, 5)
    
    # Construction du dictionnaire des données
    employee_data = {
        'Age': age,
        'DistanceFromHome': distance,
        'Education': education,
        'EnvironmentSatisfaction': 3,
        'HourlyRate': 65,
        'JobInvolvement': 3,
        'JobLevel': job_level,
        'JobSatisfaction': job_satisfaction,
        'MonthlyIncome': monthly_income,
        'MonthlyRate': 15000,
        'NumCompaniesWorked': num_companies,
        'PercentSalaryHike': 15,
        'PerformanceRating': 3,
        'RelationshipSatisfaction': 3,
        'StockOptionLevel': 1,
        'TotalWorkingYears': 10,
        'TrainingTimesLastYear': training_times,
        'WorkLifeBalance': work_life_balance,
        'YearsAtCompany': years_at_company,
        'YearsInCurrentRole': years_in_role,
        'YearsSinceLastPromotion': years_since_promotion,
        'YearsWithCurrManager': 3,
        'BusinessTravel': business_travel,
        'Department': department,
        'EducationField': education_field,
        'Gender': gender,
        'JobRole': job_role,
        'MaritalStatus': marital_status,
        'OverTime': overtime
    }
    
    st.markdown("---")
    
    # Bouton de prédiction
    if st.button("🔮 PRÉDIRE LE RISQUE", type="primary", use_container_width=True):
        with st.spinner("Analyse en cours..."):
            proba = predict_attrition(employee_data, model, scaler, feature_columns)
        
        st.markdown("### 📊 Résultat de l'analyse")
        st.progress(proba)
        
        if proba >= 0.6:
            st.markdown(f"""
            <div class='risk-high'>
                <h2>⚠️ RISQUE TRÈS ÉLEVÉ</h2>
                <p style='font-size:3rem; margin:0'>{proba*100:.1f}%</p>
                <p>Probabilité de départ élevée. Une action immédiate est recommandée.</p>
            </div>
            """, unsafe_allow_html=True)
            st.error("""
            **Actions prioritaires :**
            - Organiser un entretien de rétention immédiat
            - Évaluer la charge de travail et les conditions de travail
            - Envisager une augmentation ou une promotion
            - Proposer un plan de développement personnalisé
            """)
        elif proba >= 0.4:
            st.markdown(f"""
            <div class='risk-medium'>
                <h2>⚠️ RISQUE MODÉRÉ</h2>
                <p style='font-size:3rem; margin:0'>{proba*100:.1f}%</p>
                <p>Probabilité de départ modérée. Une attention particulière est nécessaire.</p>
            </div>
            """, unsafe_allow_html=True)
            st.warning("""
            **Actions recommandées :**
            - Programmer un entretien de suivi
            - Vérifier la satisfaction au travail
            - Considérer des opportunités de formation
            - Améliorer la reconnaissance
            """)
        else:
            st.markdown(f"""
            <div class='risk-low'>
                <h2>✅ RISQUE FAIBLE</h2>
                <p style='font-size:3rem; margin:0'>{proba*100:.1f}%</p>
                <p>Probabilité de départ faible. L'employé semble stable.</p>
            </div>
            """, unsafe_allow_html=True)
            st.success("""
            **Points à maintenir :**
            - Continuer le suivi régulier
            - Maintenir un bon équilibre vie pro/perso
            - Reconnaître et valoriser le travail
            """)

# ============================================================================
# SECTION PERFORMANCE DU MODÈLE
# ============================================================================

def create_performance_section(accuracy, roc_auc, cm, feature_importance):
    """
    Crée la section des performances du modèle.
    """
    st.markdown("<h1 class='main-header'>📊 Performance du Modèle</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>🎯 Accuracy</h3>
            <h2>{accuracy:.1%}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <h3>📈 ROC-AUC</h3>
            <h2>{roc_auc:.3f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        f1 = 2 * (0.85 * 0.13) / (0.85 + 0.13) if (0.85 + 0.13) > 0 else 0
        st.markdown(f"""
        <div class='metric-card'>
            <h3>🔬 F1-Score</h3>
            <h2>{f1:.3f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Matrice de Confusion")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Resté (Prédit)', 'Quitté (Prédit)'],
                   yticklabels=['Resté (Réel)', 'Quitté (Réel)'])
        ax.set_title("Matrice de Confusion du Modèle", fontsize=14, fontweight='bold')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("🏆 Top 15 Features Importantes")
        fig = px.bar(feature_importance.head(15), x='Importance', y='Feature', 
                    orientation='h', color='Importance',
                    color_continuous_scale='Viridis',
                    title="Importance des Variables")
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# SECTION À PROPOS
# ============================================================================

def create_about_section():
    """
    Crée la section À propos.
    """
    st.markdown("<h1 class='main-header'>ℹ️ À Propos</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Objectif du Projet
        
        Cette application a été développée dans le cadre d'un projet de fin d'études
        visant à prédire l'attrition des employés à l'aide de techniques de Machine Learning.
        
        ### 🤖 Modèle Utilisé
        
        - **Algorithme**: Random Forest Classifier
        - **Nombres d'arbres**: 300
        - **Profondeur maximale**: 15
        - **Gestion du déséquilibre**: class_weight='balanced'
        
        ### 📊 Données
        
        Le jeu de données utilisé provient de IBM HR Analytics et contient:
        - **1 470 employés**
        - **31 variables explicatives**
        - **Taux d'attrition**: 16.1%
        
        ### 🔍 Facteurs Clés Identifiés
        
        1. **Heures supplémentaires** - multiplie le risque par 3
        2. **Âge** - les jeunes employés sont plus à risque
        3. **Satisfaction au travail** - facteur protecteur important
        4. **Ancienneté** - risque élevé les premières années
        5. **Poste commercial** - turnover historiquement élevé
        
        ### 🛠️ Technologies Utilisées
        
        - Python 3.9+
        - Streamlit (Dashboard)
        - Scikit-learn (Machine Learning)
        - Pandas/NumPy (Data processing)
        - Plotly/Matplotlib/Seaborn (Visualisation)
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Statistiques Clés
        
        | Indicateur | Valeur |
        |------------|--------|
        | Total Employés | 1 470 |
        | Taux d'Attrition | 16.1% |
        | Âge Moyen | 37 ans |
        | Salaire Moyen | $6,502 |
        | Ancienneté Moyenne | 7 ans |
        
        ### 🎯 Facteurs de Risque
        
        | Facteur | Impact |
        |---------|--------|
        | OverTime | ×3 |
        | Âge < 25 | ×2 |
        | Ancienneté < 2 | ×1.8 |
        | Sales Dept | ×1.5 |
        | Low Satisfaction | ×1.4 |
        """)
    
    st.markdown("---")
    st.markdown("""
    <div class='footer'>
        <p>© 2026 - Projet de Prédiction d'Attrition | PFE Bachelor | Tous droits réservés</p>
        <p>Version 1.0.0 | Dernière mise à jour: Mai 2026</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    """
    Fonction principale de l'application.
    """
    with st.spinner("Chargement des données et du modèle..."):
        df = load_data()
        if df is None:
            st.stop()
        
        model, scaler, feature_columns, accuracy, roc_auc, cm, feature_importance = train_model()
        if model is None:
            st.stop()
    
    # Sidebar
    st.sidebar.title("📊 Navigation")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Choisissez une section",
        ["🏠 Accueil", "📈 Analyse Exploratoire", "🎯 Prédiction", "📊 Performance", "ℹ️ À propos"]
    )
    
    st.sidebar.markdown("---")
    
    # Informations supplémentaires
    st.sidebar.markdown("### 📊 Statistiques Rapides")
    st.sidebar.metric("Total Employés", f"{df.shape[0]:,}")
    attrition_rate = (df['Attribution'].sum() / df.shape[0]) * 100 if 'Attribution' in df.columns else (df['Attrition'].sum() / df.shape[0]) * 100
    st.sidebar.metric("Taux d'Attrition", f"{attrition_rate:.1f}%")
    st.sidebar.metric("Accuracy Modèle", f"{accuracy:.1%}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("🔮 **v1.0.0**")
    
    # Affichage de la page
    if page == "🏠 Accueil":
        st.markdown("<h1 class='main-header'>📊 Prédiction de l'Attrition des Employés</h1>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>👥 Total Employés</h3>
                <h2>{df.shape[0]}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>⚠️ Taux d'Attrition</h3>
                <h2>{attrition_rate:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>🎯 Précision Modèle</h3>
                <h2>{accuracy:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ['#51cf66', '#ff6b6b']
            labels = ['Employés Restés', 'Employés Quittés']
            sizes = [df['Attrition'].value_counts()[0], df['Attrition'].value_counts()[1]]
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
            ax.set_title("Distribution de l'Attrition", fontsize=14, fontweight='bold')
            st.pyplot(fig)
            plt.close()
        
        with col2:
            dept_attrition = df.groupby('Department')['Attrition'].mean() * 100
            fig = px.bar(x=dept_attrition.values, y=dept_attrition.index, 
                        orientation='h', color=dept_attrition.values,
                        color_continuous_scale='RdYlGn_r',
                        title="Taux d'Attrition par Département")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("📊 Aperçu des Données"):
            st.dataframe(df.head(10), use_container_width=True)
    
    elif page == "📈 Analyse Exploratoire":
        create_eda_section(df)
    
    elif page == "🎯 Prédiction":
        create_prediction_section(model, scaler, feature_columns)
    
    elif page == "📊 Performance":
        create_performance_section(accuracy, roc_auc, cm, feature_importance)
    
    elif page == "ℹ️ À propos":
        create_about_section()

# ============================================================================
# EXÉCUTION
# ============================================================================

if __name__ == "__main__":
    main()