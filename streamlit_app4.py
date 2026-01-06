import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# Configuration de la page
st.set_page_config(
    page_title="Détection de Fraude - Système de Prédiction",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        padding: 40px 20px;
        margin-bottom: 20px;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7e22ce 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
        line-height: 1.2;
    }

    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #64748b;
        margin-top: -20px;
        margin-bottom: 40px;
        font-weight: 400;
    }

    .info-banner {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-left: 5px solid #0284c7;
        padding: 20px;
        border-radius: 12px;
        margin: 20px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    .section-card {
        background: white;
        padding: 30px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        margin: 20px 0;
        border: 1px solid #e2e8f0;
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .section-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
    }

    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
        border: 1px solid #e2e8f0;
        text-align: center;
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
    }

    .fraud-alert {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        color: white;
        padding: 30px;
        border-radius: 16px;
        font-size: 1.8rem;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 8px 32px rgba(220, 38, 38, 0.3);
        animation: pulse 2s infinite, slideIn 0.5s ease-out;
        border: 2px solid rgba(255, 255, 255, 0.2);
    }

    .legit-alert {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        color: white;
        padding: 30px;
        border-radius: 16px;
        font-size: 1.8rem;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 8px 32px rgba(5, 150, 105, 0.3);
        animation: slideIn 0.5s ease-out;
        border: 2px solid rgba(255, 255, 255, 0.2);
    }

    .warning-alert {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 30px;
        border-radius: 16px;
        font-size: 1.8rem;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 8px 32px rgba(245, 158, 11, 0.3);
        animation: slideIn 0.5s ease-out;
        border: 2px solid rgba(255, 255, 255, 0.2);
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.9; transform: scale(1.02); }
    }

    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .stButton>button {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 16px rgba(2, 132, 199, 0.3);
        transition: all 0.3s ease;
        font-size: 1.1rem;
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #0369a1 0%, #075985 100%);
        box-shadow: 0 6px 24px rgba(2, 132, 199, 0.4);
        transform: translateY(-2px);
    }

    .stats-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 5px;
    }

    .badge-success {
        background: #d1fae5;
        color: #065f46;
    }

    .badge-warning {
        background: #fef3c7;
        color: #92400e;
    }

    .badge-danger {
        background: #fee2e2;
        color: #991b1b;
    }

    .feature-box {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid #0284c7;
        margin: 15px 0;
    }

    h1, h2, h3 {
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
    }

    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #cbd5e1, transparent);
    }
    </style>
""", unsafe_allow_html=True)

# Chargement du modèle et du scaler
@st.cache_resource
def load_model_and_scaler():
    """Charge le modèle et le scaler depuis les fichiers pkl"""
    try:
        model = joblib.load('best_fraud_detection_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler, None
    except Exception as e:
        return None, None, str(e)

# Fonction de prédiction
# def predict_fraud(model, scaler, features_dict):
#     """
#     Effectue une prédiction de fraude
    
#     Args:
#         model: Le modèle chargé
#         scaler: Le scaler chargé
#         features_dict: Dictionnaire des features
    
#     Returns:
#         dict: Résultat de la prédiction
#     """
#     # Créer un DataFrame avec les features
#     df = pd.DataFrame([features_dict])
    
#     # Normaliser les données
#     features_scaled = scaler.transform(df)
    
#     # Prédiction
#     prediction = model.predict(features_scaled)[0]
#     probability = model.predict_proba(features_scaled)[0]
    
#     return {
#         'is_fraud': bool(prediction),
#         'fraud_probability': float(probability[1]),
#         'legit_probability': float(probability[0]),
#         'risk_level': 'ÉLEVÉ' if probability[1] >= 0.8 else ('MOYEN' if probability[1] >= 0.5 else 'FAIBLE'),
#         'confidence': float(max(probability))
#     }
def predict_fraud(model, scaler, features_dict, custom_threshold=0.3):
    """
    Effectue une prédiction de fraude avec seuil personnalisable
    
    Args:
        model: Le modèle chargé
        scaler: Le scaler chargé
        features_dict: Dictionnaire des features
        custom_threshold: seuil de fraude pour alerter (ex: 0.3 = 30%)
    
    Returns:
        dict: Résultat de la prédiction
    """
    # Créer un DataFrame avec les features
    df = pd.DataFrame([features_dict])
    
    # Normaliser les données
    features_scaled = scaler.transform(df)
    
    # Prédiction
    probability = model.predict_proba(features_scaled)[0]
    fraud_prob = float(probability[1])
    legit_prob = float(probability[0])
    
    # Déterminer le statut selon le seuil personnalisé
    is_fraud = fraud_prob >= custom_threshold
    
    # Niveau de risque ajusté
    if fraud_prob >= 0.8:
        risk_level = 'ÉLEVÉ'
    elif fraud_prob >= 0.5:
        risk_level = 'MOYEN'
    elif fraud_prob >= custom_threshold:
        risk_level = 'À VÉRIFIER'
    else:
        risk_level = 'FAIBLE'
    
    return {
        'is_fraud': is_fraud,
        'fraud_probability': fraud_prob,
        'legit_probability': legit_prob,
        'risk_level': risk_level,
        'confidence': float(max(probability))
    }

# Fonction pour créer les features à partir des inputs
def create_features(step, type_transaction, amount, oldbalance_org, newbalance_orig, 
                   oldbalance_dest, newbalance_dest):
    """Crée le dictionnaire de features à partir des inputs """
    
    # Initialiser toutes les features
    features = {
        'step': step,
        'amount': amount,
        'oldbalanceOrg': oldbalance_org,
        'newbalanceOrig': newbalance_orig,
        'oldbalanceDest': oldbalance_dest,
        'newbalanceDest': newbalance_dest
    }
    
    # One-hot encoding pour le type de transaction
    transaction_types = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
    for t_type in transaction_types:
        features[f'type_{t_type}'] = 1 if type_transaction == t_type else 0
    
    # Features dérivées
    features['balanceChange_orig'] = oldbalance_org - newbalance_orig
    features['balanceChange_dest'] = newbalance_dest - oldbalance_dest
    features['amountToBalanceRatio_orig'] = amount / (oldbalance_org + 1)
    features['isOriginEmpty'] = 1 if newbalance_orig == 0 else 0
    features['isDestEmpty'] = 1 if oldbalance_dest == 0 else 0
    features['errorBalanceOrig'] = (oldbalance_org - newbalance_orig) - amount
    features['errorBalanceDest'] = (newbalance_dest - oldbalance_dest) - amount
    
    return features

# Fonction pour afficher la jauge de probabilité
def create_gauge_chart(probability):
    """Crée un graphique de jauge pour la probabilité de fraude"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Probabilité de Fraude", 'font': {'size': 20, 'family': 'Inter', 'color': '#1e293b'}},
        delta={'reference': 50, 'increasing': {'color': "#dc2626"}},
        number={'suffix': "%", 'font': {'size': 48, 'family': 'Inter'}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#64748b", 'tickfont': {'size': 14}},
            'bar': {'color': "#0284c7", 'thickness': 0.8},
            'bgcolor': "white",
            'borderwidth': 3,
            'bordercolor': "#e2e8f0",
            'steps': [
                {'range': [0, 30], 'color': '#d1fae5'},
                {'range': [30, 70], 'color': '#fef3c7'},
                {'range': [70, 100], 'color': '#fee2e2'}
            ],
            'threshold': {
                'line': {'color': "#dc2626", 'width': 5},
                'thickness': 0.85,
                'value': 80
            }
        }
    ))

    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=80, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter'}
    )

    return fig

# Fonction pour créer un graphique de probabilités comparatives
def create_probability_bar(fraud_prob, legit_prob):
    """Crée un graphique en barres pour comparer les probabilités"""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=['Transaction Légitime', 'Transaction Frauduleuse'],
        y=[legit_prob * 100, fraud_prob * 100],
        marker=dict(
            color=['#059669', '#dc2626'],
            line=dict(color=['#047857', '#b91c1c'], width=2)
        ),
        text=[f'{legit_prob*100:.1f}%', f'{fraud_prob*100:.1f}%'],
        textposition='outside',
        textfont=dict(size=16, family='Inter', color='#1e293b'),
        hovertemplate='<b>%{x}</b><br>Probabilité: %{y:.2f}%<extra></extra>'
    ))

    fig.update_layout(
        title={
            'text': "Analyse Comparative des Probabilités",
            'font': {'size': 20, 'family': 'Inter', 'color': '#1e293b'},
            'x': 0.5,
            'xanchor': 'center'
        },
        yaxis=dict(
            title="Probabilité (%)",
            titlefont=dict(size=14, family='Inter'),
            gridcolor='#e2e8f0',
            range=[0, 105]
        ),
        xaxis=dict(
            titlefont=dict(size=14, family='Inter')
        ),
        height=350,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='white',
        font={'family': 'Inter'},
        margin=dict(l=40, r=40, t=80, b=40)
    )

    return fig

# Fonction pour afficher l'historique des prédictions
def display_prediction_history(history):
    """Affiche l'historique des prédictions dans un graphique"""
    if len(history) > 0:
        df_history = pd.DataFrame(history)
        
        fig = px.line(
            df_history, 
            x='timestamp', 
            y='fraud_probability',
            title='Évolution des Probabilités de Fraude',
            labels={'fraud_probability': 'Probabilité de Fraude', 'timestamp': 'Temps'}
        )
        
        fig.add_hline(y=0.5, line_dash="dash", line_color="orange", 
                     annotation_text="Seuil de décision")
        fig.add_hline(y=0.8, line_dash="dash", line_color="red", 
                     annotation_text="Seuil d'alerte")
        
        fig.update_layout(height=300)
        
        return fig
    return None

# Interface principale
def main():
    # En-tête
    st.markdown('<h1 class="main-header">Système Intelligent de Détection de Fraude</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Analyse en temps réel des transactions financières avec Machine Learning</p>', unsafe_allow_html=True)

    # Chargement du modèle
    model, scaler, error = load_model_and_scaler()

    if error:
        st.error(f"Erreur lors du chargement du modèle: {error}")
        st.info("Assurez-vous que les fichiers 'best_fraud_detection_model.pkl' et 'scaler.pkl' sont dans le même répertoire.")
        return

    st.markdown('<div class="info-banner">Modèle de Machine Learning chargé et prêt à analyser vos transactions</div>', unsafe_allow_html=True)

    # Initialiser l'historique dans session_state
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []

    # Sidebar - Configuration
    with st.sidebar:
        st.markdown("### Configuration de l'Analyse")

        st.markdown('<div class="feature-box">', unsafe_allow_html=True)
        input_mode = st.radio(
            "Mode de saisie",
            ["📝 Saisie Manuelle", "📁 Import CSV"],
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("### À propos du Système")
        st.markdown("""
        Ce système utilise des algorithmes de Machine Learning avancés pour détecter les transactions frauduleuses en temps réel.

        **Caractéristiques:**
        - Analyse instantanée
        - Précision élevée
        - Interface intuitive
        - Export des résultats
        """)

        st.markdown("---")

        st.markdown("### Statistiques de Session")
        total_predictions = len(st.session_state.prediction_history)
        if total_predictions > 0:
            fraud_count = sum(1 for p in st.session_state.prediction_history if p['is_fraud'])
            st.metric("Analyses Effectuées", total_predictions)
            st.metric("Fraudes Détectées", fraud_count)
            st.metric("Taux de Fraude", f"{fraud_count/total_predictions*100:.1f}%")
        else:
            st.info("Aucune analyse effectuée pour le moment")
    
    # st.sidebar.markdown("---")
    # st.sidebar.markdown("### 📊 Statistiques de Session")
    # st.sidebar.metric("Prédictions Effectuées", len(st.session_state.prediction_history))
    
    # if len(st.session_state.prediction_history) > 0:
    #     fraud_count = sum(1 for p in st.session_state.prediction_history if p['is_fraud'])
    #     st.sidebar.metric("Fraudes Détectées", fraud_count)
    #     st.sidebar.metric("Taux de Fraude", f"{fraud_count/len(st.session_state.prediction_history)*100:.1f}%")
    
    # # Bouton pour réinitialiser l'historique
    # if st.sidebar.button("🔄 Réinitialiser l'Historique"):
    #     st.session_state.prediction_history = []
    #     st.rerun()
    
    # Mode Saisie Manuelle
    if input_mode == "📝 Saisie Manuelle":
        st.markdown("## 📊 Analyse de Transaction Individuelle")
        st.markdown("Entrez les détails de la transaction pour obtenir une analyse instantanée")

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3, gap="large")

        with col1:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("#### 🔢 Informations Générales")
            step = st.number_input("Step (Heure)", min_value=0, max_value=744, value=1,
                                  help="Unité de temps (1 step = 1 heure)")
            type_transaction = st.selectbox(
                "Type de Transaction",
                ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
            )
            amount = st.number_input("Montant (€)", min_value=0.0, value=10000.0, step=100.0,
                                    help="Montant de la transaction", format="%.2f")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("#### 💳 Compte Origine")
            oldbalance_org = st.number_input("Solde Initial (€)", min_value=0.0, value=50000.0,
                                            step=1000.0, key="old_orig", format="%.2f")
            newbalance_orig = st.number_input("Nouveau Solde (€)", min_value=0.0, value=40000.0,
                                             step=1000.0, key="new_orig", format="%.2f")
            balance_diff_orig = oldbalance_org - newbalance_orig
            st.info(f"Variation: {balance_diff_orig:,.2f} €")
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("#### 🏦 Compte Destination")
            oldbalance_dest = st.number_input("Solde Initial (€)", min_value=0.0, value=0.0,
                                             step=1000.0, key="old_dest", format="%.2f")
            newbalance_dest = st.number_input("Nouveau Solde (€)", min_value=0.0, value=10000.0,
                                             step=1000.0, key="new_dest", format="%.2f")
            balance_diff_dest = newbalance_dest - oldbalance_dest
            st.info(f"Variation: +{balance_diff_dest:,.2f} €")
            st.markdown('</div>', unsafe_allow_html=True)

        # Bouton de prédiction
        st.markdown("<br>", unsafe_allow_html=True)
        predict_button = st.button("🔍 Lancer l'Analyse de Fraude", type="primary", use_container_width=True)
        
        if predict_button:
            # Créer les features
            features = create_features(
                step, type_transaction, amount, oldbalance_org, 
                newbalance_orig, oldbalance_dest, newbalance_dest
            )
            
            # Faire la prédiction avec animation
            with st.spinner("Analyse en cours..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                result = predict_fraud(model, scaler, features)
            
            # Ajouter à l'historique
            result['timestamp'] = datetime.now()
            result['amount'] = amount
            result['type'] = type_transaction
            st.session_state.prediction_history.append(result)
            
            # Afficher les résultats
            st.markdown("---")
            st.markdown("## 🎯 Résultats de l'Analyse")

            # Alerte principale avec style amélioré
            if result['is_fraud']:
                if result['risk_level'] == 'ÉLEVÉ':
                    st.markdown(
                        f'<div class="fraud-alert">⚠️ ALERTE FRAUDE DÉTECTÉE ⚠️<br><span style="font-size: 1.2rem;">Niveau de Risque: {result["risk_level"]}</span></div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="warning-alert">⚠️ TRANSACTION SUSPECTE<br><span style="font-size: 1.2rem;">Niveau de Risque: {result["risk_level"]}</span></div>',
                        unsafe_allow_html=True
                    )
            else:
                st.markdown(
                    '<div class="legit-alert">✅ TRANSACTION LÉGITIME<br><span style="font-size: 1.2rem;">Aucun risque détecté</span></div>',
                    unsafe_allow_html=True
                )

            st.markdown("<br>", unsafe_allow_html=True)
            
            # Métriques
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Probabilité de Fraude",
                    f"{result['fraud_probability']*100:.2f}%",
                    delta=f"{(result['fraud_probability']-0.5)*100:.1f}%" if result['fraud_probability'] > 0.5 else None,
                    delta_color="inverse"
                )
            
            with col2:
                st.metric(
                    "Probabilité Légitime",
                    f"{result['legit_probability']*100:.2f}%"
                )
            
            with col3:
                st.metric(
                    "Niveau de Risque",
                    result['risk_level'],
                    delta="ALERTE" if result['risk_level'] == "ÉLEVÉ" else None,
                    delta_color="inverse"
                )
            
            with col4:
                st.metric(
                    "Confiance",
                    f"{result['confidence']*100:.1f}%"
                )
            
            # Graphiques
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(create_gauge_chart(result['fraud_probability']), 
                              use_container_width=True)
            
            with col2:
                st.plotly_chart(create_probability_bar(result['fraud_probability'], 
                                                      result['legit_probability']),
                              use_container_width=True)
            
            # Détails de la transaction
            st.markdown("---")
            st.markdown("### 📋 Récapitulatif de la Transaction")

            details_col1, details_col2 = st.columns(2, gap="large")

            with details_col1:
                st.markdown('<div class="feature-box">', unsafe_allow_html=True)
                st.markdown("**Informations Générales**")
                st.markdown(f"""
                - **Type de Transaction:** `{type_transaction}`
                - **Montant:** {amount:,.2f} €
                - **Timestamp:** Step {step}
                - **Date d'analyse:** {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}
                """)
                st.markdown('</div>', unsafe_allow_html=True)

            with details_col2:
                st.markdown('<div class="feature-box">', unsafe_allow_html=True)
                st.markdown("**Mouvements de Comptes**")
                balance_change_orig = oldbalance_org - newbalance_orig
                balance_change_dest = newbalance_dest - oldbalance_dest
                ratio = amount/(oldbalance_org+1)*100
                st.markdown(f"""
                - **Variation Origine:** {balance_change_orig:,.2f} €
                - **Variation Destination:** +{balance_change_dest:,.2f} €
                - **Ratio Montant/Solde:** {ratio:.2f}%
                - **Différence:** {abs(balance_change_orig - amount):,.2f} €
                """)
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Mode Import CSV
    elif input_mode == "📁 Import CSV":
        st.markdown("## 📁 Analyse par Lots de Transactions")
        st.markdown("Importez un fichier CSV pour analyser plusieurs transactions simultanément")

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Sélectionnez votre fichier CSV",
            type=['csv'],
            help="Format requis: step, type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest"
        )

        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

            st.markdown('<div class="info-banner">Fichier chargé avec succès - {} transactions prêtes à être analysées</div>'.format(len(df)), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Afficher un aperçu
            st.markdown("### 👁️ Aperçu des Données")
            st.dataframe(df.head(10), use_container_width=True, height=350)

            st.markdown("<br>", unsafe_allow_html=True)

            # Bouton d'analyse
            if st.button("🔍 Lancer l'Analyse Complète", type="primary", use_container_width=True):
                with st.spinner("Analyse en cours..."):
                    progress_bar = st.progress(0)
                    results = []
                    
                    for idx, row in df.iterrows():
                        # Créer les features
                        features = create_features(
                            row['step'], row['type'], row['amount'],
                            row['oldbalanceOrg'], row['newbalanceOrig'],
                            row['oldbalanceDest'], row['newbalanceDest']
                        )
                        
                        # Prédiction
                        result = predict_fraud(model, scaler, features)
                        result['transaction_id'] = idx
                        results.append(result)
                        
                        # Mettre à jour la barre de progression
                        progress_bar.progress((idx + 1) / len(df))
                
                # Créer un DataFrame des résultats
                results_df = pd.DataFrame(results)
                df['is_fraud_predicted'] = results_df['is_fraud']
                df['fraud_probability'] = results_df['fraud_probability']
                df['risk_level'] = results_df['risk_level']
                
                # Statistiques
                st.markdown("---")
                st.markdown("## 📊 Résultats de l'Analyse par Lots")

                col1, col2, col3, col4 = st.columns(4, gap="large")

                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Total Transactions", f"{len(df):,}", help="Nombre total de transactions analysées")
                    st.markdown('</div>', unsafe_allow_html=True)

                with col2:
                    fraud_count = df['is_fraud_predicted'].sum()
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Fraudes Détectées", fraud_count, delta=f"{fraud_count} cas", delta_color="inverse")
                    st.markdown('</div>', unsafe_allow_html=True)

                with col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Taux de Fraude", f"{fraud_count/len(df)*100:.2f}%",
                             help="Pourcentage de transactions frauduleuses")
                    st.markdown('</div>', unsafe_allow_html=True)

                with col4:
                    avg_prob = df['fraud_probability'].mean()
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Probabilité Moyenne", f"{avg_prob*100:.2f}%",
                             help="Moyenne des probabilités de fraude")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Graphiques
                st.markdown("---")
                st.markdown("### 📈 Visualisations Statistiques")

                col1, col2 = st.columns(2, gap="large")

                with col1:
                    # Distribution des probabilités
                    fig_dist = px.histogram(
                        df, x='fraud_probability',
                        title='Distribution des Probabilités de Fraude',
                        nbins=50,
                        labels={'fraud_probability': 'Probabilité de Fraude'},
                        color_discrete_sequence=['#0284c7']
                    )
                    fig_dist.update_layout(
                        font={'family': 'Inter'},
                        title={'font': {'size': 18}},
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='white',
                        height=400
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)

                with col2:
                    # Fraude par type de transaction
                    fraud_by_type = df.groupby('type')['is_fraud_predicted'].sum().reset_index()
                    fig_type = px.bar(
                        fraud_by_type, x='type', y='is_fraud_predicted',
                        title='Fraudes par Type de Transaction',
                        labels={'is_fraud_predicted': 'Nombre de Fraudes', 'type': 'Type de Transaction'},
                        color='is_fraud_predicted',
                        color_continuous_scale=['#d1fae5', '#dc2626']
                    )
                    fig_type.update_layout(
                        font={'family': 'Inter'},
                        title={'font': {'size': 18}},
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='white',
                        showlegend=False,
                        height=400
                    )
                    st.plotly_chart(fig_type, use_container_width=True)
                
                # Tableau des transactions suspectes
                st.markdown("---")
                st.markdown("### 🚨 Transactions Hautement Suspectes")
                st.markdown("Transactions avec une probabilité de fraude supérieure à 80%")

                suspicious = df[df['fraud_probability'] > 0.8].sort_values('fraud_probability', ascending=False)

                if len(suspicious) > 0:
                    st.markdown(f'<div class="info-banner">⚠️ {len(suspicious)} transaction(s) hautement suspecte(s) détectée(s)</div>', unsafe_allow_html=True)
                    st.dataframe(
                        suspicious[['type', 'amount', 'fraud_probability', 'risk_level']].head(20),
                        use_container_width=True,
                        height=400
                    )
                else:
                    st.success("Aucune transaction hautement suspecte détectée")

                # Télécharger les résultats
                st.markdown("---")
                st.markdown("### 💾 Export des Résultats")

                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown("Téléchargez les résultats complets de l'analyse au format CSV pour une utilisation ultérieure")

                with col2:
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Télécharger les Résultats",
                        data=csv,
                        file_name=f'fraud_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                        mime='text/csv',
                        use_container_width=True
                    )
    
    # Mode Données Aléatoires
    else:  # Données Aléatoires
        st.header("Générer et Analyser des Transactions Aléatoires")
        
        col1, col2 = st.columns(2)
        
        with col1:
            num_transactions = st.slider("Nombre de transactions", 1, 100, 10)
        
        with col2:
            transaction_type = st.selectbox(
                "Type de transaction",
                ["Tous", "PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
            )
        
        if st.button("🎲 Générer et Analyser", type="primary"):
            with st.spinner("Génération et analyse en cours..."):
                progress_bar = st.progress(0)
                results = []
                
                for i in range(num_transactions):
                    # Générer des données aléatoires
                    if transaction_type == "Tous":
                        t_type = np.random.choice(["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"])
                    else:
                        t_type = transaction_type
                    
                    step = np.random.randint(1, 744)
                    amount = np.random.uniform(100, 100000)
                    oldbalance_org = np.random.uniform(0, 200000)
                    newbalance_orig = max(0, oldbalance_org - amount + np.random.uniform(-1000, 1000))
                    oldbalance_dest = np.random.uniform(0, 200000)
                    newbalance_dest = oldbalance_dest + amount + np.random.uniform(-1000, 1000)
                    
                    # Créer features et prédire
                    features = create_features(
                        step, t_type, amount, oldbalance_org,
                        newbalance_orig, oldbalance_dest, newbalance_dest
                    )
                    
                    result = predict_fraud(model, scaler, features)
                    result['type'] = t_type
                    result['amount'] = amount
                    result['timestamp'] = datetime.now()
                    results.append(result)
                    
                    progress_bar.progress((i + 1) / num_transactions)
                
                # Ajouter à l'historique
                st.session_state.prediction_history.extend(results)
            
            # Afficher les résultats
            st.markdown("---")
            st.success(f"✅ {num_transactions} transactions générées et analysées!")
            
            # Statistiques
            fraud_count = sum(1 for r in results if r['is_fraud'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Transactions Générées", num_transactions)
            with col2:
                st.metric("Fraudes Détectées", fraud_count)
            with col3:
                st.metric("Taux de Fraude", f"{fraud_count/num_transactions*100:.1f}%")
            
            # Tableau des résultats
            st.markdown("---")
            results_df = pd.DataFrame(results)
            st.dataframe(results_df[['type', 'amount', 'is_fraud', 'fraud_probability', 'risk_level']], 
                        use_container_width=True)
    
    # Section Historique (visible dans tous les modes)
    if len(st.session_state.prediction_history) > 0:
        st.markdown("---")
        st.header("📈 Historique des Prédictions")
        
        fig_history = display_prediction_history(st.session_state.prediction_history)
        if fig_history:
            st.plotly_chart(fig_history, use_container_width=True)
        
        # Statistiques de l'historique
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_prob = np.mean([p['fraud_probability'] for p in st.session_state.prediction_history])
            st.metric("Probabilité Moyenne", f"{avg_prob*100:.2f}%")
        
        with col2:
            fraud_count = sum(1 for p in st.session_state.prediction_history if p['is_fraud'])
            st.metric("Total Fraudes", fraud_count)
        
        with col3:
            high_risk = sum(1 for p in st.session_state.prediction_history if p['risk_level'] == 'ÉLEVÉ')
            st.metric("Alertes Élevées", high_risk)

if __name__ == "__main__":
    main()

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #64748b; padding: 20px;'>
        <p style='margin: 0; font-size: 0.9rem;'>
            <strong>Système de Détection de Fraude</strong> | Machine Learning Application<br>
            Développé avec Streamlit, Scikit-learn et Plotly<br>
            © 2026 - Tous droits réservés
        </p>
    </div>
    """, unsafe_allow_html=True)