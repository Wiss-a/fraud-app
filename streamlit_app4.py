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
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .fraud-alert {
        background-color: #ff4444;
        color: white;
        padding: 20px;
        border-radius: 10px;
        font-size: 1.5rem;
        text-align: center;
        animation: pulse 2s infinite;
    }
    .legit-alert {
        background-color: #00C851;
        color: white;
        padding: 20px;
        border-radius: 10px;
        font-size: 1.5rem;
        text-align: center;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
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
        title={'text': "Probabilité de Fraude (%)", 'font': {'size': 24}},
        delta={'reference': 50, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#00C851'},
                {'range': [30, 70], 'color': '#ffbb33'},
                {'range': [70, 100], 'color': '#ff4444'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

# Fonction pour créer un graphique de probabilités comparatives
def create_probability_bar(fraud_prob, legit_prob):
    """Crée un graphique en barres pour comparer les probabilités"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=['Légitime', 'Fraude'],
        y=[legit_prob * 100, fraud_prob * 100],
        marker_color=['#00C851', '#ff4444'],
        text=[f'{legit_prob*100:.2f}%', f'{fraud_prob*100:.2f}%'],
        textposition='auto',
    ))
    
    fig.update_layout(
        title="Comparaison des Probabilités",
        yaxis_title="Probabilité (%)",
        height=300,
        showlegend=False
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
    st.markdown('<h1 class="main-header">🔐 Système de Détection de Fraude</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Chargement du modèle
    model, scaler, error = load_model_and_scaler()
    
    if error:
        st.error(f"❌ Erreur lors du chargement du modèle: {error}")
        st.info("Assurez-vous que les fichiers 'best_fraud_detection_model.pkl' et 'scaler.pkl' sont dans le même répertoire.")
        return
    
    st.success("✅ Modèle chargé avec succès !")
    
    # Initialiser l'historique dans session_state
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    # Sidebar - Mode de saisie
    st.sidebar.title("⚙️ Configuration")
    input_mode = st.sidebar.radio(
        "Mode de saisie",
        ["📝 Saisie Manuelle", "📁 Import CSV"]
    )
    
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
    #     st.rerun()yhtgfrghj
    
    # Mode Saisie Manuelle
    if input_mode == "📝 Saisie Manuelle":
        st.header("Saisir les Détails de la Transaction")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Informations Générales")
            step = st.number_input("Step (Heure)", min_value=0, max_value=744, value=1, 
                                  help="Unité de temps (1 step = 1 heure)")
            type_transaction = st.selectbox(
                "Type de Transaction",
                ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
            )
            amount = st.number_input("Montant", min_value=0.0, value=10000.0, step=100.0,
                                    help="Montant de la transaction")
        
        with col2:
            st.subheader("Compte Origine")
            oldbalance_org = st.number_input("Solde Initial", min_value=0.0, value=50000.0, 
                                            step=1000.0, key="old_orig")
            newbalance_orig = st.number_input("Nouveau Solde", min_value=0.0, value=40000.0, 
                                             step=1000.0, key="new_orig")
        
        with col3:
            st.subheader("Compte Destination")
            oldbalance_dest = st.number_input("Solde Initial", min_value=0.0, value=0.0, 
                                             step=1000.0, key="old_dest")
            newbalance_dest = st.number_input("Nouveau Solde", min_value=0.0, value=10000.0, 
                                             step=1000.0, key="new_dest")
        
        # Bouton de prédiction
        st.markdown("---")
        predict_button = st.button("🔍 Analyser la Transaction", type="primary", use_container_width=True)
        
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
            st.header("📊 Résultat de l'Analyse")
            
            # Alerte principale
            if result['is_fraud']:
                st.markdown(
                    f'<div class="fraud-alert">⚠️ ALERTE FRAUDE DÉTECTÉE ⚠️<br>Niveau de Risque: {result["risk_level"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="legit-alert">✅ Transaction Légitime</div>',
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
            st.subheader("📋 Détails de la Transaction")
            
            details_col1, details_col2 = st.columns(2)
            
            with details_col1:
                st.write("**Informations de Base:**")
                st.write(f"- Type: {type_transaction}")
                st.write(f"- Montant: {amount:,.2f} €")
                st.write(f"- Step: {step}")
            
            with details_col2:
                st.write("**Analyse des Comptes:**")
                balance_change_orig = oldbalance_org - newbalance_orig
                balance_change_dest = newbalance_dest - oldbalance_dest
                st.write(f"- Variation Origine: {balance_change_orig:,.2f} €")
                st.write(f"- Variation Destination: {balance_change_dest:,.2f} €")
                st.write(f"- Ratio Montant/Solde: {amount/(oldbalance_org+1)*100:.2f}%")
    
    # Mode Import CSV
    elif input_mode == "📁 Import CSV":
        st.header("Importer un Fichier de Transactions")
        
        uploaded_file = st.file_uploader(
            "Choisissez un fichier CSV",
            type=['csv'],
            help="Le fichier doit contenir les colonnes: step, type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest"
        )
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Fichier chargé: {len(df)} transactions")
            
            # Afficher un aperçu
            st.subheader("Aperçu des Données")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Bouton d'analyse
            if st.button("🔍 Analyser Toutes les Transactions", type="primary"):
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
                st.header("📊 Résultats de l'Analyse Batch")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Transactions", len(df))
                
                with col2:
                    fraud_count = df['is_fraud_predicted'].sum()
                    st.metric("Fraudes Détectées", fraud_count)
                
                with col3:
                    st.metric("Taux de Fraude", f"{fraud_count/len(df)*100:.2f}%")
                
                with col4:
                    avg_prob = df['fraud_probability'].mean()
                    st.metric("Probabilité Moyenne", f"{avg_prob*100:.2f}%")
                
                # Graphiques
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    # Distribution des probabilités
                    fig_dist = px.histogram(
                        df, x='fraud_probability',
                        title='Distribution des Probabilités de Fraude',
                        nbins=50,
                        labels={'fraud_probability': 'Probabilité de Fraude'}
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
                
                with col2:
                    # Fraude par type de transaction
                    fraud_by_type = df.groupby('type')['is_fraud_predicted'].sum().reset_index()
                    fig_type = px.bar(
                        fraud_by_type, x='type', y='is_fraud_predicted',
                        title='Fraudes par Type de Transaction',
                        labels={'is_fraud_predicted': 'Nombre de Fraudes', 'type': 'Type'}
                    )
                    st.plotly_chart(fig_type, use_container_width=True)
                
                # Tableau des transactions suspectes
                st.markdown("---")
                st.subheader("🚨 Transactions Hautement Suspectes (Probabilité > 80%)")
                suspicious = df[df['fraud_probability'] > 0.8].sort_values('fraud_probability', ascending=False)
                
                if len(suspicious) > 0:
                    st.dataframe(
                        suspicious[['type', 'amount', 'fraud_probability', 'risk_level']].head(20),
                        use_container_width=True
                    )
                else:
                    st.info("Aucune transaction hautement suspecte détectée.")
                
                # Télécharger les résultats
                st.markdown("---")
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Télécharger les Résultats (CSV)",
                    data=csv,
                    file_name=f'fraud_analysis_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                    mime='text/csv',
                )
    
    # Mode Données Aléatoires
    # else:  # Données Aléatoires
    #     st.header("Générer et Analyser des Transactions Aléatoires")
        
    #     col1, col2 = st.columns(2)
        
    #     with col1:
    #         num_transactions = st.slider("Nombre de transactions", 1, 100, 10)
        
    #     with col2:
    #         transaction_type = st.selectbox(
    #             "Type de transaction",
    #             ["Tous", "PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
    #         )
        
    #     if st.button("🎲 Générer et Analyser", type="primary"):
    #         with st.spinner("Génération et analyse en cours..."):
    #             progress_bar = st.progress(0)
    #             results = []
                
    #             for i in range(num_transactions):
    #                 # Générer des données aléatoires
    #                 if transaction_type == "Tous":
    #                     t_type = np.random.choice(["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"])
    #                 else:
    #                     t_type = transaction_type
                    
    #                 step = np.random.randint(1, 744)
    #                 amount = np.random.uniform(100, 100000)
    #                 oldbalance_org = np.random.uniform(0, 200000)
    #                 newbalance_orig = max(0, oldbalance_org - amount + np.random.uniform(-1000, 1000))
    #                 oldbalance_dest = np.random.uniform(0, 200000)
    #                 newbalance_dest = oldbalance_dest + amount + np.random.uniform(-1000, 1000)
                    
    #                 # Créer features et prédire
    #                 features = create_features(
    #                     step, t_type, amount, oldbalance_org,
    #                     newbalance_orig, oldbalance_dest, newbalance_dest
    #                 )
                    
    #                 result = predict_fraud(model, scaler, features)
    #                 result['type'] = t_type
    #                 result['amount'] = amount
    #                 result['timestamp'] = datetime.now()
    #                 results.append(result)
                    
    #                 progress_bar.progress((i + 1) / num_transactions)
                
    #             # Ajouter à l'historique
    #             st.session_state.prediction_history.extend(results)
            
    #         # Afficher les résultats
    #         st.markdown("---")
    #         st.success(f"✅ {num_transactions} transactions générées et analysées!")
            
    #         # Statistiques
    #         fraud_count = sum(1 for r in results if r['is_fraud'])
            
    #         col1, col2, col3 = st.columns(3)
    #         with col1:
    #             st.metric("Transactions Générées", num_transactions)
    #         with col2:
    #             st.metric("Fraudes Détectées", fraud_count)
    #         with col3:
    #             st.metric("Taux de Fraude", f"{fraud_count/num_transactions*100:.1f}%")
            
    #         # Tableau des résultats
    #         st.markdown("---")
    #         results_df = pd.DataFrame(results)
    #         st.dataframe(results_df[['type', 'amount', 'is_fraud', 'fraud_probability', 'risk_level']], 
    #                     use_container_width=True)
    
    # Section Historique (visible dans tous les modes)
    # if len(st.session_state.prediction_history) > 0:
    #     st.markdown("---")
    #     st.header("📈 Historique des Prédictions")
        
    #     fig_history = display_prediction_history(st.session_state.prediction_history)
    #     if fig_history:
    #         st.plotly_chart(fig_history, use_container_width=True)
        
    #     # Statistiques de l'historique
    #     col1, col2, col3 = st.columns(3)
        
    #     with col1:
    #         avg_prob = np.mean([p['fraud_probability'] for p in st.session_state.prediction_history])
    #         st.metric("Probabilité Moyenne", f"{avg_prob*100:.2f}%")
        
    #     with col2:
    #         fraud_count = sum(1 for p in st.session_state.prediction_history if p['is_fraud'])
    #         st.metric("Total Fraudes", fraud_count)
        
    #     with col3:
    #         high_risk = sum(1 for p in st.session_state.prediction_history if p['risk_level'] == 'ÉLEVÉ')
    #         st.metric("Alertes Élevées", high_risk)

if __name__ == "__main__":
    main()