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
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="collapsed"
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
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 24px;
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
    
    /* Formulaire compact */
    [data-testid="stForm"] {
        padding: 0;
    }
    [data-testid="stForm"] h3 {
        margin-top: 10px;
        margin-bottom: 10px;
        font-size: 1.1rem;
    }
    [data-testid="stForm"] .stNumberInput,
    [data-testid="stForm"] .stSelectbox {
        margin-bottom: 5px;
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

def predict_fraud(model, scaler, features_dict, custom_threshold=0.3):
    """Effectue une prédiction de fraude avec seuil personnalisable"""
    df = pd.DataFrame([features_dict])
    features_scaled = scaler.transform(df)
    probability = model.predict_proba(features_scaled)[0]
    fraud_prob = float(probability[1])
    legit_prob = float(probability[0])
    is_fraud = fraud_prob >= custom_threshold
    
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

def create_features(step, type_transaction, amount, oldbalance_org, newbalance_orig, 
                   oldbalance_dest, newbalance_dest):
    """Crée le dictionnaire de features à partir des inputs"""
    features = {
        'step': step,
        'amount': amount,
        'oldbalanceOrg': oldbalance_org,
        'newbalanceOrig': newbalance_orig,
        'oldbalanceDest': oldbalance_dest,
        'newbalanceDest': newbalance_dest
    }
    
    transaction_types = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
    for t_type in transaction_types:
        features[f'type_{t_type}'] = 1 if type_transaction == t_type else 0
    
    features['balanceChange_orig'] = oldbalance_org - newbalance_orig
    features['balanceChange_dest'] = newbalance_dest - oldbalance_dest
    features['amountToBalanceRatio_orig'] = amount / (oldbalance_org + 1)
    features['isOriginEmpty'] = 1 if newbalance_orig == 0 else 0
    features['isDestEmpty'] = 1 if oldbalance_dest == 0 else 0
    features['errorBalanceOrig'] = (oldbalance_org - newbalance_orig) - amount
    features['errorBalanceDest'] = (newbalance_dest - oldbalance_dest) - amount
    
    return features

def create_gauge_chart(probability):
    """Crée un graphique de jauge pour la probabilité de fraude"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Probabilité de Fraude (%)", 'font': {'size': 20}},
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
    
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig

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
        height=250,
        showlegend=False
    )
    
    return fig

def display_results(result, amount, type_transaction, oldbalance_org, newbalance_orig, 
                   oldbalance_dest, newbalance_dest, step):
    """Affiche les résultats de la prédiction"""
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
    col1, col2 = st.columns(2)
    
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
    
    col3, col4 = st.columns(2)
    
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
    st.plotly_chart(create_gauge_chart(result['fraud_probability']), 
                  use_container_width=True)
    
    st.plotly_chart(create_probability_bar(result['fraud_probability'], 
                                          result['legit_probability']),
                  use_container_width=True)
    
    # Détails de la transaction
    st.markdown("---")
    st.subheader("📋 Détails")
    
    st.write("**Informations de Base:**")
    st.write(f"- Type: {type_transaction}")
    st.write(f"- Montant: {amount:,.2f} €")
    st.write(f"- Step: {step}")
    
    st.write("**Analyse des Comptes:**")
    balance_change_orig = oldbalance_org - newbalance_orig
    balance_change_dest = newbalance_dest - oldbalance_dest
    st.write(f"- Variation Origine: {balance_change_orig:,.2f} €")
    st.write(f"- Variation Destination: {balance_change_dest:,.2f} €")
    st.write(f"- Ratio Montant/Solde: {amount/(oldbalance_org+1)*100:.2f}%")

def main():
    # En-tête
    st.markdown('<h1 class="main-header">🔒 Système de Détection de Fraude</h1>', unsafe_allow_html=True)
    
    # Chargement du modèle
    model, scaler, error = load_model_and_scaler()
    
    if error:
        st.error(f"❌ Erreur lors du chargement du modèle: {error}")
        st.info("Assurez-vous que les fichiers 'best_fraud_detection_model.pkl' et 'scaler.pkl' sont dans le même répertoire.")
        return
    
    
    # Initialiser l'état de session pour les résultats
    if 'current_result' not in st.session_state:
        st.session_state.current_result = None
    if 'current_transaction' not in st.session_state:
        st.session_state.current_transaction = None
    
    # Tabs pour les différents modes
    tab1, tab2 = st.tabs(["📝 Saisie Manuelle", "📁 Import CSV"])
    
    # TAB 1: Saisie Manuelle
    with tab1:
        col_left, col_right = st.columns([1, 1])
        
        with col_left:
            st.header("Formulaire de Transaction")
            
            with st.form("transaction_form"):
                # Informations Générales - en colonnes
                st.subheader("Informations Générales")
                col_a, col_b = st.columns(2)
                with col_a:
                    step = st.number_input("Step (Heure)", min_value=0, max_value=744, value=1, 
                                          help="Unité de temps (1 step = 1 heure)")
                with col_b:
                    amount = st.number_input("Montant (€)", min_value=0.0, value=10000.0, step=100.0,
                                            help="Montant de la transaction")
                
                type_transaction = st.selectbox(
                    "Type de Transaction",
                    ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
                )
                
                # Comptes en colonnes
                st.subheader("Compte Origine")
                col_c, col_d = st.columns(2)
                with col_c:
                    oldbalance_org = st.number_input("Solde Initial", min_value=0.0, value=50000.0, 
                                                    step=1000.0, key="old_orig")
                with col_d:
                    newbalance_orig = st.number_input("Nouveau Solde", min_value=0.0, value=40000.0, 
                                                     step=1000.0, key="new_orig")
                
                st.subheader("Compte Destination")
                col_e, col_f = st.columns(2)
                with col_e:
                    oldbalance_dest = st.number_input("Solde Initial", min_value=0.0, value=0.0, 
                                                     step=1000.0, key="old_dest")
                with col_f:
                    newbalance_dest = st.number_input("Nouveau Solde", min_value=0.0, value=10000.0, 
                                                     step=1000.0, key="new_dest")
                
                # Bouton de soumission
                st.markdown("<br>", unsafe_allow_html=True)
                submitted = st.form_submit_button("🔍 Analyser la Transaction", type="primary", use_container_width=True)
                
                if submitted:
                    # Créer les features
                    features = create_features(
                        step, type_transaction, amount, oldbalance_org, 
                        newbalance_orig, oldbalance_dest, newbalance_dest
                    )
                    
                    # Faire la prédiction
                    with st.spinner("Analyse en cours..."):
                        result = predict_fraud(model, scaler, features)
                    
                    # Stocker les résultats dans session_state
                    st.session_state.current_result = result
                    st.session_state.current_transaction = {
                        'amount': amount,
                        'type': type_transaction,
                        'oldbalance_org': oldbalance_org,
                        'newbalance_orig': newbalance_orig,
                        'oldbalance_dest': oldbalance_dest,
                        'newbalance_dest': newbalance_dest,
                        'step': step
                    }
                    st.rerun()
        
        with col_right:
            st.header("Résultats de l'Analyse")
            
            if st.session_state.current_result is not None:
                display_results(
                    st.session_state.current_result,
                    st.session_state.current_transaction['amount'],
                    st.session_state.current_transaction['type'],
                    st.session_state.current_transaction['oldbalance_org'],
                    st.session_state.current_transaction['newbalance_orig'],
                    st.session_state.current_transaction['oldbalance_dest'],
                    st.session_state.current_transaction['newbalance_dest'],
                    st.session_state.current_transaction['step']
                )
            else:
                st.info("👈 Remplissez le formulaire et cliquez sur 'Analyser' pour voir les résultats ici")
    
    # TAB 2: Import CSV
    with tab2:
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
                        features = create_features(
                            row['step'], row['type'], row['amount'],
                            row['oldbalanceOrg'], row['newbalanceOrig'],
                            row['oldbalanceDest'], row['newbalanceDest']
                        )
                        
                        result = predict_fraud(model, scaler, features)
                        result['transaction_id'] = idx
                        results.append(result)
                        
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
                    fig_dist = px.histogram(
                        df, x='fraud_probability',
                        title='Distribution des Probabilités de Fraude',
                        nbins=50,
                        labels={'fraud_probability': 'Probabilité de Fraude'}
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
                
                with col2:
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

if __name__ == "__main__":
    main()