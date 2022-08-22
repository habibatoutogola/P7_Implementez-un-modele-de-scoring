import pandas as pd
import numpy as np
import shap
import requests
import pickle
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt

# Chargement des données
df = pd.read_csv('data_scoring.csv', index_col=0)
df_feats = pd.read_csv('shap_values_lgb.csv', index_col=0)
df_client= pd.read_csv('df_client')
df_TARGET = df.TARGET.copy()
df.drop(columns='TARGET', inplace=True)
index1500 = [i for i in range(0,1500)]
df.index = index1500

#chargement du modéle

with open('feats_sample_explainer_shap.p', 'rb') as f3:
    explainer_shap = pickle.load(f3)


# Solvabilité client sous forme de pie

def client(df, df_client, df_feats):
    st.set_option('deprecation.showPyplotGlobalUse', False)

# prod
    url = "lien heroku"

    explainer_shap = -0.014323537009938003

    url_requests = url+"predict/"
    response = requests.get(url_requests)
    if response:
        list_client_id = response.json()['list_client_id']
        list_client_id = sorted(list_client_id)
    else:
        list_client_id = ['000000']
        print("erreur web : ", response)
def update_sk(sk_id):
        predict_proba_1=0.5
        if sk_id in list_client_id:
            url_pred = url + "predict/" + sk_id
            response = requests.get(url_pred)
            if response:
                predict_proba_1 = float(response.json()['predict_proba_1'])
            else:
                print("erreur web : ", response)

        gauge_predict = go.Figure(go.Indicator( mode = "gauge+number",
                                                value = predict_proba_1*100,
                                                domain = {'x': [0, 1], 'y': [0, 1]},
                                                gauge = {
                                                    'axis': {'range': [0, 100], 'tickwidth': 0.2, 'tickcolor': "darkblue"},
                                                    'bgcolor': "lightcoral",
                                                    'steps': [
                                                        {'range': [0, 40], 'color': 'lightgreen'},
                                                        {'range': [40, 60], 'color': 'palegoldenrod'}
                                                    ],
                                                    'threshold': {
                                                        'line': {'color': "red", 'width': 4},
                                                        'thickness': 0.75,
                                                        'value': 100}},
                                                title = {'text': f"client {sk_id}"}))

        return gauge_predict
    
#affichage formulaire
st.title('Dashboard Scoring Credit')
st.markdown("Prédictions de scoring client, notre seuil de choix est de 40 %")
    
# Information relative à un client 

option_sk = st.selectbox('Selectionner un numero de client',list_client_id)

row_df_sk = ( df['SK_ID_CURR'] == int(option_sk))
row_appli_sk = ( df_client['SK_ID_CURR'] == int(option_sk))

st.subheader("Client Information")
sex = df_client.loc[row_appli_sk, ['CODE_GENDER']].values[0][0]
st.write("Sex :",sex)
age = int(np.trunc(- int(df_client.loc[row_appli_sk, ['DAYS_BIRTH']].values)/365))
st.write("Age :", age)
family = df_client.loc[row_appli_sk, ['NAME_FAMILY_STATUS']].values[0][0]
st.write("Family status :", family)
education = df_client.loc[row_appli_sk, ['NAME_EDUCATION_TYPE']].values[0][0]
st.write("Education type :", education)
occupation = df_client.loc[row_appli_sk, ['OCCUPATION_TYPE']].values[0][0]
st.write("Occupation type :", occupation)
Own_realty = df_client.loc[row_appli_sk, ['FLAG_OWN_REALTY']].values[0][0]
st.write("Client owns a house or flat :", Own_realty)
income = str(df_client.loc[row_appli_sk, ['AMT_INCOME_TOTAL']].values[0][0])
st.write("Income of the client :", income)
income_perc = df_client.loc[row_df_sk, ['ANNUITY_INCOME_PERC']].values[0][0]
st.write(f"Loan annuity / Income of the client : {income_perc*100:.2f} %")

st.subheader("Credit Information")
type_contract = str(df_client.loc[row_appli_sk, ['NAME_CONTRACT_TYPE']].values[0][0])
st.write("Contract type :", type_contract)
credit = str(df_client.loc[row_appli_sk, ['AMT_CREDIT']].values[0][0])
st.write("Credit amount of the loan :", credit)
annuity = df_client.loc[row_appli_sk, ['AMT_ANNUITY']].values[0][0] / 12
st.write(f"Loan monthly : {annuity:.1f}")
income_credit_perc = df_client.loc[row_df_sk, ['INCOME_CREDIT_PERC']].values[0][0]
st.write(f"Income of the client / Credit amount of the loan : {income_credit_perc*100:.2f} %")


st.subheader("Retour Prediction")
st.write("""
    **le retour est un score de 0 à 100. Le seuil de refus est à 50.**
    
    1. Un retour en dessous de 40 est une acceptation du crédit.
    
    2. Un retour au dessus de 60 est un refus du crédit.
    
    3. Pour un score entre 40 et 60, on va regarder l'interpretabilité de la prediction pour aider au choix. 
    
    """)
    
    
fig = update_sk(option_sk)
st.plotly_chart(fig)

st.subheader("INTERPRETATION VALEURS SHAPLEY")
st.write("""
        ** Les variables sont classes de haut en bas par ordre d'importance dans l'interpretation.** 
        
        **La  couleur pour chaque variable est un indicateur de l'influence sur la prediction.** 
        
        **Les variables en rouge font augmenter le score et donc le risque de defaut de paiement.**
         
        """)

class ShapObject:

        def __init__(self, base_values, data, values, feature_names):
            self.base_values = base_values # Single value
            self.data = data # Raw feature values for 1 row of data
            self.values = values # SHAP values for the same row of data
            self.feature_names = feature_names # Column names


def update_shap(sk_id, fig):
        ind = df[df.SK_ID_CURR == int(sk_id)].index.values[0]
        shap_object = ShapObject(base_values=explainer_shap,
                                 values=df.loc[ind].values,
                                 feature_names=df.columns,
                                 data=df.iloc[ind,:])
        df_shap = pd.DataFrame(np.abs(df_feats.loc[ind].values), df.columns, columns=['abs_shap'])
        list_shap_feats = list(df_shap.sort_values(by='abs_shap', ascending=False).head(20).index)
        #        list_shap_feats = sorted(list_shap_feats)
        return shap.waterfall_plot(shap_object, max_display=20)

fig, ax = plt.subplots(nrows=1, ncols=1)
fig, df_top_feats = update_shap(option_sk, fig)
st.pyplot(fig)
st.write("""
    **RAPPEL DE LA SIGNIFICATION DES VARIABLES
    """)
    
st.table(df_top_feats)

st.image('shap_value_importance.png', width=600)
st.image('lime_value_importance.png', width=600)

# Comparaison à d'autres clients
#on peut utiliser lime
# afficher les valeurs moyennes des variables

with open('Neighbors.p', 'wb') as f4:
   tree= pickle.load(f4)







def main():

    page = st.sidebar.selectbox("Choose a page", ['Home','Information Général', 'Information Client', 'Comparaison Client'])
    



if __name__ == '__main__':
    main()