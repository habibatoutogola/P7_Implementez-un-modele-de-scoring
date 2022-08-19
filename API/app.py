from flask import Flask, jsonify, request, render_template
import pandas as pd
import numpy as np
import pickle
import os
os.getcwd()

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)


with open('lgbm_GridCV.p', 'rb') as f2:
     print("utilisation modele lgbm_GridCV")
     grid_lgbm = pickle.load(f2)

df = pd.read_csv('data_scoring.csv', index_col=0)
df.drop(columns='TARGET', inplace=True)
num_client = df.SK_ID_CURR.unique()


@app.route('/')
def home():
    return ("L'application qui prédit l'accord du crédit")

#json_ = request.json()
#print(json_)
            
#num_client=df[df['SK_ID_CURR']==int(json_['SK_ID_CURR'])].drop(['SK_ID_CURR','TARGET'],axis=1)
#print(num_client)

            

@app.route('/predict/')
def predict():
    
    return jsonify({"model": "'lgbm_GridCV",
                    "list_client_id" : list(num_client.astype(str))}) 
 
@app.route('/predict/<int:sk_id>')
def predict_get(sk_id):

    #sk_id=100002
    #sk_id=request.args.get('sk_id', default=1, type=int)
    """
    Returns
    liste des clients dans le fichier
    """
    #return jsonify({"model": "'lgbmc10_GridCV",
                   # "list_client_id" : list(num_client.astype(str))})
    #sk_id=100002 # affiché l'identifient du client

    if sk_id in num_client:
       predict = grid_lgbm.predict(df[df['SK_ID_CURR']==sk_id])[0]
       predict_proba = grid_lgbm.predict_proba(df[df['SK_ID_CURR']==sk_id])[0]
       predict_proba_0 = str(predict_proba[0])
       predict_proba_1 = str(predict_proba[1])
    
    else:
     predict = predict_proba_0 = predict_proba_1 = "client inconnu"
    return jsonify({ 'retour_prediction' : str(predict), 'predict_proba_0': predict_proba_0,
                 'predict_proba_1': predict_proba_1 })


#@app.route('/predict_get/',methods=['POST'])
#def predict_get(sk_id):
    """
    Parameters
    ----------
    sk_id : numero de client
    Returns
    -------
    prediction  0 pour paiement OK
                1 pour defaut de paiement
    """#récupération id client depuis argument urlsk_id = 100002
    if sk_id in num_client:
        predict = grid_lgbm.predict(df[df['SK_ID_CURR']==sk_id])[0]
        predict_proba = grid_lgbm.predict_proba(df[df['SK_ID_CURR']==sk_id])[0]
        predict_proba_0 = str(predict_proba[0])
        predict_proba_1 = str(predict_proba[1])
        
    else:
         predict = predict_proba_0 = predict_proba_1 = "client inconnu"
         
    return jsonify({
                   'retour_prediction' : str(predict), 
                   'predict_proba_0': predict_proba_0,
                   'predict_proba_1': predict_proba_1 })
    

#lancement de l'application
if __name__ == '__main__':
        app.run(debug=True ,port=8080,use_reloader=False)