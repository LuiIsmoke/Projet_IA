import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Chargement du modèle (gestion des erreurs incluse)
try:
    model = joblib.load('decision_tree_model.pkl')
except FileNotFoundError:
    st.error("Le fichier du modèle 'decision_tree_model.pkl' est introuvable.")
    st.stop()

# Chargement du fichier CSV pour l'encodage (gestion des erreurs)
try:
    df_encodage = pd.read_csv('DatasetmalwareExtrait.csv')
    # Encodage des colonnes de type 'object' du df_encodage
    for col in df_encodage.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df_encodage[col] = le.fit_transform(df_encodage[col])
except FileNotFoundError:
    st.warning("Le fichier 'DatasetmalwareExtrait.csv' est introuvable. L'encodage des variables ne pourra pas être restauré fidèlement. Assurez-vous d'encoder les données d'entrée manuellement et de les convertir en numérique.")
    df_encodage=None

# Fonction de prétraitement (ENCRODAGE UNIQUEMENT - PAS DE MISE A L'ECHELLE)
def preprocess_input(input_data):
    input_df = pd.DataFrame([input_data])

    if df_encodage is not None:  # si le fichier CSV a été trouvé
        for column in input_df.select_dtypes(include=['object']).columns:
            try:
                le = LabelEncoder()
                le.fit(df_encodage[column])
                input_df[column] = le.transform(input_df[column])
            except KeyError as e:
                st.error(f"Colonne '{e.args[0]}' non trouvée dans les données d'entraînement. Veuillez vérifier les noms des caractéristiques.")
                st.stop()
    return input_df

# Interface Streamlit
st.title('Détection de Malware')

input_data = {}
input_data['Feature1'] = st.number_input('Valeur de la caractéristique 1', value=0.0)
input_data['Feature2'] = st.text_input('Texte pour la caractéristique 2')
input_data['Feature3'] = st.selectbox('Choix pour la caractéristique 3', ['valeur1','valeur2','valeur3'])
# ... ajoute tous les champs nécessaires

# Bouton de prédiction
if st.button('Prédire'):
    try:
        input_df = preprocess_input(input_data)
        prediction = model.predict(input_df)
        st.write('Prédiction :', prediction[0])

        # Affichage des probabilités (si disponible)
        try:
            probabilities = model.predict_proba(input_df)
            st.write('Probabilités :', probabilities)
            st.write(f"Probabilité Classe 0: {probabilities[0][0]:.2f}")
            st.write(f"Probabilité Classe 1: {probabilities[0][1]:.2f}")
        except AttributeError:
            pass # Le modèle n'a pas de méthode predict_proba
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
