# app.py
import streamlit as st
from joblib import load
import re
import nltk
from nltk.corpus import stopwords

# Télécharger les stopwords si nécessaire
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Charger le modèle et le vectorizer
model = load('model/logistic_model.joblib')
vectorizer = load('model/tfidf_vectorizer.joblib')

def preprocess_text(text):
    # Prétraiter le texte comme dans l'entraînement
    text = text.lower()
    text = re.sub('[^a-z\s]', '', text)  # Nettoyage de base
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def predict_sentiment(text):
    # Transformer le texte avec le vectorizer
    text_tfidf = vectorizer.transform([text])

    # Prédire le sentiment
    prediction = model.predict(text_tfidf)

    # Convertir la prédiction numérique en label
    sentiment_map = {0: 'Négatif', 1: 'Neutre', 2: 'Positif'}
    return sentiment_map[prediction[0]]

# Interface utilisateur avec Streamlit
st.title("Analyseur de Sentiments")

st.write("Entrez un texte pour analyser le sentiment (Positif, Neutre, Négatif).")

# Champ de texte pour l'entrée utilisateur
user_input = st.text_area("Entrez votre texte ici:")

if st.button("Analyser"):
    # Prétraiter et prédire le sentiment
    preprocessed_text = preprocess_text(user_input)
    sentiment = predict_sentiment(preprocessed_text)
    
    # Afficher le résultat
    st.write(f"Le sentiment prédit est : **{sentiment}**")
