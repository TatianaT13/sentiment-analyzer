# src/predict.py
from joblib import load
import re
import nltk
from nltk.corpus import stopwords

# Télécharger les stopwords si nécessaire
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def load_model_and_vectorizer():
    # Charger le modèle et le vectorizer
    model = load('model/logistic_model.joblib')
    vectorizer = load('model/tfidf_vectorizer.joblib')
    return model, vectorizer

def predict_sentiment(text, model, vectorizer):
    # Prétraiter le texte comme dans l'entraînement
    text = text.lower()
    text = re.sub('[^a-z\s]', '', text)  # Utilisation de re.sub pour le nettoyage
    text = ' '.join([word for word in text.split() if word not in stop_words])

    # Transformer le texte avec le vectorizer
    text_tfidf = vectorizer.transform([text])

    # Prédire le sentiment
    prediction = model.predict(text_tfidf)

    # Convertir la prédiction numérique en label
    sentiment_map = {0: 'Négatif', 1: 'Neutre', 2: 'Positif'}
    return sentiment_map[prediction[0]]

if __name__ == "__main__":
    model, vectorizer = load_model_and_vectorizer()

    # Exemple d'utilisation
    text = input("Entrez un texte à analyser : ")
    sentiment = predict_sentiment(text, model, vectorizer)
    print(f"Le sentiment prédit est : {sentiment}")