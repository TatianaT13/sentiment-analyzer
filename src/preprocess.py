# src/preprocess.py
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

# Télécharge les stopwords de nltk
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_data(filepath):
    # Charger les données
    df = pd.read_csv(filepath, encoding='ISO-8859-1', header=None)
    df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']

    # Garder seulement les colonnes utiles
    df = df[['target', 'text']]

    # Remplacer les valeurs des cibles (0 = négatif, 2 = neutre, 4 = positif)
    df['target'] = df['target'].replace({0: 0, 2: 1, 4: 2})

    # Nettoyer les tweets
    df['text'] = df['text'].str.lower().str.replace('[^a-z\s]', '', regex=True)

    # Supprimer les stopwords
    df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['target'], test_size=0.2, random_state=42)

    return X_train.tolist(), X_test.tolist(), y_train.tolist(), y_test.tolist()