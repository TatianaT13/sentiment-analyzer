# src/train.py
from sklearn.linear_model import LogisticRegression
from joblib import dump
from preprocess import preprocess_data
from sklearn.feature_extraction.text import TfidfVectorizer

def train_model():
    # Charger et prétraiter les données
    X_train, X_test, y_train, y_test = preprocess_data('data/training.1600000.processed.noemoticon.csv')

    # Créer le vectorizer TF-IDF et transformer les données
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Entraîner le modèle de régression logistique
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    # Sauvegarder le modèle et le vectorizer
    dump(model, 'model/logistic_model.joblib')
    dump(vectorizer, 'model/tfidf_vectorizer.joblib')

    print("Modèle et vectorizer entraînés et sauvegardés avec succès.")

if __name__ == "__main__":
    train_model()