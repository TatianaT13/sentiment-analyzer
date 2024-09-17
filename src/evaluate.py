# src/evaluate.py
from sklearn.metrics import accuracy_score, classification_report
from joblib import load
from preprocess import preprocess_data

def evaluate_model():
    # Charger les données prétraitées
    X_train, X_test, y_train, y_test = preprocess_data('data/training.1600000.processed.noemoticon.csv')

    # Charger le modèle et le vectorizer entraînés
    model = load('model/logistic_model.joblib')
    vectorizer = load('model/tfidf_vectorizer.joblib')

    # Transformer les données de test avec le vectorizer
    X_test_tfidf = vectorizer.transform(X_test)

    # Prédire sur l'ensemble de test
    y_pred = model.predict(X_test_tfidf)

    # Afficher les métriques d'évaluation
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    evaluate_model()