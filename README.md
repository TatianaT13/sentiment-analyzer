# Sentiment Analyzer

Ce projet est un analyseur de sentiments basé sur un modèle de machine learning, qui utilise le dataset Sentiment140 pour entraîner un modèle capable de prédire le sentiment (positif, neutre, négatif) de textes, comme des tweets ou des avis clients.

## Description

L'analyseur de sentiments utilise une régression logistique pour classer les textes en trois catégories de sentiments. Les textes sont prétraités pour le nettoyage et la vectorisation à l'aide du TF-IDF. Le modèle et le vectorizer sont sauvegardés pour une utilisation future dans des cas pratiques.

## Installation

1. **Cloner le dépôt :**

git clone https://github.com/TatianaT13/sentiment-analyzer.git cd sentiment-analyzer

2. **Créer un environnement virtuel et activer-le :**

python -m venv env source env/bin/activate # Sur Windows : env\Scripts\activate

3. **Installer les dépendances :**

pip install -r requirements.txt

4. **Télécharger le dataset :**

Télécharge le dataset Sentiment140 depuis Kaggle et place-le dans le dossier `data/`.

## Utilisation

### Entraînement du Modèle

Pour entraîner le modèle, exécute le script `train.py` :

python src/train.py

Cela entraînera le modèle sur le dataset et sauvegardera le modèle et le vectorizer dans le dossier `model/`.

### Évaluation du Modèle

Pour évaluer le modèle sur des données de test, exécute le script `evaluate.py` :

python src/evaluate.py

Ce script affichera les métriques d'évaluation, y compris l'accuracy et le rapport de classification.

### Prédire des Sentiments

Pour prédire les sentiments de nouveaux textes, utilise le script `predict.py` :

python src/predict.py

Entre un texte lorsque le script le demande, et il affichera le sentiment prédit.