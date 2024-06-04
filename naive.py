import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Cargar los conjuntos de datos
iris = load_iris()
wine = load_wine()
cancer = load_breast_cancer()

datasets = {
    "Iris": iris,
    "Wine": wine,
    "Breast Cancer": cancer
}

def evaluate_naive_bayes(dataset, dataset_name):
    X = dataset.data
    y = dataset.target

    # Hold-Out 70/30
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Naïve Bayes Clasificador
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # 10-Fold Cross Validation
    cv_scores = cross_val_score(nb, X, y, cv=10, scoring='accuracy')

    # Resultados
    print(f"\n{dataset_name} Dataset Results (Naïve Bayes):")
    print(f"Naïve Bayes clasificador con Accuracy (Hold-Out): {accuracy:.2f}")
    print(f"Naïve Bayes clasificador con Confusion Matrix (Hold-Out):\n{cm}")
    print(f"Naïve Bayes clasificador con Accuracy (10-Fold CV): {cv_scores.mean():.2f}")

for name, dataset in datasets.items():
    evaluate_naive_bayes(dataset,name)