import sys
import os
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pickle
from knn import KNN
def accuracy(y_true, y_pred):
    acc = np.sum(y_true == y_pred) / len(y_true)
    return acc
iris = datasets.load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)
k_value = 3
knn_model = KNN()
knn_model.fit(X_train, y_train)
print(f"KNN model with k={k_value} trained successfully.")
predictions = knn_model.predict(X_test)
model_accuracy = accuracy(y_test, predictions)
print(f"KNN classification accuracy: {model_accuracy:.4f}")
with open('./knn_model.pkl', 'wb') as f:
    pickle.dump(knn_model, f)
print("KNN model trained and saved to 'saved_models/knn_model.pkl'.")