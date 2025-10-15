import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pickle
from logistic_regression import LogisticRegression
X_log, y_log = datasets.make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=2, random_state=42, n_clusters_per_class=1)
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_log, y_log, test_size=0.2, random_state=123)

log_reg = LogisticRegression()
log_reg.fit(X_train_log, y_train_log)

with open('./logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(log_reg, f)
print("Logistic Regression model trained and saved.")

