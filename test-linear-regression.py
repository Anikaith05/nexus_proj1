import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pickle
from linear_regression import LinearRegression
X_lin, y_lin = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
X_train_lin, X_test_lin, y_train_lin, y_test_lin = train_test_split(X_lin, y_lin, test_size=0.2, random_state=123)
lin_reg = LinearRegression()
lin_reg.fit(X_train_lin, y_train_lin)
with open('./linear_regression_model.pkl', 'wb') as f:
    pickle.dump(lin_reg, f)
print("Linear Regression model trained and saved.")