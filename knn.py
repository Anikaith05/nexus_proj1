import numpy as np
from collections import Counter
def euclidean_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))
class KNN:
    def __init__(self):
        self.k=3
    def fit(self,x,y):
        self.X_train=x
        self.Y_train=y
    def predict(self,x):
        y_pred=[self.predict1(i) for i in x]
        return np.array(y_pred)
    def predict1(self,x):
        distances=[euclidean_distance(x,x_train) for x_train in self.X_train]
        k_indices=np.argsort(distances)[:self.k]
        k_nearest_labels=[self.Y_train[i] for i in k_indices]
        most_common=Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]