import numpy as np
class LinearRegression:
    def __init__(self):
        self.learning_rate=0.01
        self.iterations=1000
        self.weights=None
        self.bias=None
    def fit(self,x,y):
        samples_n,features_n=x.shape
        self.weights=np.zeros(features_n)
        self.bias=0
        for i in range(self.iterations):
            y_predicted=np.dot(x,self.weights)+self.bias
            dw=(1/samples_n)*np.dot(x.T,(y_predicted-y))
            db=(1/samples_n)*np.sum(y_predicted-y)
            self.weights-=self.learning_rate*dw
            self.bias-=self.learning_rate*db
    def predict(self,x):
        return np.dot(x,self.weights)+self.bias