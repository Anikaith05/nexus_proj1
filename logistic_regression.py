import numpy as np
class LogisticRegression:
    def __init__(self):
        self.learning_rate=0.01
        self.iterations=1000
        self.weights=None
        self.bias=None
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    def fit(self,x,y):
        samples_n,features_n=x.shape
        self.weights=np.zeros(features_n)
        self.bias=0
        for i in range(self.iterations):
            linear_pred=np.dot(x,self.weights)+self.bias
            y_predicted=self.sigmoid(linear_pred)
            dw=(1/samples_n)*np.dot(x.T,(y_predicted-y))
            db=(1/samples_n)*np.sum(y_predicted-y)
            self.weights-=self.learning_rate*dw
            self.bias-=self.learning_rate*db
    def predict(self,x):
        linear_pred=np.dot(x,self.weights)+self.bias
        y_predicted=self.sigmoid(linear_pred)
        y_predicted_final=[1 if i>0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_final)