import numpy as np
import sklearn


class Linear_regression:
    def __init__(self):
        self.theta_0=None
        self.theta_1=None

    def hypothesis(self,x):
        return self.theta_0 + self.theta_1*x
    def grad_theta0(self,x,y):
        y_pred=self.hypothesis(x)
        return (y_pred-y)

    def grad_theta1(self, x, y):
        y_pred = self.hypothesis(x)
        return (y_pred - y)*x

    def fit(self,x,y,epochs=1,learningrate=.01):
        self.theta_0=0.1
        self.theta_1=0.1
        m=x.shape[0]
        for i in range(epochs):
            d_theta0 = 0
            d_theta1 = 0


            for(x,y) in zip(x,y):
                d_theta0 = d_theta0 + self.grad_theta0(x, y)
                d_theta1 = d_theta1 + self.grad_theta1(x, y)
            self.theta_0=self.theta_0-learningrate*d_theta0/m
            self.theta_1=self.theta_1-learningrate*d_theta1/m

    def predict(self,X):
        y_pred=[]
        for x in X:
            y_pred.append(self.hypothesis(x))
        return np.array(y_pred)


