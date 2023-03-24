import numpy as np
from scipy.optimize import minimize


class KernelLogisticRegression():
    def __init__(self, kernel, **kernel_kwargs):
        self.kernel = kernel
        self.kernel_kwargs = kernel_kwargs
    
    
    def fit(self, X, y, alpha = 0.1, max_epochs = 1000):
        X_ = self.pad(X)
        self.X_train = X_
        km = self.kernel(X_, X_, **self.kernel_kwargs)
        self.v = np.random.rand(X_.shape[0])
        
        done = False
        prev_loss = np.inf

        self.loss_history = []
        self.score_history = []
        
        step = 0
        while not done:
            gradient_descent = self.empirical_risk(X, y)
            self.v = self.v - alpha*gradient_descent
            
            new_loss = self.logistic_loss(km, y)
            
            self.score_history.append(self.score(X, y))
            self.loss_history.append(new_loss)
            
            if step == max_epochs:
                done = True
            else:
                step = step + 1

    
    
    def predict(self, X):
        X_ = self.pad(X)
        km = self.kernel(self.X_train, X_, **self.kernel_kwargs)
        return np.where(np.dot(self.v, km) > 0, 1, 0)

    

    def score(self, X, y):
        X_ = self.pad(X)
        correct = 0
        y_pred = self.predict(X)
        for i in range(X_.shape[0]):
            if y_pred[i] == y[i]:
                correct = correct + 1
        return correct/X_.shape[0]
        
    
    def empirical_risk(self, X, y):
        X_ = self.pad(X)
        loss_sum = np.zeros(X_.shape[0])
        km = self.kernel(X_, X_, **self.kernel_kwargs)
        for i in range(X_.shape[0]):
            sigmoid_v_kxi = 1/(1+np.exp(-(np.dot(self.v, km[i]))))
            loss_sum = loss_sum + (sigmoid_v_kxi - y[i])*km[i]
        return loss_sum / X_.shape[0]
        #return self.logistic_loss(km, y).mean()
    
    
    def logistic_loss(self, km, y):
        """
        Return the average value of loss
        """
        y_hat = np.dot(self.v, km)
        sigmoid_y_hat = 1/(1+np.exp(-y_hat))
        loss = -y*np.log(sigmoid_y_hat) - (1-y)*np.log(1-sigmoid_y_hat)
        return np.sum(loss).mean()
        
        
    
    def pad(self, X):
        return np.append(X, np.ones((X.shape[0], 1)), 1)
        