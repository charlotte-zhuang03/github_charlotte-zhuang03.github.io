import numpy as np

class LinearRegression():
    def __init__(self):
        pass
    
    def fit_gradient(self, X, y, alpha = 0.01, max_iter = 100):
        """
        Fit the model using gradient descent approach, stop when reaching maximum iteration
        """
        X_ = self.pad(X)
        done = False
        prev_loss = np.inf
        self.w = np.random.rand(X_.shape[1]) #initialize weight vector w
        self.score_history = [] # initialize score_history
        step = 0
        P = X_.T@X_
        q = X_.T@y
        while not done:
            gradient_descent = 2*(P@self.w - q)/ X_.shape[0]
            self.w = self.w - alpha*gradient_descent
            self.score_history.append(self.score(X, y))
            if step == max_iter:
                done = True
            else:
                step = step + 1

        
    def fit_analytical(self, X, y):
        """
        Fit the model using analytical formula
        """
        X_ = self.pad(X)
        w_hat = np.linalg.inv(X_.T@X_)@X_.T@y
        self.w = w_hat

    
    
    def score(self, X, y):
        """
        Compute the coefficient of determination
        """
        X_ = self.pad(X)
        y_hat = X_@self.w
        avg = np.mean(y)
        y_bar = np.array([avg]*len(y))        
        upper = np.sum((y_hat - y)**2)
        lower = np.sum((y_bar - y)**2)
        c = 1 - (upper/lower)
        return c
        

    def pad(self, X):
        """
        Append 1s at the end of X to ensure that X contains a column of 1s prior to any major computations
        """
        return np.append(X, np.ones((X.shape[0], 1)), 1)