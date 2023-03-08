import numpy as np

class LogisticRegression():
    def __init__(self):
        pass
    
    def pad(self, X):
        """
        Append 1s at the end of X to ensure that X contains a column of 1s prior to any major computations
        """
        return np.append(X, np.ones((X.shape[0], 1)), 1)
    
    def fit(self, X, y, alpha = 0.1, max_epochs = 1000):
        """
        Update the weight vector, loss history and score history that keeps track of the weight w, \\
        loss of current approximation and accuracy of approximation respectively
        
        Parameters:
        X: a matrix of predictor variables, with X.shape[0] observations and X.shape[1] features
        y: a vector of binary labels 0 and 1
        alpha: learning rate
        max_epochs: number of loops going through, set default to 1000
        
        Return:
        None
        """
        X_ = self.pad(X)
        done = False
        prev_loss = np.inf
        self.w = np.random.rand(X_.shape[1])
        self.loss_history = []
        self.score_history = []
        step = 0
        while not done:
            gradient_descent = self.gradient_descent(X, y)
            self.w = self.w - alpha*gradient_descent
            
            new_loss = self.loss(X, y)
            
            self.score_history.append(self.score(X, y))
            self.loss_history.append(new_loss)
            
            #if np.isclose(new_loss, prev_loss):
                #done = True
            #else:
                #prev_loss = new_loss
            if step == max_epochs:
                done = True
            else:
                step = step + 1
        
    def fit_stochastic(self, X, y, alpha, momentum, batch_size, max_epochs):
        """
        Update the weight vector and the loss history that keeps track of the change in loss.
        
        Parameters:
        X: a matrix of predictor variables, with X.shape[0] observations and X.shape[1] features
        y: a vector of binary labels 0 and 1
        alpha: learning rate
        momentum: a variable that mixes the current gradient direction with the previously taken step to accelerate the program
        batch_size: size of the random subset S
        max_epochs: number of loops going through, set default to 1000
        
        Return:
        None
        """
        
        
        
        n = X.shape[0]
        X_ = self.pad(X)
        
        previous_w = np.zeros(X_.shape[1])
        self.w = np.random.rand(X_.shape[1])
                                 
        
        self.loss_history = []
        
        for j in np.arange(max_epochs):
            
            order = np.arange(n)
            np.random.shuffle(order)
            
            for batch in np.array_split(order, n // batch_size + 1):
                x_batch = X[batch, :]
                y_batch = y[batch]
                sto_gradient = self.stochastic_gradient(x_batch, y_batch, x_batch.shape[0])
                if momentum == True:
                    beta = 0.8
                else:
                    beta = 0
                current_w = self.w
                self.w = current_w - alpha*sto_gradient + beta*(current_w - previous_w)
                previous_w = current_w
                current_w = self.w
                
                
                                                                                          
            self.loss_history.append(self.loss(X, y))
                

    
    def predict(self, X):
        """
        Return a list of vector showing whether the dot product of w and X is greater than 0 or smaller than 0
        
        0: smaller than 0
        1: greater than 0
        """
        return np.where(np.dot(X, self.w) > 0, 1, 0)
    
    
    def predict2(self, X):
        """
        Return a list of vector showing whether the dot product of w and X is greater than 0 or smaller than 0
        
        -1: smaller than 0
        1: greater than 0
        """
        return np.sign(np.dot(X, self.w))

    
    def score(self, X, y):
        """
        Return the average value of accuracy
        """
        X_ = self.pad(X)
        y_tilde = 2*y-1
        return np.dot(self.predict2(X_),y_tilde)/X_.shape[0]

   
    def loss(self, X, y):
        """
        Return the average value of loss
        """
        X_ = self.pad(X)
        y_hat = np.dot(X_, self.w)
        sigmoid_y_hat = 1/(1+np.exp(-y_hat))
        loss = -y*np.log(sigmoid_y_hat) - (1-y)*np.log(1-sigmoid_y_hat)
        return np.sum(loss)/X_.shape[0]

    
    def gradient_descent(self, X, y):
        """
        Compute the gradient descent for the fit() method
        """
        X_ = self.pad(X)
        loss_sum = np.zeros(X_.shape[1])
        for i in range(X_.shape[0]):
            sigmoid_w_xi = 1/(1+np.exp(-(np.dot(self.w, X_[i]))))
            loss_sum = loss_sum + (sigmoid_w_xi - y[i])*X_[i]
        return loss_sum / X_.shape[0]
    
    def stochastic_gradient(self, X, y, S):
        """
        Compute the stochastic gradient for the fit_stochastic() method
        """
        X_ = self.pad(X)
        sto_loss_sum = np.zeros(X_.shape[1])
        for i in range(S):
            sto_sigmoid_w_xi = 1/(1+np.exp(-(np.dot(self.w, X_[i]))))
            sto_loss_sum = sto_loss_sum + (sto_sigmoid_w_xi - y[i])*X_[i]
        return sto_loss_sum / S
        
        
        
        

        
        
        
    
    


        
        
    

        
        
        

    