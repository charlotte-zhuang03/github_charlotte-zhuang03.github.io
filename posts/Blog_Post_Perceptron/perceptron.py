import numpy as np

class Perceptron:
    def __init__(self):
        pass


    
    def fit(self, X, y, max_steps=1000):
        """
        Update the weight vector w and the history vector which keeps track of the change of accuracy
        
        Parameters:
        X: a matrix of predictor variables, with X.shape[0] observations and X.shape[1] features
        y: a vector of binary labels 0 and 1
        max_steps: number of loops going through, set default to 1000
        
        Return:
        None
        """
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1) # initialize X_ by appending 1 at the end of X    
        
        # Initialize instance variables
        
        self.w = np.random.rand(X_.shape[1]) # initialize a random weight vector w
        self.history = [] # initialize history vector to be an empty vector
        
        for _ in range(max_steps):
            i = np.random.randint(X_.shape[0]) # pick a random index i within the range of number of observations
            
            y_tilde = 2*y-1 # transform y from a vector containing 0 and 1 to a vector containing -1 and 1
            
            determine = np.where((y_tilde[i]*self.predict2(X_[i])) < 0, 1, 0) 
            # determine = 0 if predicted and actual y have same sign
            # determine = 1 if predicted and actual y have different sign, which triggers the update of w

            self.w = self.w + determine*y_tilde[i]*X_[i] # update w

            self.history.append(self.score(X, y)) # add recent score value to history

    
    def predict(self, X):
        """
        Return a list of vector showing whether the dot product of w and X is greater than 0 or smaller than 0
        
        0: smaller than 0
        1: greater than 0
        """
        return (np.sign(np.dot(X, self.w))+1)//2
    
    def predict2(self, X):
        """
        Return a list of vector showing whether the dot product of w and X is greater than
        
        -1: smaller than 0
        1: greater than 0
        """
        return np.sign(np.dot(X, self.w))
        

    
    def score(self, X, y):
        """
        Return the average value of accuracy
        """
        X_ = np.append(X, np.ones((X.shape[0], 1)), 1)
        y_tilde = 2*y-1
        #check = np.where(np.dot(self.predict(X_),y) > 0, 1, 0)
        return np.dot(self.predict2(X_),y_tilde)/X_.shape[0]
        #return np.dot(self.predict(X_),y)/X_.shape[0]
        #return np.dot(self.predict(X_),y)/X_.shape[0]
        #return check/X_.shape[0]
