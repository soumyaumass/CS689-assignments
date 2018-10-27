import numpy as np
from scipy.optimize import fmin_l_bfgs_b

class InformativePriorLogisticRegression:
    r"""Logistic regression with general spherical Gaussian prior.

    Arguments:
        w0 (ndarray, shape = (n_features,)): coefficient prior
        b0 (float): bias prior
        reg_param (float): regularization parameter $\lambda$ (default: 0)
    """

    def __init__(self, w0=None, b0=0, reg_param=0):
        self.w0 = w0   # prior coefficients
        self.b0 = b0   # prior bias
        self.reg_param = reg_param   # regularization parameter (lambda)
        self.set_params(np.zeros_like(w0), 0)

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                Training output vector. Each entry is either -1 or 1.
        """
        wb = np.append(self.w0,self.b0)
        # Calling fmin_l_bfgs_b to optimize objective function
        w_optimum,f,d = fmin_l_bfgs_b(func=self.objective,fprime=self.objective_grad,x0=wb,args=(X,y))
        self.w=w_optimum[:-1] # Extracting w from optimum wb 
        self.b=w_optimum[-1]  # Extracting b from optimum wb
        
        return self

    def predict(self, X):
        """Predict class labels for samples in X.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)): test data

        Returns:
            y (ndarray, shape = (n_samples,):
                Predictions with values in {-1, +1}.
        """
        ybar = (X.dot(self.w) + self.b)
        return (ybar/np.absolute(ybar)).astype(np.int8)

    def objective(self, wb, X, y):
        """Compute the objective function

        Arguments:
            wb (ndarray, shape = (n_features + 1,)):
                concatenation of the coefficient and the bias parameters
                wb = [w, b]
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                training label.

        Returns:
            loss (float):
                the objective function evaluated on w.
        """
        X = np.hstack((X,np.ones((X.shape[0],1))))
        ybar = X.dot(wb)
        loss = 0
        # Calculating loss
        loss = np.sum(np.log(1 + np.exp(-y*ybar)))
        loss += self.reg_param * ((np.linalg.norm(wb[:-1]-self.w0)**2) + ((wb[-1]-self.b0)**2))

        return loss

    def objective_grad(self, wb, X, y):
        """Compute the derivative of the objective function

        Arguments:
            wb (ndarray, shape = (n_features + 1,)):
                concatenation of the coefficient and the bias parameters
                wb = [w, b]
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                training label.

        Returns:
            loss_grad (ndarray, shape = (n_features + 1,)):
                derivative of the objective function with respect to w.
        """
        loss_grad = np.zeros_like(wb)
        X = np.hstack((X,np.ones((X.shape[0],1))))
        ybar = X.dot(wb)
        
        # Calculating gradient
        for i in range(y.shape[0]):
            loss_grad += -(y[i]*X[i,:])/(1 + np.exp(y[i]*ybar[i]))
            
        loss_grad += 2 * self.reg_param * (wb - np.append(self.w0,self.b0))
        return loss_grad

    def get_params(self):
        """Get parameters for the model.

        Returns:
            A tuple (w,b) where w is the learned coefficients (ndarray)
            and b is the learned bias (float).
        """
        return self.w, self.b

    def set_params(self, w, b):
        r"""Set the parameters of the model.

        Arguments:
            w (ndarray, shape = (n_features,)): coefficient prior
            b (float): bias prior
            reg_param (float): regularization parameter $\lambda$ (default: 0)
        """
        self.w = w
        self.b = b

def main():
    
    np.random.seed(1)

    train_X = np.load('../data/q2_train_X.npy')
    train_y = np.load('../data/q2_train_y.npy')
    test_X = np.load('../data/q2_test_X.npy')
    test_y = np.load('../data/q2_test_y.npy')
    w0 = np.load('../data/q2_w_prior.npy').squeeze()
    b0 = np.load('../data/q2_b_prior.npy')
    
    results = {}        # Dictionary to store accuracy values for different batches
    num_train = train_X.shape[0]
    for batch_size in range(10,410,10):  # Increasing batch_size in steps of 10
        mask = np.random.choice(num_train,batch_size)  # Taking random samples of fixed batch size
        X_batch = train_X[mask]
        y_batch = train_y[mask]
          
        
        clf = InformativePriorLogisticRegression(w0,b0,reg_param=0)
        clf.fit(X_batch,y_batch)     # Training with samples of training cases
        pred_y = clf.predict(test_X)
        
        count = 0
        for i in range(test_y.shape[0]):
            if(test_y[i] == pred_y[i]):
                count += 1
        
        accuracy0 = 100*count/test_y.shape[0] # Accuracy for lambda = 0
        
        clf = InformativePriorLogisticRegression(w0,b0,reg_param=10)
        clf.fit(X_batch,y_batch)
        pred_y = clf.predict(test_X)
        
        count = 0
        for i in range(test_y.shape[0]):
            if(test_y[i] == pred_y[i]):
                count += 1
                
        accuracy10 = 100*count/test_y.shape[0] # Accuracy for lambda = 10
        results[batch_size] = (accuracy0, accuracy10)
    
    batch_sizes = results.keys()
    acc0 = [val[0] for val in results.values()]
    acc10 = [val[1] for val in results.values()]
    
    # Plotting test accuracy vs batch size of training data
    import matplotlib.pyplot as plt
    plt.scatter(batch_sizes,acc0)
    plt.plot(batch_sizes,acc0,'g-',label='Lambda = 0')
    plt.scatter(batch_sizes,acc10)
    plt.plot(batch_sizes,acc10,'b-',label='Lambda = 10')
    plt.title('Test Accuracy vs Batch size of training data')
    plt.xlabel('Training Batch Size')
    plt.ylabel('Test Accuracy in percentage')
    plt.legend(loc='lower right')
    fig = plt.gcf()
    fig.set_size_inches(12,8)
    plt.show()

if __name__ == '__main__':
    main()