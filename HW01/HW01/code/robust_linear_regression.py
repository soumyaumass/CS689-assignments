import numpy as np
from scipy.optimize import fmin_l_bfgs_b

class RobustLinearRegression:
    """Generalized robust linear regression.

    Arguments:
        delta (float): the cut-off point for switching to linear loss
        k (float): parameter controlling the order of the polynomial part of
            the loss
    """

    def __init__(self, delta, k):
        self.delta = delta   # cut-off point
        self.k = k    # polynomial order parameter
        self.w = 0
        self.b = 0

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                Training output vector. Each entry is either -1 or 1.
        """
        wb = np.append(self.w,self.b)
        # Calling fmin_l_bfgs_b to optimize objective function
        w_optimum,f,d = fmin_l_bfgs_b(func=self.objective,fprime=self.objective_grad,x0=wb,args=(X,y))
        self.w=w_optimum[:-1] # Extracting w from optimum wb
        self.b=w_optimum[-1]  # Extracting b from optimum wb
        
        return self

    def predict(self, X):
        """Predict using the linear model.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)): test data

        Returns:
            y (ndarray, shape = (n_samples,): predicted values
        """
        return X.dot(self.w) + self.b

    def objective(self, wb, X, y):
        """Compute the loss function.

        Arguments:
            wb (ndarray, shape = (n_features + 1,)):
                concatenation of the coefficient and the bias parameters
                wb = [w, b]
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                target values.

        Returns:
            loss (float):
                the objective function evaluated on w.
        """
        X = np.hstack((X,np.ones((X.shape[0],1))))
        ybar = X.dot(wb)
        diff = y-ybar
        abs_diff = np.fabs(diff)
        loss = 0

        # Calculating loss
        for i in range(y.shape[0]):
            if(abs_diff[i] <= self.delta):
                loss += (np.power(diff[i],2*self.k))/(2*self.k)
            else:
                loss += (np.power(self.delta,2*self.k-1))*(abs_diff[i] - self.delta*(1-1/(2*self.k)))
        
        return loss

    def objective_grad(self, wb, X, y):
        """Compute the derivative of the loss function.

        Arguments:
            wb (ndarray, shape = (n_features + 1,)):
                concatenation of the coefficient and the bias parameters
                wb = [w, b]
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                target values.

        Returns:
            loss_grad (ndarray, shape = (n_features + 1,)):
                derivative of the objective function with respect to w.
        """
        
        loss_grad = np.zeros_like(wb)
        X = np.hstack((X,np.ones((X.shape[0],1))))
        ybar = X.dot(wb)
        diff = ybar - y
        abs_diff = np.fabs(diff)
        
        # Calculating gradient
        for i in range(y.shape[0]):
            if(abs_diff[i] <= self.delta):
                loss_grad += np.power(diff[i],2*self.k - 1)*X[i,:]
            elif diff[i] < 0:
                loss_grad += - np.power(self.delta,2*self.k - 1) * X[i,:]
            elif diff[i] > 0:
                loss_grad += np.power(self.delta,2*self.k - 1) * X[i,:]
        
        return loss_grad

    def get_params(self):
        """Get learned parameters for the model. Assumed to be stored in
           self.w, self.b.

        Returns:
            A tuple (w,b) where w is the learned coefficients (ndarray)
            and b is the learned bias (float).
        """
        return self.w, self.b

    def set_params(self, w, b):
        """Set the parameters of the model. When called, this
           function sets the model parameters tha are used
           to make predictions. Assumes parameters are stored in
           self.w, self.b.

        Arguments:
            w (ndarray, shape = (n_features,)): coefficient prior
            b (float): bias prior
        """
        self.w = w
        self.b = b


def main():
    np.random.seed(0)
    from sklearn.metrics import mean_squared_error
    from sklearn.linear_model import LinearRegression
    import matplotlib.pyplot as plt

    train_X = np.load('../data/q3_train_X.npy')
    train_y = np.load('../data/q3_train_y.npy')
    
    # Using RobustLinearRegression to train the data and then predict on the test data
    clf_robust = RobustLinearRegression(1,1) 
    clf_robust.set_params(np.zeros(train_X.shape[1]),1)
    clf_robust.fit(train_X,train_y)
    pred_y = clf_robust.predict(train_X)
    
    print(f'MSE of robust_linear_regression {mean_squared_error(pred_y,train_y)}')
    
    # Using Least Squares LinearRegression to train the data and then predict on the test data
    clf_least = LinearRegression()  
    clf_least.fit(train_X,train_y)
    pred_y = clf_least.predict(train_X)
    
    print(f'MSE of leastsquares_linear_regression {mean_squared_error(pred_y,train_y)}')

    # Plotting the data points and the linear fit of the above two models
    w,b = clf_robust.get_params()
    plt.scatter(train_X, train_y,color='b')
    x = range(-8, 8)
    y = [w * xi + b for xi in x]
    plt.plot(x, y,'r',linewidth=2, label='Robust Linear Regression')
    y_least = [clf_least.predict(xi) for xi in x]
    plt.plot(x, y_least,'g',linewidth=2, label='Standard Least Squares Model')
    plt.legend(loc='lower right', frameon=False)
    plt.title("Comparison of Robust and Standard Least Squares Linear Regression")
    plt.ylim(-70,60)
    fig = plt.gcf()
    fig.set_size_inches(12,8)
    plt.show()
    
if __name__ == '__main__':
    main()
