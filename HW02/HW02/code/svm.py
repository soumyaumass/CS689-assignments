import numpy as np
import gzip
import pickle


class SVM:
    """SVC with subgradient descent training.

    Arguments:
        C: regularization parameter (default: 1)
        iterations: number of training iterations (default: 500)
    """
    def __init__(self, C=1, iterations=500):
        self.C = C
        self.iterations = iterations

    def fit(self, X, y, learning_rate=0.0009, batch_size=200):
        """Fit the model using the training data.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                Training target. Each entry is either -1 or 1.
        """
        
        self.set_model(np.zeros(X.shape[1]), 0)
        
        for i in range(self.iterations):
            mask = np.random.choice(X.shape[0], batch_size)
            X_batch = X[mask]
            y_batch = y[mask]
            subgrad_w, subgrad_b = self.subgradient(X_batch, y_batch)
            self.w = self.w - learning_rate * subgrad_w
            self.b = self.b - learning_rate * subgrad_b
        

    def objective(self, X, y):
        """Compute the objective function for the SVM.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                Training target. Each entry is either -1 or 1.

        Returns:
            obj (float): value of the objective function evaluated on X and y.
        """
        
        error = 1 - y*(np.matmul(X,self.w) + self.b)
        error[error<0] = 0
        
        obj = np.sum(self.C*(error)) + self.w.T.dot(self.w)
        
        return obj

    def subgradient(self, X, y):
        """Compute the subgradient of the objective function.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)):
                Training input matrix where each row is a feature vector.
            y (ndarray, shape = (n_samples,)):
                Training target. Each entry is either -1 or 1.

        Returns:
            subgrad_w (ndarray, shape = (n_features,)):
                subgradient of the objective function with respect to
                the coefficients of the linear model.
            subgrad_b (float):
                subgradient of the objective function with respect to
                the bias term.
        """
        
        error = 1 - y*(np.matmul(X,self.w) + self.b)
        
        mask = np.zeros_like(error)
        mask[error<0] = 0
        mask[error>0] = 1
   
        subgrad_w = np.sum(-X*(y.reshape(y.shape[0],1))*(mask.reshape(mask.shape[0],1)),axis=0)
        subgrad_b = np.sum(-y*mask)
                
        subgrad_w *= self.C
        subgrad_b *= self.C
        subgrad_w += 2* self.w

        return subgrad_w, subgrad_b

    def predict(self, X):
        """Predict class labels for samples in X.

        Arguments:
            X (ndarray, shape = (n_samples, n_features)): test data

        Returns:
            y (ndarray, shape = (n_samples,):
                Predictions with values of -1 or 1.
        """
        y_pred = np.matmul(X,self.w) + self.b
        y_pred[y_pred >= 0] = 1
        y_pred[y_pred < 0] = -1
        return y_pred

    def get_model(self):
        """Get the model parameters.

        Returns:
            w (ndarray, shape = (n_features,)):
                coefficient of the linear model.
            b (float): bias term.
        """
        return self.w, self.b

    def set_model(self, w, b):
        """Set the model parameters.

        Arguments:
            w (ndarray, shape = (n_features,)):
                coefficient of the linear model.
            b (float): bias term.
        """
        self.w, self.b = w, b

def main():
    np.random.seed(0)
    
    from sklearn.metrics import accuracy_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model.logistic import _logistic_loss

    with gzip.open('../data/svm_data.pkl.gz', 'rb') as f:
        train_X, train_y, test_X, test_y = pickle.load(f)

    cls = SVM(1,600)
    cls.fit(train_X, train_y)
    y_pred_train=cls.predict(train_X)
    obj = cls.objective(train_X, train_y)
    y_pred=cls.predict(test_X)
    w_svm, b_svm = cls.get_model()
    acc_test = accuracy_score(test_y,y_pred)
    acc_train = accuracy_score(train_y,y_pred_train)
    print(f'SVC Objective= {obj:.2f}')
    print(f'SVC Test Accuracy = {acc_test*100:.2f}%')
    print(f'SVC Train Accuracy = {acc_train*100:.2f}%')
    
    cls_logistic = LogisticRegression()
    cls_logistic.fit(train_X,train_y)
    y_pred = cls_logistic.predict(train_X)
    y_pred_test = cls_logistic.predict(test_X)
    acc_logistic = accuracy_score(train_y, y_pred)
    acc_logistic_test = accuracy_score(test_y, y_pred_test)
    print(f'Logistic Train Accuracy = {acc_logistic*100:.2f}%')
    print(f'Logistic Test Accuracy = {acc_logistic_test*100:.2f}%')

    w_lr = cls_logistic.coef_.reshape(w_svm.shape)
    b_lr = cls_logistic.intercept_.reshape(b_svm.shape)
    obj = _logistic_loss(w_lr, train_X, train_y,alpha=1)
    print(f'Logistic Objective = {obj:.2f}')

    cls = SVM()
    cls.set_model(w_lr,b_lr)
    obj = cls.objective(train_X, train_y)
    print(f'SVC Objective at w_lr, b_lr = {obj:.2f}')
    y_pred_train = cls.predict(train_X)
    acc_train = accuracy_score(train_y, y_pred_train)
    print(f'SVC Train Accuracy at w_lr, b_lr = {100*acc_train:.2f}%')

if __name__ == '__main__':
    main()
