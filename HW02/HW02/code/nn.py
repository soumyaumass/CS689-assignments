import numpy as np
import gzip
import pickle
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class NN(nn.Module):
    """A network architecture of simultaneous localization and
       classification of objects in images.

    Arguments:
        alpha: trade-off parameter for the composite objective function.
        epochs: number of epochs for training
    """
    def __init__(self, alpha=.5, epochs=200):
        self.alpha = alpha
        self.epochs = epochs
        super(NN, self).__init__()
        self.loss = 0
        self.h1 = nn.Linear(3600,256)
        self.h2 = nn.Linear(256,64)
        self.h3p = nn.Linear(64,32)
        self.h3l = nn.Linear(64,32)
        self.cp = nn.Linear(32,1)
        self.loc = nn.Linear(32,2)

    def objective(self, X, y_class, y_loc):
        """Objective function.

        Arguments:
            X (numpy ndarray, shape = (samples, 3600)):
                Training input matrix where each row is a feature vector.
            y_class (numpy ndarray, shape = (samples,)):
                Training labels. Each entry is either 0 or 1.
            y_loc (numpy ndarray, shape = (samples, 2)):
                Training (vertical, horizontal) locations of the objects.

        Returns:
            Composite objective function value.
        """
        X = torch.tensor(X, requires_grad=False)
        y_class = torch.tensor(y_class, requires_grad=False)
        y_loc = torch.tensor(y_loc, requires_grad=False)
        
        h_out = F.relu(self.h1(X))
        h_out = F.relu(self.h2(h_out))
        h3p_out = F.relu(self.h3p(h_out))
        h3l_out = F.relu(self.h3l(h_out))
        cp_out = self.cp(h3p_out)
        loc_out = self.loc(h3l_out)
        
        cp_loss = self.alpha*nn.BCEWithLogitsLoss(reduction='sum')(cp_out.flatten().float(), y_class.float())
        local_loss = (1-self.alpha)*(torch.norm(loc_out - y_loc)**2)
        
        self.loss= cp_loss + local_loss

        return self.loss.item()

    def predict(self, X):
        """Predict class labels and object locations for samples in X.

        Arguments:
            X (numpy ndarray, shape = (samples, 3600)):
                Input matrix where each row is a feature vector.

        Returns:
            y_class (numpy ndarray, shape = (samples,)):
                predicted labels. Each entry is either 0 or 1.
            y_loc (numpy ndarray, shape = (samples, 2)):
                The predicted (vertical, horizontal) locations of the
                objects.
        """
        X = torch.tensor(X, requires_grad=False)
       
        h_out = F.relu(self.h1(X))
        h_out = F.relu(self.h2(h_out))
        h3p_out = F.relu(self.h3p(h_out))
        h3l_out = F.relu(self.h3l(h_out))
        cp_out = torch.sigmoid(self.cp(h3p_out)).flatten()
        loc_out = self.loc(h3l_out)
        
        cp_out[cp_out < 0.5] = 0
        cp_out[cp_out >= 0.5] = 1
        cp_out = cp_out
        
        y_class = cp_out.detach().numpy()
        y_loc = loc_out.detach().numpy()
        
        return y_class, y_loc

    def fit(self, X, y_class, y_loc, learning_rate=0.001):
        """Train the model according to the given training data.

        Arguments:
            X (numpy ndarray, shape = (samples, 3600)):
                Training input matrix where each row is a feature vector.
            y_class (numpy ndarray, shape = (samples,)):
                Training labels. Each entry is either 0 or 1.
            y_loc (numpy ndarray, shape = (samples, 2)):
                Training (vertical, horizontal) locations of the
                objects.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        batch_size = 200
        
        for i in range(self.epochs):
            mask = np.random.choice(X.shape[0],batch_size)
            X_batch = X[mask]
            y_class_batch = y_class[mask]
            y_loc_batch = y_loc[mask]
            optimizer.zero_grad()
            self.objective(X_batch,y_class_batch,y_loc_batch)
            self.loss.backward()
            optimizer.step()

    def get_model_params(self):
        """Get the model parameters.

        Returns:
            w1 (numpy ndarray, shape = (3600, 256)):
            b1 (numpy ndarray, shape = (256,)):
                weights and biases for FC(3600, 256)

            w2 (numpy ndarray, shape = (256, 64)):
            b2 (numpy ndarray, shape = (64,)):
                weights and biases for FC(256, 64)

            w3 (numpy ndarray, shape = (64, 32)):
            b3 (numpy ndarray, shape = (32,)):
                weights and biases for FC(64, 32)

            w4 (numpy ndarray, shape = (64, 32)):
            b4 (numpy ndarray, shape = (32,)):
                weights and biases for FC(64, 32)

            w5 (numpy ndarray, shape = (32, 1)):
            b5 (float):
                weights and biases for FC(32, 1) for the logit for
                class probability output

            w6 (numpy ndarray, shape = (32, 2)):
            b6 (numpy ndarray, shape = (2,)):
                weights and biases for FC(32, 2) for location outputs
        """
        w1 = self.h1.weight.data.numpy().T
        b1 = self.h1.bias.data.numpy()
        w2 = self.h2.weight.data.numpy().T
        b2 = self.h2.bias.data.numpy()
        w3 = self.h3p.weight.data.numpy().T
        b3 = self.h3p.bias.data.numpy()
        w4 = self.h3l.weight.data.numpy().T
        b4 = self.h3l.bias.data.numpy()
        w5 = self.cp.weight.data.numpy().T
        b5 = self.cp.bias.data.numpy().item()
        w6 = self.loc.weight.data.numpy().T
        b6 = self.loc.bias.data.numpy()

        return w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6

    def set_model_params(self, w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6):
        """Set the model parameters.

        Arguments:
            w1 (numpy ndarray, shape = (3600, 256)):
            b1 (numpy ndarray, shape = (256,)):
                weights and biases for FC(3600, 256)

            w2 (numpy ndarray, shape = (256, 64)):
            b2 (numpy ndarray, shape = (64,)):
                weights and biases for FC(256, 64)

            w3 (numpy ndarray, shape = (64, 32)):
            b3 (numpy ndarray, shape = (32,)):
                weights and biases for FC(64, 32)

            w4 (numpy ndarray, shape = (64, 32)):
            b4 (numpy ndarray, shape = (32,)):
                weights and biases for FC(64, 32)

            w5 (numpy ndarray, shape = (32, 1)):
            b5 (float):
                weights and biases for FC(32, 1) for the logit for
                class probability output

            w6 (numpy ndarray, shape = (32, 2)):
            b6 (numpy ndarray, shape = (2,)):
                weights and biases for FC(32, 2) for location outputs
        """
        
        self.h1.weight = Parameter(torch.from_numpy(w1.T), requires_grad=True)
        self.h1.bias = Parameter(torch.from_numpy(b1), requires_grad=True)
        self.h2.weight = Parameter(torch.from_numpy(w2.T), requires_grad=True)
        self.h2.bias = Parameter(torch.from_numpy(b2), requires_grad=True)
        self.h3p.weight = Parameter(torch.from_numpy(w3.T), requires_grad=True)
        self.h3p.bias = Parameter(torch.from_numpy(b3), requires_grad=True)
        self.h3l.weight = Parameter(torch.from_numpy(w4.T), requires_grad=True)
        self.h3l.bias = Parameter(torch.from_numpy(b4), requires_grad=True)
        self.cp.weight = Parameter(torch.from_numpy(w5.T), requires_grad=True)
        self.cp.bias = Parameter(torch.from_numpy(b5), requires_grad=True)
        self.loc.weight = Parameter(torch.from_numpy(w6.T), requires_grad=True)
        self.loc.bias = Parameter(torch.from_numpy(b6), requires_grad=True)

def main():
    np.random.seed(0)
    from sklearn.metrics import accuracy_score, mean_squared_error
    import matplotlib.pyplot as plt
    
    with gzip.open('../data/nn_data.pkl.gz', 'rb') as f:
        (train_X, train_y_class, train_y_loc,
         test_X, test_y_class, test_y_loc) = pickle.load(f)

    test_accuracies = []
    test_mses = []
    for alpha in torch.arange(0,1,0.05):
        model = NN(alpha,400)
        
        model.fit(train_X, train_y_class, train_y_loc)
    
        y_class_predict, y_loc_predict = model.predict(test_X)
        y_class_predict_train, y_loc_predict_train = model.predict(train_X)
        acc_test = accuracy_score(y_class_predict, test_y_class)
        test_accuracies.append(round(acc_test*100,2))
        acc_train = accuracy_score(y_class_predict_train, train_y_class)
        print(f'Test Accuracy for alpha {alpha:.2f} = {acc_test*100:.2f}%')
        print(f'Train Accuracy for alpha {alpha:.2f} = {acc_train*100:.2f}%')
        
        mse_test = mean_squared_error(y_loc_predict,test_y_loc)
        mse_train = mean_squared_error(y_loc_predict_train,train_y_loc)
        test_mses.append(round(mse_test,2))
        print(f'Test MSE for alpha {alpha:.2f} = {mse_test:.2f}')
        print(f'Train MSE for alpha {alpha:.2f} = {mse_train:.2f}')
    
    print(test_accuracies)
    print(test_mses)
    fig, ax1 = plt.subplots()
    fig.set_size_inches(14,10)
    ax1.set_title("Test Accuracy and Test MSE vs Alpha graph")
    ax1.set_xlabel('Alpha')
    ax1.set_ylabel('Test Accuracy',color='g')
    ax1.plot(np.arange(0,1,0.05), np.array(test_accuracies),'g', linewidth=2)
    ax1.set_yticks(range(50,101,5))
    ax1.set_xticks(np.arange(0,1,0.05))
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Test MSE',color='r')
    ax2.plot(np.arange(0,1,0.05), np.array(test_mses),'r', linewidth=2)
    ax2.set_ylim(5,11)
    
if __name__ == '__main__':
    main()
