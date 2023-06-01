import numpy as np

from src.utils import get_n_classes, label_to_onehot, onehot_to_label


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters=500):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        ##
        ###
        #### WRITE YOUR CODE HERE! 
        ###
        ##
        n_classes = get_n_classes(training_labels)
        n_features = training_data.shape[1]
        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros(n_classes)
        onehot_labels = label_to_onehot(training_labels, n_classes)

        for i in range(self.max_iters):
            z = np.dot(training_data, self.weights) + self.bias
            y_pred = self.softmax(z)
            loss = self.cross_entropy_loss(y_pred, onehot_labels)
            gradient_w = np.dot(training_data.T, (y_pred - onehot_labels)) / training_data.shape[0]
            gradient_b = np.mean(y_pred - onehot_labels, axis=0)
            self.weights -= self.lr * gradient_w
            self.bias -= self.lr * gradient_b

        pred_labels = self.predict(training_data)
        return pred_labels
    
    def predict(self, test_data):
        """
        Runs prediction on the test data.
        
        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        ##
        ###
        #### WRITE YOUR CODE HERE! 
        ###
        ##
        z = np.dot(test_data, self.weights) + self.bias
        y_pred = self.softmax(z)
        pred_labels = onehot_to_label(y_pred)
        return pred_labels
    
    def softmax(self, z):
        """
        Softmax activation function.

        Arguments:
            z (array): linear transformation
        Returns:
            (array): probability estimates
        """
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    

    def cross_entropy_loss(self, y_pred, onehot_labels):
        """
        Computes cross-entropy loss.

        Arguments:
            y_pred (array): predicted probability estimates
            onehot_labels (array): one-hot encoded labels
        Returns:
            (float): cross-entropy loss
        """
        loss = -np.mean(onehot_labels * np.log(y_pred + 1e-8))
        return loss