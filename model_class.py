import numpy as np


class LinearRegressionModel:

    def __init__(self, weights, bias, mean, std, features):
        self.weights = weights
        self.bias = bias
        self.mean = mean
        self.std = std
        self.features = features

    def predict(self, X):

        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Scale using training mean and std
        X_scaled = (X - self.mean) / self.std

        # Linear regression
        predictions = np.dot(X_scaled, self.weights) + self.bias

        return predictions