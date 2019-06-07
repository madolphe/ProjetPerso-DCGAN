from keras.datasets import mnist
import numpy as np

(X_train, _), (X_test, _) = mnist.load_data()
X = np.concatenate((X_train, X_test), axis=0)
X = np.reshape(X, (X.shape[0], 28, 28, 1))
