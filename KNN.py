import numpy as np

class KNN:
  def __init__(self, k):
    self.k = k

  def fit(self, X, Y):
    self.X_train = X
    self.Y_train = Y

  def predict(self, X):
    Y_pred = []
    for x in X :
      distances = self.euclidean_distance(x, self.X_train)
      k_indices = np.argsort(distances)[:self.k]
      k_nearest_label = self.Y_train[k_indices]
      Y_pred.append(np.bincount(k_nearest_label).argmax())
    return np.array(Y_pred)

  def euclidean_distance(self, x1, x2):
    return np.sqrt(np.sum((x1- x2)**2, axis=1))

  def evaluate (self, X, Y):
    Y_pred = self.predict(X)
    accuracy =np.sum(Y_pred == Y)/ len(Y)
    return accuracy