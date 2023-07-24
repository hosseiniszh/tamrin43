import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


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
  
if __name__ == '__main__':
    iris = load_iris()
    X = iris.data
    Y = iris.target

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2)

    knn = KNN(3)
    knn.fit(X_train, Y_train)
    accuracy = knn.evaluate(X_test, Y_test)
    print("KNN (k=3) Accuracy:", accuracy)

    knn = KNN(5)
    knn.fit(X_train, Y_train)
    accuracy = knn.evaluate(X_test, Y_test)
    print("KNN (k=5) Accuracy:", accuracy)

    knn = KNN(7)
    knn.fit(X_train, Y_train)
    accuracy = knn.evaluate(X_test, Y_test)
    print("KNN (k=7) Accuracy:", accuracy)

    knn_sklearn = KNeighborsClassifier(n_neighbors=3)
    knn_sklearn.fit(X_train, Y_train)
    accuracy = knn_sklearn.score(X_test, Y_test)
    print("KNN (sklearn, k=3) Accuracy:", accuracy)

    # Calculate confusion matrix for KNN (k=3)
    knn = KNN(3)
    knn.fit(X_train, Y_train)
    Y_pred_k3 = knn.predict(X_test)
    cm_k3 = confusion_matrix(Y_test, Y_pred_k3)
    print("Confusion Matrix (KNN, k=3):")
    print(cm_k3)

    # Calculate confusion matrix for KNN (k=5)
    knn = KNN(5)
    knn.fit(X_train, Y_train)
    Y_pred_k5 = knn.predict(X_test)
    cm_k5 = confusion_matrix(Y_test, Y_pred_k5)
    print("Confusion Matrix (KNN, k=5):")
    print(cm_k5)

    # Calculate confusion matrix for KNN (k=7)
    knn = KNN(7)
    knn.fit(X_train, Y_train)
    Y_pred_k7 = knn.predict(X_test)
    cm_k7 = confusion_matrix(Y_test, Y_pred_k7)
    print("Confusion Matrix (KNN, k=7):")
    print(cm_k7)