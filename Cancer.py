import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = []
        for x in X:
            distances = self.euclidean_distance(x, self.X_train)
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            y_pred.append(np.bincount(k_nearest_labels).argmax())
        return np.array(y_pred)

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        return accuracy

if __name__ == '__main__':
    breast_cancer = load_breast_cancer()
    X = breast_cancer.data
    y = breast_cancer.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    knn = KNN(3)
    knn.fit(X_train, y_train)
    accuracy_k3 = knn.evaluate(X_test, y_test)
    print("Accuracy (k=3):", accuracy_k3)

    knn = KNN(5)
    knn.fit(X_train, y_train)
    accuracy_k5 = knn.evaluate(X_test, y_test)
    print("Accuracy (k=5):", accuracy_k5)

    knn = KNN(7)
    knn.fit(X_train, y_train)
    accuracy_k7 = knn.evaluate(X_test, y_test)
    print("Accuracy (k=7):", accuracy_k7)

    knn = KNN(3)
    knn.fit(X_train, y_train)
    y_pred_k3 = knn.predict(X_test)
    cm_k3 = confusion_matrix(y_test, y_pred_k3)
    print("Confusion Matrix (k=3):\n", cm_k3)
