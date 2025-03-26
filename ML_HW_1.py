
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

class KNNClassifier:
    def __init__(self, k=3, distance_metric="euclidean"):
        self.k = k
        self.distance_metric = distance_metric

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        y_pred = [self._predict(x) for x in X_test.to_numpy()]
        return np.array(y_pred)

    def _predict(self, x):
        distances = []
        for i in range(len(self.X_train)):
            if self.distance_metric == "euclidean":
                dist = euclidean_distance(x, self.X_train.iloc[i])
            elif self.distance_metric == "manhattan":
                dist = manhattan_distance(x, self.X_train.iloc[i])
            distances.append((dist, self.y_train.iloc[i]))

        distances = sorted(distances)[:self.k]
        k_neighbors = [label for _, label in distances]
        most_common = Counter(k_neighbors).most_common(1)[0][0]
        return most_common
