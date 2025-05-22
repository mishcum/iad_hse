from collections import Counter
import numpy as np

class KNN:
    def __init__(self, k: int):
        self.k = k

    def fit(self, X, y):
        if X.shape[0] != y.shape[0]:
            raise ValueError ('The number of objects X and labels y must match')
        self.X = np.array(X) 
        self.y = np.array(y)
        return self # чтобы можно было делать knn = KNN(k).fit(X, y)
        

    def predict(self, X):
        if X.ndim != 2:
            raise ValueError('X must be a two-dimensional array')
        if X.shape[1] != self.X.shape[1]:
            raise ValueError((f"Number of features in X ({X.shape[1]}) does not match training data ({self.X.shape[1]})"))
        
        X = np.array(X)  
        
        distances = self.count_distances(X) 
        k_nearest_inds = np.argpartition(distances, self.k, axis=0)[:self.k]
        k_nearest = self.y[k_nearest_inds] # определяем категории K ближайших соседей
        
        return np.array([Counter(k).most_common(1)[0][0] for k in k_nearest.T])

    def count_distances(self, X):
        return np.linalg.norm(self.X[:, None] - X, axis=2) # добавим размерность, чтобы вичислить для всех сразу