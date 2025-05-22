import numpy as np

class GaussBiasClsf:
    def __init__(self, n_classes: int, P: np.array = None, lm: np.array = None):
        if P is None:
            self.P = np.array([1.0 / n_classes] * n_classes)
        else:
            self.P = P
        if lm is None: 
            self.lm = np.zeros(n_classes)
        else:
            self.lm = lm
        self.n_classes = n_classes
        
    def fit(self, X_train: np.array, y_train: np.array):
        self.classes = np.unique(y_train)
        self.means = np.zeros((self.n_classes, X_train.shape[1]))
        self.cov_matrixes = [np.zeros((X_train.shape[1], X_train.shape[1])) for _ in range(self.n_classes)]
        
        for i, cls in enumerate(self.classes):
            class_samples = X_train[y_train == cls]
            self.means[i] = np.mean(class_samples, axis=0)
            self.cov_matrixes[i] = np.cov(class_samples, rowvar=False)
        return self
    
    def __clsf__(self, x: np.array) -> np.int64:
        def gaussian_pdf(x, mean, cov_matrix):
            dim = len(x)
            cov_matrix = cov_matrix + np.eye(dim) * 1e-6
            coef = 1 / ((2 * np.pi) ** (dim / 2) * np.linalg.det(cov_matrix) ** 0.5)
            diff = x - mean
            exponent = -0.5 * diff.T @ np.linalg.inv(cov_matrix) @ diff
            return coef * np.exp(exponent)
        
        probs = np.array([gaussian_pdf(x, mean_v, cov_matrix) 
                         for mean_v, cov_matrix in zip(self.means, self.cov_matrixes)])
        
        probs = probs * self.P - self.lm
        
        return self.classes[np.argmax(probs)]
    
    def predict(self, X: np.array):
        return np.array([self.__clsf__(x) for x in X])