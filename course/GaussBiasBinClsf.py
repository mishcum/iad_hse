import numpy as np

class GaussBiasBinClsf:
    def __init__(self, P1: np.float64 = 0.5, P2: np.float64 = None, lm1: np.float64 = .0, lm2: np.float64 = .0):
        if P2 is None:
            P2 = 1 - P1
        self.P1 = P1
        self.P2 = P2
        self.lm1 = lm1
        self.lm2 = lm2 
    
    def fit(self, X_train: np.array, y_train: np.array):
        self.m_1 = np.mean(X_train[y_train == -1], axis=0)
        self.m_2 = np.mean(X_train[y_train == 1], axis=0) 

        self.cov_matrix_1 = np.cov(X_train[y_train == -1], rowvar=False)
        self.cov_matrix_2 = np.cov(X_train[y_train == 1], rowvar=False)
        return self
    
    def __clsf__(self, x: np.array) -> np.int64:
        def gaussian_pdf(x, mean, cov_matrix):
            dim = len(x)
            coef = 1 / ((2 * np.pi) ** (dim / 2) * np.linalg.det(cov_matrix) ** 0.5)
            exponent = -0.5 * (x - mean).T @ np.linalg.inv(cov_matrix) @ (x - mean)
            return coef * np.exp(exponent)
        
        p1 = gaussian_pdf(x, self.m_1, self.cov_matrix_1) * self.P1
        p2 = gaussian_pdf(x, self.m_2, self.cov_matrix_2) * self.P2
        
        p1 -= self.lm1
        p2 -= self.lm2
        
        return np.int64(-1) if p1 > p2 else np.int64(1)
    
    def predict(self, X: np.array):
        return np.array([self.__clsf__(x) for x in X])