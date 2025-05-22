import numpy as np

class BiasBinClsf:
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

        self.Dx_1 = np.var(X_train[y_train == -1], axis=0)
        self.Dx_2 = np.var(X_train[y_train == 1], axis=0)
        return self
    
    def __clsf__(self, x: np.array) -> np.int64:
        p1 = (1 / (2 * np.pi * np.sqrt(np.prod(self.Dx_1)))) * np.exp(np.sum(-(x - self.m_1) ** 2 / (2 * self.Dx_1)))
        p2 = (1 / (2 * np.pi * np.sqrt(np.prod(self.Dx_2)))) * np.exp(np.sum(-(x - self.m_2) ** 2 / (2 * self.Dx_2)))

        p1 *= self.P1
        p2 *= self.P2

        p1 -= self.lm1
        p2 -= self.lm2

        return np.int64(-1) if p1 > p2 else np.int64(1)

    
    def predict(self, X: np.array):
        return np.array([self.__clsf__(x) for x in X])
    