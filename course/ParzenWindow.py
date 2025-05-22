import numpy as np

class ParzenWindow:
    def __init__(self, kernel: str, weights = None, p = 1, h = 1):
        allowed_kernels = ['gaussian', 'uniform', 'triangular', 'epanechnikov', 'biweight']
        if kernel not in allowed_kernels:
            raise ValueError(f"Invalid kernel. Allowed values are: {allowed_kernels}")
        self.kernel = kernel

        self.weights = weights

        if p is not None:
            self.p = p
        else: self.p = 1

        if p is not None:
            self.h = h
        else: self.h = 1

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        if self.weights is None:
            self.weights = np.ones(shape=X.shape[1])
        return self
    
    def predict(self, X):
        kernels = {
            'gaussian' : lambda dist, h: 1 / np.sqrt(2 * np.pi) * np.exp(- (dist / h) ** 2 / 2),
            'uniform' : lambda dist, h: 0.5 * ((dist / h) <= 1),
            'triangular' : lambda dist, h: (1 - np.abs(dist / h)) * ((dist / h) <= 1),
            'epanechnikov' : lambda dist, h: 0.75 * (1 - (dist / h) ** 2) * ((dist / h) <= 1),
            'biweight' : lambda dist, h: (15 / 16 * (1 - (dist / h) ** 2) ** 2) * ((dist / h) <= 1)
        }

        preds = []
        kernel_func = kernels[self.kernel]
        
        for x in X:
            classes = {c : 0.0 for c in np.unique(self.y)}
            distances = self.__count_distance__(x)
            values = kernel_func(distances, self.h)

            for i in range(len(self.y)):
                classes[self.y[i]] += values[i]

            preds.append(max(classes, key=classes.get))

        return np.array(preds)
        
    def __count_distance__(self, x):
        return np.sum(self.weights * (np.abs(self.X - x) ** self.p), axis=1) ** (1 / self.p)