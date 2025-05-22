import numpy as np

class Node:
    def __init__(self, feature=None, value=None, children=None, class_label=None):
        self.feature = feature
        self.value = value
        self.children = children if children is not None else {}
        self.class_label = class_label

class DecisionTreeID3:
    def __init__(self):
        self.root = None

    def entropy(self, y : np.ndarray):
        entropy = 0
        for u in np.unique(y):
            p = np.mean(y == u)
            entropy -= p * np.log2(p + 1e-10)
        return entropy
    
    def information_gain(self, X : np.ndarray, y : np.ndarray, feature_idx):
        total_entropy = self.entropy(y)
        values, counts = np.unique(X[:, feature_idx], return_counts=True)
        weighted_entropy = 0
        for val, count in zip(values, counts):
            mask = X[:, feature_idx] == val
            weighted_entropy += (count / len(y)) * self.entropy(y[mask])
        return total_entropy - weighted_entropy
    
    def best_feature(self, X : np.ndarray, y : np.ndarray, used_features):
        best_gain, best_feature = -1, None
        for feature_idx in range(X.shape[1]):
            if feature_idx not in used_features:
                gain = self.information_gain(X, y, feature_idx)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
        return best_feature
    
    def build_tree(self, X : np.ndarray, y : np.ndarray, used_features=None):
        if used_features is None:
            used_features = set()

        if len(np.unique(y)) == 1:
            return Node(class_label=y[0])

        if X.shape[1] == len(used_features):
            values, counts = np.unique(y, return_counts=True)
            return Node(class_label=values[np.argmax(counts)])
        
        best_feature_idx = self.best_feature(X, y, used_features)
        if best_feature_idx is None:
            values, counts = np.unique(y, return_counts=True)
            return Node(class_label=values[np.argmax(counts)])
        
        node = Node(feature=best_feature_idx)
        used_features.add(best_feature_idx)

        vals = np.unique(X[:, best_feature_idx])
        for val in vals:
            mask = X[:, best_feature_idx] == val
            if len(y[mask]) == 0:
                values, counts = np.unique(y, return_counts=True)
                child = Node(class_label=values[np.argmax(counts)])
            else:
                child = self.build_tree(X[mask], y[mask], used_features.copy())
            node.children[val] = child

        return node
    
    def fit(self, X : np.ndarray, y : np.ndarray):
        self.root = self.build_tree(X, y)
        return self
    
    def _predict_one(self, x : np.ndarray, node : Node):
        if node.class_label is not None:
            return node.class_label
        feature_val = x[node.feature]
        # Если значения нет в детях, возвращаем первого ребенка (как запасной вариант)
        return self._predict_one(x, node.children.get(feature_val, list(node.children.values())[0]))
        
    def predict(self, X : np.ndarray):
        return np.array([self._predict_one(x, self.root) for x in X])