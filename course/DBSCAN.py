import numpy as np

class DBSCAN:

    def __init__(self, eps: float, min_samples: int, distance_func: str, p: float = 1):
        self.eps = eps
        self.min_samples = min_samples
        self.distance_func = distance_func
        self.p = p

    def fit(self, X: np.ndarray):
        self.X_ = np.array(X)
        n_samples = self.X_.shape[0]
        self.clusters_ = np.array([-1] * n_samples)
        self.visited_ = [False] * n_samples
        cluster_id = 0

        for i in range(n_samples):
            if self.visited_[i]:
                continue
            self.visited_[i] = True
            
            neighbours = self._region_query(i)
            if len(neighbours) < self.min_samples:
                self.clusters_[i] = -1  # шум
            else:
                self._expand_cluster(i, neighbours, cluster_id)
                cluster_id += 1

    def _region_query(self, idx: int):
        distances = self._count_distance(idx)
        return np.where(distances <= self.eps)[0]

    def _count_distance(self, idx: int):
        distance_funcs = {
            'euclidean': self.euclidean,
            'manhattan': self.manhattan,
            'minkowski': self.minkowski
        }
        distances = [distance_funcs[self.distance_func](x, self.X_[idx]) for x in self.X_]
        return np.array(distances)

    def _expand_cluster(self, point_idx, neighbours, cluster_id):
        self.clusters_[point_idx] = cluster_id
        to_visit = list(neighbours)
        
        while to_visit:
            current_point_idx = to_visit.pop()
            if not self.visited_[current_point_idx]:
                self.visited_[current_point_idx] = True
                current_neighbours = self._region_query(current_point_idx)
                if len(current_neighbours) >= self.min_samples:
                    to_visit.extend(current_neighbours)
            if self.clusters_[current_point_idx] == -1:
                self.clusters_[current_point_idx] = cluster_id

    def euclidean(self, a, b):
        return np.linalg.norm(a - b)

    def manhattan(self, a, b):
        return np.sum(np.abs(a - b))

    def minkowski(self, a, b):
        return np.sum(np.abs(a - b) ** self.p) ** (1 / self.p)