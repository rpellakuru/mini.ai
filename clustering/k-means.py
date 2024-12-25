import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    def find_closest_centroids(self, X, centroids):
        centroids_idx = np.zeros(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            min_l = float('inf')
            for j in range(centroids.shape[0]):
                d = (X[i] - centroids[j]) ** 2
                d_sqr = np.sum(d, axis = 0)
                if d_sqr < min_l:
                    centroids_idx[i] = j
                    min_l = d_sqr 
        return centroids_idx


    def find_closest_centroids_2(self, X, centroids):
        centroids_idx = np.zeros(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            distance = []
            for j in range(centroids.shape[0]):
                norm_ij = np.linalg.norm(X[i] - centroids[j])
                distance.append(norm_ij)
            centroids_idx[i] = np.argmin(distance)
        return centroids_idx


    # ----------

    def compute_centroids(self, X, centroids_idx, K):
        _, n = X.shape
        centroids = np.zeros((K, n), dtype=int)
        for i in range(K):
            points = X[centroids_idx == i]
            centroids[i] = np.mean(points, axis = 0)
        return centroids

    def run_kmeans(self, X, initial_centroids, max_iters=0):
        centroids = initial_centroids
        for _ in range(max_iters):
            idx = self.find_closest_centroids_2(X, centroids)
            centroids = self.compute_centroids(X, idx, initial_centroids.shape[0])
        return idx, centroids


    def start_clustering(self, X, K):
        randomize_inputs = np.random.permutation(X.shape[0])
        initial_centroids = X[randomize_inputs[:K]]
        idx, centroids = self.run_kmeans(X, initial_centroids, 100)
        plt.figure(figsize=(8, 6))
        for cluster in range(initial_centroids.shape[0]):
            cluster_points = X[idx == cluster]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label = f'Cluster {cluster}')
        plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='black', marker='x', label='center')
        plt.show()


kmeans = KMeans()
K = 3
X = np.random.randint(0, 100, (1000, 2))
kmeans.start_clustering(X, K)


